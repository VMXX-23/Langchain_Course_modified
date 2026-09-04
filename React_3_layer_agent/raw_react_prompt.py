import json
import re,os , inspect
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langsmith import traceable
from google import genai
from google.genai import types

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found! Check your .env file or path.")

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

MAX_ITERATION = 10
MODEL = "gemini-3.5-flash"
#TOOLS
@traceable(name="Get Product Price Tool")
def get_prod_price(product_id: str) -> str:
    """
    Returns the price of a product.

    Args:
        product_id: Product name.
        Valid values:
        - laptop
        - keyboard
        - mouse pad

    Returns:
        float if found,
        otherwise "Product not found".
    """
    print(f"Fetching price for product: {product_id}")
    
    # Example Simulate fetching product price from a database or API
    product_prices = {
        "mouse pad": 30.0,
        "keyboard": 1500.0,
        "laptop": 5000.0,
    }
    return product_prices.get(product_id.lower(), "Product not found")

@traceable(name="Apply Discount Tool")
def apply_discount(price: float , discount_tier : str) -> float:
    """
    Applies a discount.

    Args:
        price: Numeric product price.
        discount_tier:
            tier1 = Gold
            tier2 = Silver
            tier3 = Bronze

    Returns:
        Discounted price.
    """
    print(f"Applying discount for price: {price} with tier: {discount_tier}")
    
    discount_rates = {
        "tier1": 0.30,  # 30% discount
        "tier2": 0.50,  # 50% discount
        "tier3": 0.80,  # 80% discount
        "tier1": 0.30,
        "bronze": 0.30,
        "tier2": 0.50,
        "silver": 0.50,  # Additional maps added just in case the user uses the tier names instead of the tier numbers
        "tier3": 0.80,
        "gold": 0.80,
    }
    
    discount_rate = discount_rates.get(discount_tier, 0)
    discounted_price = price * (1 - discount_rate)
    
    return round(discounted_price, 2)

#Map tool name to functions
tools_dict = {
        "get_prod_price": get_prod_price,
        "apply_discount": apply_discount
    }

# Convert Python tools into plain-text prompt descriptions using inspect.
# Used by the raw LLM implementation instead of native function calling.
def get_tool_descrp(tools_dict):
    tool_descrp = []
    for tool_name, tool_func in tools_dict.items():
        orignal_func = getattr(tool_func, "__wrapped__", tool_func) #Wrapped attribute is added by the @traceable decorator, so we need to get the original function for accurate signature and docstring.
        signature = inspect.signature(orignal_func)
        docstring = inspect.getdoc(orignal_func) or "No description available."
        tool_descrp.append(f"{tool_name}{signature} \nDesciption: {docstring}\n")
    return "\n".join(tool_descrp)

tool_names =",".join(tools_dict.keys())
tool_descriptions = get_tool_descrp(tools_dict)
#React prompt
react_prompt = """ 
You are a product pricing assistant.
Requirements:
- No guessing or arithmetic.
- Price → `get_prod_price` only.
- Discount → `apply_discount` only.
- Use exact tool outputs.
- Max 1 tool/step.
- Product missing → stop.
- Tier missing → ask user.
- Continue after Observation.
- Never generate another Question.
- Finish with `Final Answer:`.

Tools:

{tool_descriptions}

After every Observation continue reasoning.

Format:

Question: ...
Thought: ...
Action: [{tool_names}]
Action Input: ...
Observation: ...

Thought: I now know the final answer.
Final Answer: ...
Begin.
Question: {question}
Thought:
"""
api_calls = 0

#LLM chat
@traceable(name="Gemini Chat", run_type="llm")
def gemini_chat_traced(prompt: str):
    global api_calls
    api_calls += 1
    print(f"\nGemini API Call #{api_calls}")

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        #''' config={ "tools": tools_for_llm }'''
        config=types.GenerateContentConfig( temperature = 0,
                                           stop_sequences=["Observation:"],
       # tools=tools_dict,
       )
    )
    #print(f"Gemini response: {response}")
    return response

#Agent loop
@traceable(name="Agent loop")
def run_agent(question: str):
    print(f"Question: {question}") #Debug check for question
    
    #Single prompt string to replace SYSTEM/ USER messages
    prompt = react_prompt.format( tool_descriptions=tool_descriptions, tool_names=tool_names, question=question,)
    history = ""
    
    for iteration in range(1, MAX_ITERATION):  # Limit to MAX iterations
        print(f"Iteration {iteration}:")
        full_prompt = prompt + history
        response = gemini_chat_traced(full_prompt)
        output = response.text
        #or output = response.candidates[0].content.parts[0].text 
        
        print(f"LLM Output: {output}")  # Debug check for LLM output
        
        print(f"Checking for final answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer found: {final_answer}")
            return final_answer
        
        #Parse tool calls from Raw text with Re, check with LLM prior to usage
        print(f" Looking for Action and Action Input in LLM output...")
        
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break
        
        tool_name = action_match.group(1).strip()
        #tool_input_raw = action_input_match.group(1).strip()
        args = json.loads(action_input_match.group(1).strip())
        #Execute tools and get the tool outputs directly:
        #print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        #raw_args = [x.strip() for x in tool_input_raw.split(",")]
        #args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]
        
        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools_dict:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools_dict.keys())}"
        else:
            observation = str(tools_dict[tool_name](**args))

        print(f"  [Tool Result] {observation}")
        
        #Append the LLM output + observation to the scratch pad for the next iteration
        history += output
        history += f"\nObservation: {observation}\nThought:"
    
    print("ERROR: Max iterations reached without a final answer")
    return None

def main():
    print("Welcome to the Product Price and Discount Agent!")
    user_input_txt = (
        "What is the product and discount tier you want to know about? (e.g., 'prod_001 tier_1'):\n"
    " Discount Tiers:\n"
    "  -> tier1 : 30% Discount\n"
    "  -> tier2 : 50% Discount\n"
    "  -> tier3 : 80% Discount\n"
    ":>")
    user_question = input(user_input_txt)
    result = run_agent(user_question)  
    print(f"Result: {result}")
    
if __name__ == "__main__":
    main()
