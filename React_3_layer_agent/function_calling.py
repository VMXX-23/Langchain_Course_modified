import os 
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langsmith import traceable
#Need to use Google SDK for function calling
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
#import ollama
# Just checks for constant price list and discount tiers, no database or API calls are made in this example.

MAX_ITERATION = 10

#This model does not support Function calling
#model = "gemma3:270m"
#Need to use Google SDK not Gemini
'''
chat_model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
    temperature=0,
    max_retries = 2
)
'''
SYSTEM_PROMPT = """
You are a helpful shopping assistant.

You have access to a product catalog tool and a discount tool.

STRICT RULES:
1. NEVER guess or assume any product price.
2. You MUST call get_product_price first to retrieve the actual price.
3. Only call apply_discount AFTER receiving the price from get_product_price.
4. Pass the exact price returned by get_product_price.
5. NEVER calculate discounts yourself.
6. If the user does not specify a discount tier, ask for it.
"""
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found! Check your .env file or path.")

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

#Toools for LLM is not required for Google SDK as JSON
tools_for_llm = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_prod_price",
                description=(
                    "Only the product name. Examples: "
                    "'laptop', 'keyboard', 'mouse pad'. "
                    "Never pass 'laptop gold' or 'keyboard silver'."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "product_id": {
                            "type": "STRING",
                            "description": "Product name"
                        }
                    },
                    "required": ["product_id"],
                },
            ),
            types.FunctionDeclaration(
                name="apply_discount",
                description="Apply a discount.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "price": {
                            "type": "NUMBER"
                        },
                        "discount_tier": {
                            "type": "STRING"
                        }
                    },
                    "required": ["price", "discount_tier"],
                },
            ),
        ]
    )
]
#Chat model with tools: GEMMA, GEMINI
#chat_model = chat_model.bind_tools(tools_for_llm)

#TOOLS
@traceable(name="Get Product Price Tool")
def get_prod_price(product_id: str) -> str:
    """
    Get the price of a product by its ID.
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
    Apply a discount to the price based on the discount tier.
    Available discount tiers: Gold(tier1), Silver(tier2), Bronze(tier3) 
    """
    print(f"Applying discount for price: {price} with tier: {discount_tier}")
    
    discount_rates = {
        "tier1": 0.30,  # 30% discount
        "tier2": 0.50,  # 50% discount
        "tier3": 0.80,  # 80% discount
    }
    
    discount_rate = discount_rates.get(discount_tier, 0)
    discounted_price = price * (1 - discount_rate)
    
    return round(discounted_price, 2)

#Model 
@traceable(name="Gemini Chat", run_type="llm")
def gemini_chat_traced(messages):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        #''' config={ "tools": tools_for_llm }'''
        config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=tools_for_llm,)
    )
    return response


#Helper function to trace Gemini calls
@traceable(name="Gemini Model Call")
def run_agent(question: str):
    tools_dict = {
        "get_prod_price": get_prod_price,
        "apply_discount": apply_discount
    }
    print(f"Question: {question}")
    print("=" * 60)

    '''OLLAMA : messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]'''
    #GEN AI SDK 
    messages = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=question)
            ],
        )
    ]

    for iteration in range(1, MAX_ITERATION + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Difference 5: gemini_chat_traced() directly instead of llm_with_tools.invoke()
        response = gemini_chat_traced(messages)
        #ai_message = response.message
        #Debug
        #print(f"Response: {response}")
        #For google gen ai SDK
        parts = response.candidates[0].content.parts
        function_call = None
        for part in parts:
            if getattr(part, "function_call", None):
                function_call = part.function_call
                break
 #Get final answer from Google SDK
        if function_call is None:
            # Final answer
            text = "".join(
                part.text
                for part in parts
                if getattr(part, "text", None)
            )
            #print(text)
            return text
        #tool_calls = ai_message.tool_calls
        #Gemini syntax
        #tool_calls = ai_message.tool_calls[0]
        # If no tool calls, this is the final answer
        '''if not tool_calls: #WORKS FOR OLLAMA
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
            '''
        # Process only the FIRST tool call — force one tool per iteration
        #tool_call = tool_calls[0] "not required for Gemini"
        # Difference 6: (.function.name) instead of dict access (.get("name")): Attr accessed differently
        ''' tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments '''
       
       
        '''WORKS FOR OLLAMA
           tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
'''
        tool_name = function_call.name
        tool_args = dict(function_call.args)

        #print(f"[Tool Selected] {tool_name}")
        #print(f"Arguments: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Unknown tool: {tool_name}")



        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)


        #print(f"  [Tool Result] {observation}")

        messages.append(response.candidates[0].content)
        
        ''' WORKS FOR GEMMA messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        ) '''
        messages.append(
            types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": observation},
                    )
                ],
            )
        )

    print("ERROR: Max iterations reached without a final answer")
    return None





def main():
    print("Welcome to the Product Price and Discount Agent!")
    user_question = input("What is the product : and discount tier you want to know about? (e.g., 'prod_001 tier_1'): ")
    result = run_agent(user_question)  
    print(f"Result: {result}")
    
    
if __name__ == "__main__":
    main()

#SWITCH TO GEMINI TO SUPPORT FUNCTION CALLING
    