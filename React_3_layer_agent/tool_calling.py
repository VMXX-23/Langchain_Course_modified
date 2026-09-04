from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI
#Just checks for constant price list and discount tiers, no database or API calls are made in this example.

MAX_ITERATION = 10

#model = "gemma3:270m"


#TOOLS
@tool
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

@tool
def apply_discount(price: float , discount_tier : str) -> float:
    """
    Apply a discount to the price based on the discount tier.
    Available discount tiers: gold( or tier1), silver( or tier2), bronze( or tier3) 
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


#Agent Loop
@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_prod_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    #chat_model = init_chat_model(model=model,temperature=0)
    chat_model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0,
)
    #Include tool descriptions in the system message
    chat_model_tools = chat_model.bind_tools(tools)
    
    #System Message 
    messages = [
    SystemMessage(
        content="""
        You are an intelligent pricing assistant responsible for retrieving product prices and calculating discounts.

        Always:
        1. Retrieve product information using the available tools.
        2. Validate that the requested product exists.
        3. Validate that the requested discount tier exists.
        4. Never assume missing information.
        5. Never invent prices or discounts.
        6. Perform all calculations accurately.
        7. Explain how the final price was calculated.

        If the required information cannot be found, state exactly what is missing instead of making assumptions.

        Respond using this format:

        Product:
        Original Price:
        Discount Tier:
        Discount:
        Discount Amount:
        Final Price:
        Source:
        """
        ),
    HumanMessage(content=question),
    ]
    
    for iteration in range(1, MAX_ITERATION+1):
        print(f"Iteration {iteration}:")
        ai_message = chat_model_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        #Ai message containing the observation from the tool call
        messages.append(ai_message)
        #If no tool calls are made, return the AI's response
        if not tool_calls:
            print('No tool calls made. Returning AI response:')
            content = ai_message.content
            if isinstance(content, list):
                text_blocks = [
                    block["text"]
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "\n".join(text_blocks)

            return str(content)
        
        
        #Process only the first tool call for simplicity
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"Tool called: {tool_name} with args: {tool_args}")
        
        tools_to_use = tools_dict.get(tool_name)
        if tools_to_use is None:
            raise ValueError(f"Tool {tool_name} not found.")
        
        observation = tools_to_use.invoke(tool_args)
        
        print(f"Observation from tool {tool_name}: {observation}")
        

        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
  

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

    