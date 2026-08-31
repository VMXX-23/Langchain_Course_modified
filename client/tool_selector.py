from typing import List
from langchain_core.tools import BaseTool

# Mandatory core tools that should ALWAYS be available to the agent !
CORE_TOOL_NAMES = {"ask_user"}

# Mapping keywords to specific tool names (for fast explicitly target matching)
KEYWORD_TOOL_MAP = {
    # Weather keywords
    "weather": ["get_weather"],
    "temp": ["get_weather"],
    "temperature": ["get_weather"],
    "climate": ["get_weather"],
    "humidity": ["get_weather"],
    "wind": ["get_weather"],
    "rain": ["get_weather"],
    "forecast": ["get_weather"],
    # Math symbols & operators
    "+": ["add"],
    "-": ["subtract"],
    "*": ["multiply"],
    "/": ["divide"],
    "^": ["power"],
    "%": ["modulo"],
    # Math keywords
    "add": ["add"],
    "plus": ["add"],
    "sum": ["add", "sum_of_digits"],
    "subtract": ["subtract"],
    "minus": ["subtract"],
    "multiply": ["multiply"],
    "times": ["multiply"],
    "product": ["multiply"],
    "divide": ["divide"],
    "ratio": ["divide"],
    "power": ["power"],
    "exponent": ["power"],
    "root": ["root"],
    "sqrt": ["root"],
    "cube root": ["root"],
    "factorial": ["factorial"],
    "modulo": ["modulo"],
    "remainder": ["modulo"],
    "prime": ["is_prime", "prime_factors"],
    "gcd": ["gcd"],
    "lcm": ["lcm"],
    "fibonacci": ["fibonacci"],
    "binary": ["decimal_to_binary"]
}

def select_tools(user_input: str, tools: List[BaseTool],  default_fallback: bool = True) -> List[BaseTool]:
    """
    Selects tools based on user input keywords.
    
    Args:
        tools (List[BaseTool]): List of available tools.
        user_input (str): User input string to analyze for keywords.
        
    Returns:
        List[BaseTool]: List of selected tools based on keywords.
    """
    selected_tools = set(CORE_TOOL_NAMES)  # core tools always included
    user_input_lower = user_input.lower()
    
    # Check for keywords in user input
    for keyword, tool_names in KEYWORD_TOOL_MAP.items():
        if keyword in user_input_lower:
            selected_tools.update(tool_names)
    for tool in tools:
        if tool.name.lower() in user_input_lower:
            selected_tools.add(tool.name)
            
    # Filter the actual BaseTool objects matching the selected names
    matched_tools = [t for t in tools if t.name in selected_tools]
    
    if len(matched_tools) == len(CORE_TOOL_NAMES) and default_fallback:
        print("⚠️ [Tool Catcher] No specific keywords matched. Defaulting to standard tool subset : + - *")
        # Return basic math + weather as general fallback
        fallback_names = CORE_TOOL_NAMES.union({"get_weather", "add", "multiply", "subtract"})
        matched_tools = [t for t in tools if t.name in fallback_names]
    
    print(f"🎯 [Tool Catcher] Filtered {len(tools)} total tools down to {len(matched_tools)}: {[t.name for t in matched_tools]}")
    return matched_tools