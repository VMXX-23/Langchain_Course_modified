#MCP server implementation to add 2 nos
# math_server.py
from mcp.server.fastmcp import FastMCP
print("Starting math server...")
mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    print("Server ready")
    mcp.run(transport="stdio")