#Weather server MCP implementation to get weather data

# weather_server.py
from typing import List
from mcp.server.fastmcp import FastMCP

print("Starting weather server...")
mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in Hyderabad"

if __name__ == "__main__":
    print("Weather Server ready")
    mcp.run(transport="sse")