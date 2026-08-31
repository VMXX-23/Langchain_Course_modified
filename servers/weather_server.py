#Weather server MCP implementation to get weather data
import httpx
# weather_server.py
from typing import List
from mcp.server.fastmcp import FastMCP

print("Starting weather server...")
mcp = FastMCP("Weather")
@mcp.tool()
async def get_weather(location: str) -> str:
    """"Get real-time weather information for any given location or city.
    
    Returns structured weather details including a raw integer temperature field 
    intended for direct use in downstream mathematical tool chaining.
    """
    print(f"Fetching weather data for: {location}")
    
    # Custom headers prevent APIs from blocking automated httpx clients
    headers = {"User-Agent": "MCP-Weather-Agent/1.0"}
    
    async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
        try:
            # 1. Geocoding: Convert city name into latitude and longitude
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
            geo_res = await client.get(geo_url)
            geo_data = geo_res.json()

            if not geo_data.get("results"):
                return f"Could not find coordinates for location: '{location}'."

            city_info = geo_data["results"][0]
            lat = city_info["latitude"]
            lon = city_info["longitude"]
            city_name = city_info.get("name", location)
            country = city_info.get("country", "")

            # 2. Weather Fetching: Query Open-Meteo API
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m"
            weather_res = await client.get(weather_url)
            weather_data = weather_res.json()
            
            current = weather_data.get("current", {})
            if not current:
                return f"Could not retrieve weather data for {city_name}."
            
            print(f"Weather data fetched successfully for {city_name}.")
            
            # 3. Return clean string
            return (
                f"Current Weather for {city_name}, {country}:\n"
                f"- Temperature: {current.get('temperature_2m')}°C (Feels like {current.get('apparent_temperature')}°C)\n"
                f"- Humidity: {current.get('relative_humidity_2m')}%\n"
                f"- Wind Speed: {current.get('wind_speed_10m')} km/h\n"
                f"- Precipitation: {current.get('precipitation')} mm"
            )

        except Exception as e:
            print(f"DEBUG ERROR: {e}")
            return f"Error fetching weather: {str(e)}"

if __name__ == "__main__":
    print("Weather Server ready")
    mcp.run(transport="sse")