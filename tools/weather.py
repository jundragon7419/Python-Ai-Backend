WEATHER_TOOL = {
    "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 날씨를 가져온다",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "도시 이름"}
                },
                "required": ["city"]
            }
        }
    }

## API 차후 주입 ##
def get_weather(city: str) -> str:
    return f"{city}의 날씨는 맑음, 23도입니다."