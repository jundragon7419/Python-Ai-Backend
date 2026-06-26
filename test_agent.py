import os
from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# 도구 정의
tools = [
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "특정 회사의 주가를 가져온다",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "회사 이름"}
                },
                "required": ["company"]
            }
        }
    }
]

# 도구 실행 함수들 (mock)
def get_weather(city: str) -> str:
    return f"{city}의 날씨는 맑음, 23도입니다."

def get_stock_price(company: str) -> str:
    return f"{company}의 주가는 71,400원입니다."

# 도구 이름 → 함수 매핑
TOOL_MAP = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
}

def run_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]
    
    print(f"질문: {user_message}")
    
    # ReAct 루프
    while True:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools
        )
        
        choice = response.choices[0]
        
        # 도구 호출이 필요 없으면 최종 답변
        if choice.finish_reason != "tool_calls":
            print(f"최종 답변: {choice.message.content}\n")
            break
        
        # 도구 호출이 필요하면 실행
        messages.append(choice.message)
        
        for tool_call in choice.message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"  → 도구 선택: {tool_name}({tool_args})")
            
            result = TOOL_MAP[tool_name](**tool_args)
            print(f"  → 결과: {result}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

run_agent("서울 날씨랑 삼성전자 주가 알려줘")
print("---")
run_agent("파이썬이 뭐야?")