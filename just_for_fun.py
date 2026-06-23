from fastapi import FastAPI, HTTPException # FastAPI, HTTPExeption(Client exception handling)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator, FieldValidationInfo # Server input exception handling
import uuid
from uuid import uuid5 # SHA-1 based hash
from groq import AsyncGroq, APIError # free llm module / Error handling
from dotenv import load_dotenv # Setting environment variables
import os

SYSTEM_PROMPT = [{"role": "system", "content": "You are a helpful assistant. Always respond in Korean with detailed answers."}]

load_dotenv()
client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

app = FastAPI() 

sessions: dict[str, list[dict[str, str]]] = {} # username based session history

LLM_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]

ALLOWED_ROLES = {"user", "system"}

class ChatRequest(BaseModel): # request(input)
    user_id: str
    message: str
    role: str = "user"
    llm_model: str = "llama-3.3-70b-versatile"

    @field_validator("user_id", "message") # user_id, message empty check
    @classmethod
    def user_id_not_empty(cls, v: str, info: FieldValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} is empty.")
        return v
    
    @field_validator("llm_model") # llm_model check
    @classmethod
    def wrong_llm_model(cls, v:str, info: FieldValidationInfo) -> str:
        if v not in LLM_MODELS:
            raise ValueError(f"{info.field_name} is an invalid model name.")
        return v
    
    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str, info: FieldValidationInfo) -> str:
        if v not in ALLOWED_ROLES:
            raise ValueError(f"{info.field_name} must be one of {ALLOWED_ROLES}")
        return v

class ChatResponse(BaseModel): # response(output)
    reply: str
    session_id: str
    history: list[dict[str, str]]

async def call_llm(history, llm_model, session_id):
    try:
        response = await client.chat.completions.create(
            model = llm_model,
            messages = SYSTEM_PROMPT + history,
            stream = False
        )
        return response.choices[0].message.content
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.message}")


async def call_llm_stream(history: list[dict[str, str]], llm_model: str, session_id: str):
    try:
        stream = await client.chat.completions.create(
            model = llm_model,
            messages = SYSTEM_PROMPT + history,
            stream = True
        )
        fully_reply = ""
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                fully_reply += content
                yield content

        sessions[session_id].append({"role": "assistant", "content": fully_reply})
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.message}")

@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = str(uuid5(uuid.NAMESPACE_DNS, request.user_id))
    
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append({"role": request.role, "content": request.message})

    reply = await call_llm(history, request.llm_model, session_id)
    history.append({"role": "assistant", "content": reply})

    return ChatResponse(reply=reply, session_id=session_id, history=history)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    session_id = str(uuid5(uuid.NAMESPACE_DNS, request.user_id))

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append({"role": request.role, "content": request.message})


    return StreamingResponse(
        call_llm_stream(history, request.llm_model, session_id),
        media_type = "text/plain"
    )

@app.delete("/chat/{user_id}")
def reset(user_id: str):
    session_id = str(uuid5(uuid.NAMESPACE_DNS, user_id))
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session does not exist")
    del sessions[session_id]
    return {"message": f"{user_id} deleted"}