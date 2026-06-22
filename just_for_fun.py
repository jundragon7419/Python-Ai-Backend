from fastapi import FastAPI, HTTPException # FastAPI, HTTPExeption(Client exception handling)
from pydantic import BaseModel, field_validator, FieldValidationInfo # Server input exception handling
import uuid
from uuid import uuid5 # SHA-1 based hash

app = FastAPI()

sessions: dict[str, list[dict[str, str]]] = {} # username based session history

LLM_MODELS = ["sonet 4.6", "opus 4.8", "Fable 5"]

ALLOWED_ROLES = {"user", "system"}

class ChatRequest(BaseModel): # request(input)
    user_id: str
    message: str
    role: str = "user"
    llm_model: str = "sonet 4.6"

    @field_validator("user_id", "message") # user_id, message empty check
    @classmethod
    def user_id_not_empty(cls, v: str, info: FieldValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} is empty.")
        return v
    
    @field_validator("llm_model")
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

def fake_llm(message: str, history: list[dict[str, str]], llm_model: str) -> str: # fake LLM
    return f"echo: {message} (conversation {len(history) + 1}) [written by {llm_model}]"

@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    session_id = str(uuid5(uuid.NAMESPACE_DNS, request.user_id))
    
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append({"role": request.role, "content": request.message})

    reply = fake_llm(request.message, history, request.llm_model)
    history.append({"role": "assistant", "content": reply})

    return ChatResponse(reply=reply, session_id=session_id, history=history)

@app.delete("/chat/{user_id}")
def reset(user_id: str):
    session_id = str(uuid5(uuid.NAMESPACE_DNS, user_id))
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session does not exist")
    del sessions[session_id]
    return {"message": f"{user_id} deleted"}