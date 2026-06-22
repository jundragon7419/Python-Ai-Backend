from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import uuid

app = FastAPI()

sessions: dict[str, list[dict[str, str]]] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("메시지가 비어있습니다")
        return v

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    history: list[dict[str, str]]

def fake_llm(message: str, history: list[dict[str, str]]) -> str:
    return f"에코: {message} (대화 {len(history) + 1}번째)"

@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append({"role": "user", "content": request.message})

    reply = fake_llm(request.message, history)
    history.append({"role": "assistant", "content": reply})

    return ChatResponse(reply=reply, session_id=session_id, history=history)

@app.delete("/chat/{session_id}")
def reset(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="존재하지 않는 세션")
    del sessions[session_id]
    return {"message": f"{session_id} 삭제됨"}