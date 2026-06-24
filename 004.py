from fastapi import FastAPI, HTTPException # 웹 프레임워크, client 예외
from fastapi.responses import StreamingResponse # 비동기 스트림
from pydantic import BaseModel, field_validator, FieldValidationInfo # 서버 input 예외 핸들링
import uuid # namespace dns에 사용
from uuid import uuid5 # SHA-1 based 해시 (세션에서 사용)
from groq import AsyncGroq, APIError # free llm module / Error 핸들링
from dotenv import load_dotenv # 환경변수 처리
import os
import tiktoken # Token 계산용
from typing import AsyncGenerator # 제너레이터(비동기) 타입 설정용
from typing import TypedDict # 타입 설정용
from fastembed import TextEmbedding # 텍스트 임베딩
import chromadb # db

## 타입 정의 ##
class Message(TypedDict): # history 내부 타입 정의
    role: str
    content: str
    tokens: int

class LLMMessage(TypedDict):  # LLM에게 전해지는 내용(history 일부) 정의
    role: str
    content: str

sessions: dict[str, list[Message]] = {} # username based session history

## 글로벌 ##
app = FastAPI()
client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
load_dotenv()
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5") # 텍스트 임베딩 모델 (영어 특화)
# model = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = chromadb.Client() # db 객체
collection = chroma_client.get_or_create_collection("documents") # chroma에 저장을 위함

## 글로벌 변수 ##
ALLOWED_ROLES = {"user", "system"}
SYSTEM_PROMPT: list[LLMMessage] = [{"role": "system", "content": "You are a helpful assistant. Respond in the same language as the user's message."}]
LLM_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]

## 토큰 ##
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

## 입력 정의 ##
class ChatRequest(BaseModel):
    user_id: str
    message: str
    role: str = "user"
    llm_model: str = "llama-3.3-70b-versatile"

    ## 예외 처리 ##
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
    
    @field_validator("role") # role check
    @classmethod
    def role_must_be_valid(cls, v: str, info: FieldValidationInfo) -> str:
        if v not in ALLOWED_ROLES:
            raise ValueError(f"{info.field_name} must be one of {ALLOWED_ROLES}")
        return v

## 출력 정의 ##
class ChatResponse(BaseModel): # response(output)
    reply: str
    session_id: str
    history: list[Message]

## 토큰 사용량 기반 history 정리 ##
def trim_history(history: list[Message], max_tokens: int = 4000) -> list[LLMMessage]:
    total = 0
    trimmed = []
    for msg in reversed(history):
        msg_tokens = msg["tokens"]
        if total + msg_tokens > max_tokens:
            break
        trimmed.append(msg)
        total += msg_tokens
    return [{"role": m["role"], "content": m["content"]} for m in reversed(trimmed)]


## 비동기 llm (스트림 x) ##
async def call_llm(history: list[Message], llm_model: str) -> str:
    try:
        response = await client.chat.completions.create(
            model = llm_model,
            messages = SYSTEM_PROMPT + trim_history(history),
            stream = False
        )
        reply = response.choices[0].message.content

        return reply
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.message}")


## 비동기 llm (스트림 o) ##
async def call_llm_stream(history: list[Message], llm_model: str, session_id: str) -> AsyncGenerator[str, None]:
    try:
        stream = await client.chat.completions.create(
            model = llm_model,
            messages = SYSTEM_PROMPT + trim_history(history),
            stream = True
        )
        fully_reply = ""
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                fully_reply += content
                yield content

        sessions[session_id].append({"role": "assistant", "content": fully_reply, "tokens": count_tokens(fully_reply)}) # token calculation

    except APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.message}")

## call llm ##
@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = str(uuid5(uuid.NAMESPACE_DNS, request.user_id))
    
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append({"role": request.role, "content": request.message, "tokens": count_tokens(request.message)})

    reply = await call_llm(history, request.llm_model)
    history.append({"role": "assistant", "content": reply, "tokens": count_tokens(reply)})

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        history=history
    )

## call streaming llm ##
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    session_id = str(uuid5(uuid.NAMESPACE_DNS, request.user_id))

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append({"role": request.role, "content": request.message, "tokens": count_tokens(request.message)})


    return StreamingResponse(
        call_llm_stream(history, request.llm_model, session_id),
        media_type = "text/plain"
    )

## delete session ##
@app.delete("/chat/{user_id}")
def reset(user_id: str):
    session_id = str(uuid5(uuid.NAMESPACE_DNS, user_id))
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session does not exist")
    del sessions[session_id]
    return {"message": f"{user_id} deleted"}

## chunking ##
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]: # 기본 청크 사이즈 200, 오버랩 50
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

## document 입력 정의 ##
class DocumentRequest(BaseModel):
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str, info: FieldValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} is empty.")
        return v

## document 입력 ##
@app.post("/documents")
def add_document(request: DocumentRequest):
    chunks = chunk_text(request.content)
    embeddings = list(embedding_model.embed(chunks)) # numpy 배열임
    
    ids = [f"chunk_{i}_{hash(chunk)}" for i, chunk in enumerate(chunks)]
    
    collection.add(
        documents=chunks,
        embeddings=[e.tolist() for e in embeddings], # numpy 배열을 list로 전환 384차원 -> 리스트
        ids=ids
    )
    return {"message": f"{len(chunks)}개 chunk 저장됨"}

## RAG를 통한 검색 (Retrieval-Augmented Generation, 검색으로 보강된 생성) ##
@app.post("/chat/rag")
async def chat_rag(request: ChatRequest) -> ChatResponse:
    session_id = str(uuid5(uuid.NAMESPACE_DNS, request.user_id))

    if session_id not in sessions:
        sessions[session_id] = []

    # 1. 질문 embedding → Chroma 검색
    query_embedding = list(embedding_model.embed([request.message]))[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()], # 임베딩된 질문 numpy를 리스트화
        n_results=3 # 검색결과 3개 가져오기 (정보부족 가능성 이유로)
    )
    retrieved_chunks = results["documents"][0] # 첫 쿼리 출력 
    ## 중복 제거
    retrieved_chunks = list(dict.fromkeys(retrieved_chunks))
    context = "\n".join(retrieved_chunks)

    # 2. 검색된 문서를 프롬프트에 주입
    context = "\n".join(retrieved_chunks)
    rag_message = f"아래 문서를 참고해서 답해:\n{context}\n\n질문: {request.message}"

    # 3. LLM 호출
    history = sessions[session_id]
    history.append({"role": request.role, "content": rag_message, "tokens": count_tokens(rag_message)})

    reply = await call_llm(history, request.llm_model)
    history.append({"role": "assistant", "content": reply, "tokens": count_tokens(reply)})

    return ChatResponse(reply=reply, session_id=session_id, history=history)




