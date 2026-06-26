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
from tools import TOOLS, get_weather, search_document # 도구 
import json
import hashlib

class Message(TypedDict):
    """
    세션 history에 저장되는 단일 메시지.

    Attributes:
        role: 메시지 작성 주체. "user" / "assistant" / "system"
        content: 해당 role이 보낸 실제 텍스트
        tokens: content의 토큰 수. history trimming의 기준값
    """
    role: str
    content: str
    tokens: int

class LLMMessage(TypedDict):
    """
    LLM API에 전달하기 위한 메시지 형태.

    내부 저장용 Message에서 tokens를 제외하고 role/content만 남긴 구조로,
    Groq chat.completions가 요구하는 형식과 일치한다.

    Attributes:
        role: 메시지 작성 주체. "user" / "assistant" / "system"
        content: 해당 role의 텍스트
    """
    role: str
    content: str

## 글로벌 ##
app = FastAPI()
load_dotenv()
client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
"""비동기 Groq """
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
"""텍스트 임베딩 모델"""
# model = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = chromadb.Client()
"""DB"""
collection = chroma_client.get_or_create_collection("documents")
"""DB 문서"""
sessions: dict[str, list[Message]] = {}
"""session_id를 key로, 메시지 history를 value로 갖는 in-memory 저장소.

서버 재시작 시 초기화되며 영속성은 없다.
"""

ALLOWED_ROLES = {
    "user",
    "system"
    }
"""ChatRequest.role로 허용되는 값의 집합. role validator에서 검증 기준으로 사용."""

SYSTEM_PROMPT: list[LLMMessage] = [{"role": "system", "content": "You are a helpful assistant. Respond in the same language as the user's message."}]
"""모든 LLM 호출 앞에 고정으로 붙는 system 메시지. 사용자 언어로 응답하도록 지시."""

LLM_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b"
]
"""사용 가능한 Groq 모델 화이트리스트. llm_model validator의 검증 기준."""

TOOL_MAP = {
    "get_weather": get_weather,
    "search_document": lambda **kwargs: search_document(
        embedding_model=embedding_model,
        collection=collection,
        **kwargs
    )
}
"""tool 이름 → 실제 호출 가능한 함수 매핑.

search_document는 embedding_model/collection을 미리 바인딩하기 위해 lambda로 감쌌다.
agent 루프에서 LLM이 요청한 tool_name으로 함수를 조회해 실행한다.
"""

enc = tiktoken.get_encoding("cl100k_base")
"""tiktoken 인코더. count_tokens에서 사용.

cl100k_base는 OpenAI 계열 tokenizer이므로 Llama/Groq 모델의 실제 토큰 수와는
다르며, 여기서 세는 토큰 수는 근사치다.
"""

def count_tokens(text: str) -> int:
    """
    텍스트를 토큰화해 토큰 개수를 반환한다.

    Args:
        text: 토큰 수를 셀 문자열
    Returns:
        cl100k_base 기준 토큰 수(근사치)
    """
    return len(enc.encode(text))

def trim_history(history: list[Message], max_tokens: int = 4000) -> list[LLMMessage]:
    """
    토큰 예산 안에서 최신 메시지부터 history를 잘라 LLM 입력 형태로 변환한다.

    최신 메시지부터 역순으로 누적 토큰을 더하다가 max_tokens를 넘으면 멈추고,
    Message에서 tokens를 제거한 LLMMessage 리스트를 시간순으로 반환한다.

    Args:
        history: 자를 대상 메시지 리스트
        max_tokens: 포함할 메시지들의 누적 토큰 상한
    Returns:
        시간순으로 정렬된, LLM에 전달 가능한 메시지 리스트
    """
    total = 0
    trimmed = []
    for msg in reversed(history):
        msg_tokens = msg["tokens"]
        if total + msg_tokens > max_tokens:
            break
        trimmed.append(msg)
        total += msg_tokens
    return [{"role": m["role"], "content": m["content"]} for m in reversed(trimmed)]

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    텍스트를 일정 크기로 겹치며 자른다(sliding window chunking).

    각 청크는 chunk_size 문자이며, 인접 청크끼리 overlap 문자를 공유한다.
    청크 경계에서 문맥이 끊기는 것을 줄이기 위해 overlap을 둔다.
    (토큰 단위가 아니라 문자 단위로 자른다.)

    Args:
        text: 분할할 원문
        chunk_size: 청크 1개의 최대 문자 수
        overlap: 인접 청크가 겹치는 문자 수
    Returns:
        분할된 청크 리스트
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

class ChatRequest(BaseModel):
    """
    /chat 계열 엔드포인트의 요청 body.

    Attributes:
        user_id: 세션 식별의 기준이 되는 사용자 ID
        message: 사용자 입력 텍스트
        role: 메시지 role. 기본값 "user"
        llm_model: 사용할 LLM 모델명. 기본값 "llama-3.3-70b-versatile"

    Raises:
        ValidationError: 필드 validator가 ValueError를 던지면 Pydantic이
            ValidationError로 묶어 발생시키며, FastAPI가 422 응답으로 변환한다.
    """
    user_id: str
    message: str
    role: str = "user"
    llm_model: str = "llama-3.3-70b-versatile"

    ## 예외 처리 ##
    @field_validator("user_id", "message")
    @classmethod
    def user_id_not_empty(cls, v: str, info: FieldValidationInfo) -> str:
        """user_id와 message가 공백만으로 이루어지지 않았는지 검증한다."""
        if not v.strip():
            raise ValueError(f"{info.field_name} is empty.")
        return v
    
    @field_validator("llm_model")
    @classmethod
    def wrong_llm_model(cls, v:str, info: FieldValidationInfo) -> str:
        """llm_model이 LLM_MODELS 화이트리스트에 포함되는지 검증한다."""
        if v not in LLM_MODELS:
            raise ValueError(f"{info.field_name} is an invalid model name.")
        return v
    
    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str, info: FieldValidationInfo) -> str:
        """role이 ALLOWED_ROLES에 포함되는지 검증한다."""
        if v not in ALLOWED_ROLES:
            raise ValueError(f"{info.field_name} must be one of {ALLOWED_ROLES}")
        return v

class ChatResponse(BaseModel):
    """
    /chat 계열 엔드포인트의 응답 body.

    Attributes:
        reply: LLM이 생성한 응답 텍스트
        session_id: user_id로부터 생성된 세션 식별자(uuid5)
        history: 이번 응답까지 반영된 전체 대화 history
    """
    reply: str
    session_id: str
    history: list[Message]

class DocumentRequest(BaseModel):
    """
    /documents 엔드포인트의 요청 body.

    Attributes:
        content: 임베딩해 저장할 원문 텍스트
    """
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str, info: FieldValidationInfo) -> str:
        """content가 공백만으로 이루어지지 않았는지 검증한다."""
        if not v.strip():
            raise ValueError(f"{info.field_name} is empty.")
        return v

async def call_llm(history: list[Message], llm_model: str) -> str:
    """
    history를 LLM에 전달해 단일 응답을 받아 반환한다(non-streaming).

    SYSTEM_PROMPT를 앞에 붙이고 trim_history로 잘라낸 history를 전달한다.

    Args:
        history: 전체 대화 history
        llm_model: 호출할 모델명
    Returns:
        LLM 응답 텍스트
    Raises:
        HTTPException: Groq APIError 발생 시 status_code 502로 변환.
    """
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

async def call_llm_stream(history: list[Message], llm_model: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    history를 LLM에 전달해 응답을 청크 단위로 스트리밍한다.

    각 청크의 content를 yield하고, 스트림이 끝나면 전체 응답을
    sessions[session_id]에 assistant 메시지로 누적 저장한다.

    Args:
        history: 전체 대화 history
        llm_model: 호출할 모델명
        session_id: 완성된 응답을 저장할 세션 키
    Yields:
        응답 텍스트 청크
    Raises:
        HTTPException: APIError 발생 시 502. 단, 이미 청크를 한 번이라도
            yield한 뒤에는 HTTP status/header가 전송된 상태라 502로 변환되지
            않고 스트림만 중단된다.
    """
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

@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    사용자 메시지를 받아 LLM 응답을 생성하고 history에 누적한다(non-streaming).

    user_id로 세션을 찾거나 새로 만들고, 요청 메시지와 생성된 응답을 history에 추가한다.
    """
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

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    사용자 메시지를 받아 LLM 응답을 스트리밍으로 반환한다(text/plain).

    assistant 응답의 history 저장은 이 함수가 아니라 call_llm_stream 내부에서
    스트림 완료 시점에 이뤄진다.
    """
    session_id = str(uuid5(uuid.NAMESPACE_DNS, request.user_id))

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append({"role": request.role, "content": request.message, "tokens": count_tokens(request.message)})


    return StreamingResponse(
        call_llm_stream(history, request.llm_model, session_id),
        media_type = "text/plain"
    )

@app.delete("/chat/{user_id}")
def reset(user_id: str):
    """
    user_id에 해당하는 세션 history를 삭제한다.

    Args:
        user_id: 삭제할 세션의 사용자 ID
    Raises:
        HTTPException: 해당 세션이 없으면 404.
    """
    session_id = str(uuid5(uuid.NAMESPACE_DNS, user_id))
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session does not exist")
    del sessions[session_id]
    return {"message": f"{user_id} deleted"}

@app.post("/documents")
def add_document(request: DocumentRequest):
    """
    문서를 청크로 나눠 임베딩한 뒤 Chroma collection에 저장한다.

    각 청크를 임베딩하고 "chunk_{index}_{hash}" 형태의 id를 부여해 저장한다.
    """
    chunks = chunk_text(request.content)
    embeddings = list(embedding_model.embed(chunks)) # numpy 배열임
    
    ids = [
        f"chunk_{hashlib.md5(chunk.encode()).hexdigest()}"
        for chunk in chunks
    ]
    
    collection.add(
        documents=chunks,
        embeddings=[e.tolist() for e in embeddings], # numpy 배열을 list로 전환 384차원 -> 리스트
        ids=ids
    )
    return {"message": f"{len(chunks)}개 chunk 저장됨"}

@app.post("/chat/rag")
async def chat_rag(request: ChatRequest) -> ChatResponse:
    """
    질문을 임베딩해 Chroma에서 관련 문서를 검색하고, 그 내용을 프롬프트에
    주입해 LLM 응답을 생성한다(RAG).

    검색 결과 상위 3개 청크를 중복 제거한 뒤 context로 묶어 질문과 함께 전달한다.
    """
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
    rag_message = f"아래 문서를 참고해서 답해:\n{context}\n\n질문: {request.message}"

    # 3. LLM 호출
    history = sessions[session_id]
    history.append({"role": request.role, "content": rag_message, "tokens": count_tokens(rag_message)})

    reply = await call_llm(history, request.llm_model) # 구조화된 response가 어려워 call_llm 사용 (stream x)
    history.append({"role": "assistant", "content": reply, "tokens": count_tokens(reply)})

    return ChatResponse(reply=reply, session_id=session_id, history=history)

@app.post("/chat/agent")
async def chat_agent(request: ChatRequest) -> ChatResponse:
    """
    LLM이 필요 시 tool을 호출하도록 하는 tool-use 에이전트 루프(ReAct 형태).

    응답의 finish_reason이 "tool_calls"인 동안, 요청된 tool을 TOOL_MAP에서 찾아
    실행하고 결과를 messages에 추가하며 반복한다. tool 호출이 없으면 최종 응답을 반환한다.
    """
    session_id = str(uuid5(uuid.NAMESPACE_DNS, request.user_id))
    
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append({"role": request.role, "content": request.message, "tokens": count_tokens(request.message)})

    messages = [{"role": h["role"], "content": h["content"]} for h in history]

    while True:
        response = await client.chat.completions.create(
            model=request.llm_model,
            messages=SYSTEM_PROMPT + messages,
            tools=TOOLS
        )

        choice = response.choices[0]

        if choice.finish_reason != "tool_calls":
            reply = choice.message.content
            break

        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            result = TOOL_MAP[tool_name](**tool_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    history.append({"role": "assistant", "content": reply, "tokens": count_tokens(reply)})

    return ChatResponse(reply=reply, session_id=session_id, history=history)
