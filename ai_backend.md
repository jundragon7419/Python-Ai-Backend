# Python AI 백엔드 학습 노트 — Phase 1 & 2

## AI 백엔드란?

LLM(GPT, Claude 등)은 텍스트를 받아 텍스트를 뱉는 함수에 불과하다.
AI 백엔드는 그것을 실제 서비스로 만들기 위한 모든 주변 로직이다.

```
일반 백엔드: 클라이언트 요청 → DB 조회/수정 → 결과 반환
AI 백엔드:  클라이언트 요청 → (전처리 → LLM 호출 → 후처리) → 결과 반환
```

### AI 백엔드가 하는 일

| 역할 | 설명 |
|------|------|
| 프롬프트 조립 | 시스템 프롬프트 + 문서 + 지시사항을 조합해 LLM에 전달 |
| LLM 호출 + 스트리밍 | API 호출 후 토큰 단위로 클라이언트에 전송 |
| 컨텍스트 관리 | 대화 히스토리를 저장하고 매 요청마다 프롬프트에 주입 |
| RAG 파이프라인 | 벡터 DB에서 관련 문서 검색 후 프롬프트에 주입 |
| Tool Use | LLM 판단에 따라 실제 API를 대신 호출 |
| 비용/안정성 관리 | rate limiting, 캐싱, 비용 추적, retry 로직 |

> LLM을 요리사라고 하면, AI 백엔드는 주방 시스템 전체다. LLM 전후처리를 맡는 것.

---

## Phase 1 — Python 핵심 개념

### Type Hints

실제 동작에는 영향을 주지 않고, 함수의 입력/출력 타입을 명시하는 주석 역할이다.
IDE와 협업자에게 타입 정보를 전달한다.

```python
def process(items: list[str]) -> dict[str, int]:
#                  ^^^^^^^^^^    ^^^^^^^^^^^^^^
#                  인자 타입 힌트  반환값 타입 힌트 (실행에 영향 없음)
    result = {}
    for item in items:
        result[item] = len(item)
    return result
```

### Optional과 Union

```python
from typing import Optional

# Optional[int] = int | None
# 값이 있을 수도, None일 수도 있음
tokens: Optional[int] = None  # 기본값 None

# Python 3.10+ 에서는 | 로 표현 가능
tokens: int | None = None      # 위와 동일
role: str | int = "user"       # str 또는 int
#예외의 경우 BaseModel로 처리 가능
```

### dataclass

데이터 전용 클래스를 간단하게 정의하는 도구.
`__init__`, `__eq__`, `__repr__` 등을 자동으로 생성해준다.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Message:
    role: str
    content: str
    tokens: Optional[int] = None  # 기본값 있는 필드

msg = Message(role="user", content="안녕")
# msg.tokens → None (기본값)
```

> 주의: dataclass는 타입 힌트대로 검증하지 않는다.
> 잘못된 타입을 넣어도 에러가 나지 않는다.(BaseModel 사용!!)

### Generator와 yield

`return`은 값을 반환하고 함수가 종료되지만,
`yield`는 값을 하나 반환하고 일시정지한다. 다음 호출 때 이어서 실행된다. (제너레이터다!)

```python
def token_chunks(text: str, chunk_size: int):
    words = text.split()  # list 반환 (tuple 아님)
    for i in range(0, len(words), chunk_size):
        yield words[i:i + chunk_size]  # 하나 주고 일시정지

# AI 백엔드에서의 활용
# LLM이 토큰을 스트리밍으로 뱉을 때 이 패턴을 그대로 사용
# 제너레이터 미사용할 경우를 상상하면 처참함
async def llm_stream(message: str):
    async for token in llm_client.stream(message):
        yield token  # 토큰 나올 때마다 yield → 클라이언트에 즉시 전송
```

### async/await

동시성 코드를 작성하는 방법이다.
동기는 순차, 비동기는 병렬(동시가 더 적절한 예시)
`await`은 "기다리는 동안 CPU를 다른 코루틴에 양보"한다.

```python
import asyncio

async def fetch_response(prompt: str) -> str: # async로 동시성 코드 할게요 설정 
    await asyncio.sleep(1)  # 1초 동안 CPU 양보 (다른 코루틴 실행 가능)
    return f"응답: {prompt}"

async def main():
    # gather: 3개를 동시에(동시성) 실행 → 총 1초 (순차였으면 3초)
    results = await asyncio.gather(
        fetch_response("질문1"),
        fetch_response("질문2"),
        fetch_response("질문3"),
    )

asyncio.run(main())
# 그냥 main()하면 안되나요? -> 코루틴 객체만 생성되고 실행 안됨
# 이벤트 루프를 만들고 코루틴을 실제로 실행하려면 run 해야됨
```

> `time.sleep(1)` vs `await asyncio.sleep(1)`
> - `time.sleep`: CPU를 붙잡고 있어서 다른 코루틴 실행 불가
> - `await asyncio.sleep`: CPU를 양보해서 다른 코루틴 실행 가능

---

## Phase 2 — FastAPI로 AI API 서버 구축

### 설치 및 실행

```bash
pip install fastapi uvicorn
uvicorn main:app --reload
```

### 기본 구조

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")           # GET / → 이 함수 실행
def root():
    return {"message": "hello"}

@app.get("/echo/{text}")  # URL 경로 파라미터
def echo(text: str):
    return {"you_said": text}
```

`@app.get("/chat")`은 데코레이터다.
- URL 경로 (`/chat`) 와 HTTP 메서드 (`GET`) 를 함수에 등록한다.
- 같은 URL이라도 메서드가 다르면 다른 함수가 실행된다.
- 같은 URL와 같은 메서드에서는 오직 하나의 함수만 가진다.
- 만약
```python
@app.get("/chat/{session_id}")  # 경로 파라미터
def get_by_id(session_id: str): ...

@app.get("/chat/active")  # 고정 경로
def get_active(): ...
```
- 과 같은 구조라면 session_id를 자동을 active로 넣는다.
- FastApi가 GET, /chat/active의 get_by_id만을 저장한다.
- 다만 메서드가 다를경우 상관없다.

### HTTP 메서드

| 메서드 | 의미 | 예시 |
|--------|------|------|
| `GET` | 데이터 조회 | 히스토리 불러오기 |
| `POST` | 데이터 생성/전송 | 메시지 보내기 |
| `DELETE` | 데이터 삭제 | 히스토리 삭제 |
| `PUT` | 데이터 전체 수정 | 메시지 교체 |
| `PATCH` | 데이터 일부 수정 | 메시지 일부 수정 |

> 기능적으로는 전부 POST로 구현 가능하지만,
> REST API 설계 원칙에 따라 메서드를 구분하면 협업 시 의도가 명확해진다.

### Pydantic BaseModel

dataclass와 달리 **타입 검증**을 실제로 수행한다.

```python
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 100  # 기본값

class ChatResponse(BaseModel):
    reply: str
    token_count: int

@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    reply = f"에코: {request.message}"
    return ChatResponse(
        reply=reply,
        token_count=len(request.message)
    )
```

> dataclass vs BaseModel
> - dataclass: 타입 힌트만 있고 검증 없음 (잘못된 타입 넣어도 에러 없음)
> - BaseModel: 실제로 타입 검증 수행 (잘못된 타입 → 자동으로 422 에러)

### /docs — Swagger UI

FastAPI는 코드를 기반으로 API 문서를 자동 생성한다.
`http://localhost:8000/docs` 에서 확인 및 직접 테스트 가능.
프론트엔드 없이도 API 테스트가 가능해서 개발 효율이 높다.

### Streaming Response

LLM이 토큰을 하나씩 생성할 때, 전부 완성될 때까지 기다리지 않고
생성되는 즉시 클라이언트에 전송하는 패턴이다.

```python
from fastapi.responses import StreamingResponse
import asyncio

async def fake_llm_stream(message: str):
    words = f"안녕하세요! '{message}'에 대한 답변입니다.".split()
    for word in words:
        yield word + " "          # 단어 하나 전송
        await asyncio.sleep(0.3)  # 0.3초 대기 (실제 LLM은 생성 속도)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        fake_llm_stream(request.message),
        media_type="text/plain"
    )
```

StreamingResponse 내부 동작:
```python
# StreamingResponse가 내부적으로 하는 일 (의사 코드)
for chunk in fake_llm_stream(request.message):
    socket.send(chunk)  # chunk 나올 때마다 즉시 전송
```
- StreamingResponse의 경우 iterable을 입력받아도 되지만 결국 목적에 맞는 것은 제너레이터이다.
- 동기 제너레이터의 경우 한명의 사용자가 하나의 세션에서 사용할 때는 상관없으나 여러 사용자가 동시에 스트리밍 요청시 사용자 한명 한명 순차적으로 실행되기에 매우매우 비효율적이라 사실상 비동기 제너레이터만 입력함.

### Session 기반 히스토리 관리

LLM은 매 요청마다 기억이 초기화된다.
백엔드가 대화 내용을 저장하고 매번 프롬프트에 주입해야 한다.

```python
import uuid

# 전역변수로 세션별 히스토리 관리
sessions: dict[str, list[dict[str, str]]] = {}

@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    # session_id 없으면 새로 생성
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append({"role": "user", "content": request.message})

    reply = fake_llm(request.message, history)
    history.append({"role": "assistant", "content": reply})

    return ChatResponse(reply=reply, session_id=session_id, history=history)
```

> Python의 `or` 동작
> ```python
> None or "uuid-1234"   # → "uuid-1234" (None이 falsy)
> "abc" or "uuid-1234"  # → "abc"       (abc가 truthy)
> ```
> A or B: A가 falsy면 B 반환, truthy면 A 반환

> uuid 종류
> - `uuid4`: 랜덤 기반 고유 식별자 (세션 ID 등 랜덤 생성에 적합)
> - `uuid3/5`: 이름 기반 (사용자명으로 동일 세션 보장할 때 적합, uuid5가 보안상 우수)

> 전역변수 문제점
> 모든 사용자가 히스토리를 공유하게 된다.
> 실제 서비스에서는 user_id 또는 session_id로 반드시 분리해야 한다.

### 에러 처리

```python
from fastapi import HTTPException
from pydantic import field_validator

class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("메시지가 비어있습니다")
        return v

@app.delete("/chat/{session_id}")
def reset(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="존재하지 않는 세션")
    del sessions[session_id]
    return {"message": f"{session_id} 삭제됨"}
```

| 예외 | 사용 위치 | 상태코드 |
|------|----------|---------|
| `ValueError` | Pydantic 검증 단계 (field_validator 내부) | 자동으로 422 |
| `HTTPException` | FastAPI 라우터 (비즈니스 로직) | 직접 지정 |

> @field_validator 실행 시점
> ```
> ChatRequest(message="   ") 호출
>     → Pydantic이 message 필드 받음
>     → @field_validator 실행 → ValueError 발생
>     → Pydantic이 자동으로 422로 변환
>     → chat() 함수는 실행조차 안 됨
> ```

---