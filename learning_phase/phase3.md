# Python AI 백엔드 학습 노트 — Phase 3

## Phase 3 목표

`fake_llm()`을 실제 LLM API로 교체하고, 프로덕션에 필요한 패턴을 익힌다.

---

## 환경변수 관리 — dotenv

API 키를 코드에 직접 쓰면 깃허브에 올라가는 순간 노출된다.
`.env` 파일에 저장하고 `dotenv`로 불러오는 게 표준 패턴이다.

```bash
pip install python-dotenv
```

```plaintext
# .env 파일
GROQ_API_KEY=your_api_key_here
```

```python
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일 로드
client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
```

> `.env` 파일은 반드시 `.gitignore`에 추가해야 한다.

---

## Groq SDK — 비동기 LLM 호출

Groq는 무료로 쓸 수 있는 LLM API 서비스다.
`AsyncGroq`를 쓰면 비동기로 LLM을 호출할 수 있다.

```python
from groq import AsyncGroq, APIError

client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
```

> `Groq` vs `AsyncGroq`
> - `Groq`: 동기 클라이언트 — LLM 응답 기다리는 동안 서버 블로킹
> - `AsyncGroq`: 비동기 클라이언트 — 응답 기다리는 동안 다른 요청 처리 가능
> 
> FastAPI 서버에서는 반드시 `AsyncGroq`를 써야 한다.

---

## 비스트리밍 LLM 호출

```python
async def call_llm(history: list[Message], llm_model: str) -> str:
    try:
        response = await client.chat.completions.create(
            model=llm_model,
            messages=SYSTEM_PROMPT + trim_history(history),
            stream=False
        )
        return response.choices[0].message.content
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.message}")
```

> 502 상태코드
> "우리 서버는 정상인데 외부 API(LLM)가 실패했다"는 의미.
> LLM API 에러는 500이 아니라 502로 반환하는 게 맞다.

---

## 스트리밍 LLM 호출

LLM이 토큰을 생성하는 즉시 클라이언트에 전송하는 방식이다.
ChatGPT가 답변을 한 글자씩 출력하는 것과 같은 패턴이다.

```python
async def call_llm_stream(history: list[Message], llm_model: str, session_id: str) -> AsyncGenerator[str, None]:
    try:
        stream = await client.chat.completions.create(
            model=llm_model,
            messages=SYSTEM_PROMPT + trim_history(history),
            stream=True  # 스트리밍 활성화
        )
        fully_reply = ""
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                fully_reply += content  # 누적 (히스토리 저장용)
                yield content           # 클라이언트에 즉시 전송

        # 스트리밍 완료 후 히스토리에 저장
        sessions[session_id].append({
            "role": "assistant",
            "content": fully_reply,
            "tokens": count_tokens(fully_reply)
        })
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.message}")
```

> `stream=False` vs `stream=True`
> | | 비스트리밍 | 스트리밍 |
> |--|--|--|
> | 응답 방식 | 완성 후 한 번에 반환 | 토큰 단위로 즉시 전송 |
> | 반환 타입 | `str` | `AsyncGenerator[str, None]` |
> | 히스토리 저장 | 호출한 쪽에서 저장 | 함수 내부에서 저장 (스트리밍 완료 후) |
> | 용도 | 구조화된 JSON 응답 필요 시 | 사용자에게 실시간으로 보여줄 때 |

> 스트리밍에서 `fully_reply` 누적이 중요한 이유
> 스트리밍 중에는 전체 응답을 알 수 없다.
> 토큰을 하나씩 `fully_reply`에 더해가다가 스트리밍이 끝난 후에야 히스토리에 저장할 수 있다.

---

## System Prompt

LLM의 동작 방식을 제어하는 프롬프트다.
매 요청마다 히스토리 앞에 붙여서 LLM에 전달한다.

```python
SYSTEM_PROMPT: list[LLMMessage] = [{
    "role": "system",
    "content": "You are a helpful assistant. Respond in the same language as the user's message."
}]

# LLM에 넘길 때
messages = SYSTEM_PROMPT + trim_history(history)
```

> LLM 자체를 바꾸지 않고 system prompt만 바꿔도 동작 방식이 완전히 달라진다.
> 이게 프롬프트 엔지니어링의 가장 기본적인 패턴이다.

---

## 토큰 관리

### 토큰이란

LLM이 텍스트를 처리하는 단위다. 글자 수와 다르게 동작한다.

```
"hello" → 1토큰
"안녕하세요" → 한글은 글자당 2~3토큰 (영어보다 많이 씀)
```

API 비용은 토큰 수로 계산된다:
- **input tokens**: 우리가 LLM에 보내는 모든 텍스트 (system prompt + 히스토리 + 새 메시지)
- **output tokens**: LLM이 생성한 텍스트 (output이 input보다 2~3배 비쌈)

### tiktoken으로 토큰 수 계산

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))
```

### trim_history — context window 관리

히스토리가 쌓이면 LLM의 context window 한계를 초과해서 에러가 난다.
최신 대화를 유지하고 오래된 대화를 잘라내는 함수가 필요하다.

```python
def trim_history(history: list[Message], max_tokens: int = 4000) -> list[LLMMessage]:
    total = 0
    trimmed = []
    for msg in reversed(history):  # 최신 메시지부터 역순으로 순회
        msg_tokens = msg["tokens"]
        if total + msg_tokens > max_tokens:
            break  # 한계 초과 시 오래된 메시지부터 잘림
        trimmed.append(msg)
        total += msg_tokens
    return [{"role": m["role"], "content": m["content"]} for m in reversed(trimmed)]
    #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #        tokens 키 제거 — LLM API는 role, content만 받음
```

> `reversed` 두 번 쓰는 이유
> 1. 첫 번째 `reversed`: 최신 메시지부터 순회해서 최근 대화를 우선 유지
> 2. 두 번째 `reversed`: 원래 시간 순서로 복구 (LLM이 대화 순서를 이해하도록)

---

## TypedDict로 타입 명확화

```python
from typing import TypedDict

class Message(TypedDict):      # 히스토리 저장용
    role: str
    content: str
    tokens: int

class LLMMessage(TypedDict):   # LLM API 전달용
    role: str
    content: str

sessions: dict[str, list[Message]] = {}
SYSTEM_PROMPT: list[LLMMessage] = [...]
```

> `dict[str, str | int]` 대신 TypedDict를 쓰면:
> - 어떤 키가 있는지 명확해짐
> - IDE 자동완성 지원
> - `msg["tokens"]` 접근 시 타입 안전성 확보

---

## 최종 코드 구조

```
main.py
├── 상수
│   ├── SYSTEM_PROMPT       # LLM 동작 제어
│   ├── LLM_MODELS          # 허용 모델 목록
│   └── ALLOWED_ROLES       # 허용 role 목록
│
├── TypedDict
│   ├── Message             # 히스토리 저장용 (role, content, tokens)
│   └── LLMMessage          # LLM API 전달용 (role, content)
│
├── Pydantic Models
│   ├── ChatRequest         # 입력 검증 (user_id, message, role, llm_model)
│   └── ChatResponse        # 출력 구조 (reply, session_id, history)
│
├── 유틸 함수
│   ├── count_tokens()      # tiktoken 토큰 수 계산
│   └── trim_history()      # context window 초과 방지
│
├── LLM 함수
│   ├── call_llm()          # 비스트리밍 호출
│   └── call_llm_stream()   # 스트리밍 호출
│
└── 엔드포인트
    ├── POST /chat          # 비스트리밍 응답
    ├── POST /chat/stream   # 스트리밍 응답
    └── DELETE /chat/{user_id}  # 세션 삭제
```

---

## 다음 단계 — Phase 4 RAG

지금은 LLM이 자기 학습 데이터로만 답한다.
RAG(Retrieval-Augmented Generation)를 붙이면 우리가 가진 문서를 기반으로 답하게 만들 수 있다.

핵심 흐름:
```
문서 → chunking → embedding → 벡터 DB 저장
질문 → embedding → 벡터 DB 검색 → 관련 문서 → 프롬프트에 주입 → LLM 응답
```