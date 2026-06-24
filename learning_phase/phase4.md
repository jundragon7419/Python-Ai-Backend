# Python AI 백엔드 학습 노트 — Phase 4

## Phase 4 목표

RAG(Retrieval-Augmented Generation) 파이프라인을 구현한다.
LLM이 자기 학습 데이터가 아닌 우리가 가진 문서를 기반으로 답하게 만든다.

---

## RAG란

**"검색으로 보강된 생성"**이다.

LLM의 두 가지 한계를 해결한다:
- 학습 데이터 이후 정보를 모름
- 우리 회사 내부 문서를 모름

해결 방식: LLM을 재학습시키는 게 아니라, **질문할 때 관련 문서를 찾아서 같이 넘겨주는 것**이다.

```
# RAG 없이
질문: "우리 회사 환불 정책이 뭐야?"
LLM: "모르겠습니다" 또는 hallucination

# RAG 있이
질문: "우리 회사 환불 정책이 뭐야?"
백엔드: 문서 DB에서 "환불 정책" 관련 chunk 검색
LLM: "환불은 구매 후 7일 이내에 가능하며..." (문서 기반 정확한 답변)
```

---

## 전체 흐름

### 문서 저장 파이프라인 (사전 작업)

```
문서
 → chunking      : 긴 문서를 작은 조각으로 나눔
 → embedding     : 각 chunk를 벡터로 변환
 → 벡터 DB 저장  : Chroma에 저장
```

### 검색 + 생성 파이프라인 (질문 처리)

```
질문
 → embedding          : 질문도 벡터로 변환
 → similarity search  : 벡터 DB에서 가장 관련 있는 chunk 검색
 → 프롬프트 조립      : "아래 문서 참고해서 답해: [검색된 chunk]"
 → LLM 호출           : 문서 기반 답변 생성
```

---

## Embedding이란

텍스트를 숫자 벡터로 변환하는 과정이다.
의미가 비슷한 텍스트는 벡터도 비슷해진다.

```
"강아지" → [0.2, 0.8, 0.1, 0.5, ...]  (384차원 숫자 배열)
"개"     → [0.2, 0.7, 0.1, 0.5, ...]  ← 비슷한 벡터
"자동차" → [0.9, 0.1, 0.8, 0.2, ...]  ← 다른 벡터
```

키워드 검색("FastAPI"라는 단어가 있는 문서 찾기)이 아니라
**의미 기반 검색**이라 훨씬 정확하다.

### LLM vs Embedding 모델

| | LLM (Groq) | Embedding 모델 |
|--|--|--|
| 역할 | 텍스트 생성 | 텍스트 → 벡터 변환 |
| 출력 | 문장 | 숫자 배열 (384차원) |
| 실행 위치 | 클라우드 API | 로컬 (fastembed) |
| 예시 | llama-3.3-70b | BAAI/bge-small-en-v1.5 |

---

## fastembed

PyTorch 없이 ONNX 런타임으로 동작하는 embedding 라이브러리다.
처음 실행 시 모델을 다운로드하고, 이후엔 로컬에서 실행된다.

```python
from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")  # 영어 특화
# model = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  # 다국어

embeddings = list(model.embed(["FastAPI는...", "Pydantic은..."]))
# [numpy.ndarray([0.23, -0.15, ...]), numpy.ndarray([0.71, 0.32, ...])]
```

> numpy.ndarray → Python list 변환
> Chroma는 numpy 배열을 직접 못 받아서 `.tolist()`로 변환해야 한다.
> ```python
> [e.tolist() for e in embeddings]
> ```

---

## Chunking

긴 문서를 작은 조각으로 나누는 과정이다.

```python
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # overlap만큼 겹치게 이동
    return chunks
```

### overlap이 필요한 이유

chunk 경계에서 문장이 잘릴 수 있다.
overlap으로 앞 chunk의 끝부분을 다음 chunk 앞에 포함시켜서 문맥이 끊기지 않게 한다.

```
overlap 없이:
chunk0: "...uvicorn으로 실행한"   ← 불완전한 문장
chunk1: "다. Pydantic은..."      ← 앞 문맥 없음

overlap 있이:
chunk0: "...uvicorn으로 실행한"
chunk1: "으로 실행한다. Pydantic은..."  ← 앞 내용 포함
```

---

## Chroma — 벡터 DB

벡터를 저장하고 유사도로 검색하는 DB다.

```python
import chromadb

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("documents")
#                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                   컬렉션 이름 (테이블 같은 개념)
```

### 문서 저장

```python
collection.add(
    documents=chunks,                        # 원본 텍스트 (검색 결과로 반환)
    embeddings=[e.tolist() for e in embeddings],  # 벡터 (유사도 검색에 사용)
    ids=[f"chunk_{i}_{hash(chunk)}" for i, chunk in enumerate(chunks)]  # 고유 ID
)
```

### 유사도 검색

```python
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3  # 가장 유사한 chunk 3개 반환
)

# results 구조
# {
#     "documents": [["chunk1", "chunk2", "chunk3"]],  # [0]으로 접근
#     "distances": [[0.12, 0.34, 0.67]],              # 낮을수록 유사
#     "ids": [["chunk_0_abc", "chunk_1_def", ...]]
# }

retrieved_chunks = results["documents"][0]  # 첫 번째 쿼리 결과
```

> `results["documents"]`가 2중 리스트인 이유
> 쿼리를 여러 개 동시에 보낼 수 있어서 쿼리 단위로 묶여있다.
> 지금은 쿼리가 1개라서 `[0]`으로 첫 번째 결과만 꺼낸다.

> n_results 선택 기준
> - 너무 적으면: 관련 정보 부족
> - 너무 많으면: 관련 없는 chunk 포함 → 토큰 낭비, 응답 품질 저하
> - 실서비스에서는 3~5가 적당

---

## RAG 프롬프트 패턴

검색된 chunk를 프롬프트에 주입하는 방식이다.

```python
retrieved_chunks = results["documents"][0]

# 중복 chunk 제거
retrieved_chunks = list(dict.fromkeys(retrieved_chunks))

context = "\n".join(retrieved_chunks)
rag_message = f"아래 문서를 참고해서 답해:\n{context}\n\n질문: {request.message}"
```

LLM이 받는 프롬프트:
```
아래 문서를 참고해서 답해:
FastAPI는 Python으로 만든 고성능 웹 프레임워크다...
uvicorn으로 실행한다...

질문: FastAPI 실행 방법 알려줘
```

---

## 엔드포인트 구조

```
POST /documents    # 문서 추가 (chunking → embedding → Chroma 저장)
POST /chat/rag     # RAG 기반 답변 (검색 → 프롬프트 주입 → LLM 호출)
```

---

## 최종 코드 구조

```
main.py
├── 기존 (Phase 2~3)
│   ├── /chat          # 비스트리밍
│   ├── /chat/stream   # 스트리밍
│   └── /chat/{user_id} DELETE  # 세션 삭제
│
└── 추가 (Phase 4)
    ├── embedding_model  # fastembed 모델
    ├── chroma_client    # Chroma DB 클라이언트
    ├── collection       # 문서 컬렉션
    ├── chunk_text()     # 문서 chunking
    ├── /documents POST  # 문서 저장
    └── /chat/rag POST   # RAG 기반 채팅
```

---

## 다음 단계 — Phase 5 Agent

LLM이 스스로 도구를 선택하고 실행하는 agentic workflow를 구현한다.

```
사용자: "오늘 날씨 알려줘"
Agent:  → weather API 호출해야겠다 (tool 선택)
        → weather API 호출 (tool 실행)
        → 결과를 받아서 답변 생성
```

핵심 개념:
- Tool Use / Function Calling
- ReAct 루프 (Reasoning + Acting)
- Multi-agent 아키텍처