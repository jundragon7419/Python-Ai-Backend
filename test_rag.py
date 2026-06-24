from fastembed import TextEmbedding
import chromadb

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # overlap만큼 겹치게 이동
    return chunks

# 샘플 긴 문서
document = """
FastAPI는 Python으로 만든 고성능 웹 프레임워크다. 
Starlette와 Pydantic을 기반으로 동작한다.
비동기 처리를 기본으로 지원하며 uvicorn으로 실행한다.
Pydantic은 데이터 검증 라이브러리로 타입 힌트를 기반으로 동작한다.
FastAPI는 자동으로 OpenAPI 문서를 생성해준다.
RAG는 Retrieval-Augmented Generation의 약자다.
벡터 DB는 텍스트를 숫자 벡터로 저장하고 유사도로 검색한다.
embedding은 텍스트를 숫자 벡터로 변환하는 과정이다.
chunking은 긴 문서를 작은 조각으로 나누는 과정이다.
tiktoken은 OpenAI가 만든 토큰 계산 라이브러리다.
"""

# 1. chunking
chunks = chunk_text(document, chunk_size=100, overlap=20)
print(f"chunk 개수: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"chunk{i}: {chunk!r}")

# 2. embedding
model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = list(model.embed(chunks))

# 3. Chroma에 저장
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("rag_test")

collection.add(
    documents=chunks,
    embeddings=[e.tolist() for e in embeddings],
    ids=[f"chunk{i}" for i in range(len(chunks))]
)

# 4. 검색 테스트
query = "FastAPI 실행 방법"
query_embedding = list(model.embed([query]))[0]

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=2
)

print(f"\n질문: {query}")
print(f"검색 결과:")
for doc in results["documents"][0]:
    print(f"  - {doc!r}")