from fastembed import TextEmbedding # 텍스트 임베딩
from chromadb import Collection

DOCUMENT_TOOL = {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": "저장된 문서에서 관련 내용을 검색한다",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색할 내용"}
                },
                "required": ["query"]
            }
        }
    }

## 문서 찾아보기 ##
def search_document(query: str, embedding_model: TextEmbedding, collection: Collection) -> str:
    query_embedding = list(embedding_model.embed([query]))[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )
    chunks = list(dict.fromkeys(results["documents"][0]))
    return "\n".join(chunks)