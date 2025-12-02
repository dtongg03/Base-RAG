# qa.py
from embedding import embed_texts
from qdrant_setup import get_qdrant_client, search_qdrant, points_to_context
COLLECTION_NAME = "documents"

def answer_question(question: str, client, top_k: int = 1):
    # 1. Question -> Embedding
    query_emb = embed_texts([question])[0]
    # 2. Vector DB -> search
    results = search_qdrant(client, COLLECTION_NAME, query_emb, top_k=top_k)
    if not results:
        print("Không tìm được đoạn nào liên quan trong vector DB.")
        return
    context = points_to_context(results)
    print("\n===== CONTEXT LẤY TỪ VECTOR DB =====")
    print(context)

def main():
    print("Classbot Q&A mode (gõ 'exit' để thoát)")
    client = get_qdrant_client(path="./vector_store")
    while True:
        q = input("\nCâu hỏi của bạn: ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        answer_question(q, client, top_k=1)

if __name__ == "__main__":
    main()
