from embedding import embed_texts
from qdrant_setup import get_qdrant_client, search_qdrant, points_to_context
from llm_connect import response_with_llm
COLLECTION_NAME = "documents"

def answer_question(question: str, client, top_k: int = 3):
    query_emb = embed_texts([question])[0]
    results = search_qdrant(client, COLLECTION_NAME, query_emb, top_k=top_k)
    if not results:
        print("Không tìm được đoạn nào liên quan trong vector DB.")
        return
    context = points_to_context(results)
    return context

def main():
    print("Classbot Q&A mode (gõ 'exit' để thoát)")
    client = get_qdrant_client(path="./vector_store")
    while True:
        q = input("\nCâu hỏi của bạn: ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        raw_qerry = answer_question(q, client)
        response = response_with_llm(q, raw_qerry)
        print("RAG: ", response)

if __name__ == "__main__":
    main()
