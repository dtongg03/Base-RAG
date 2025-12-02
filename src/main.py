from embedding import embed_texts
from data_process import load_text_files, load_pdf_files, build_sentence_chunks
from qdrant_setup import get_qdrant_client, init_collection, upsert_embeddings
from qdrant_setup import search_qdrant

def main():
    print("Đang load dữ liệu từ thư mục ../data ...")
    docs = load_text_files("../data")
    print(f"   -> Tìm được {len(docs)} documents")
    print("Count Chunk")
    chunks = build_sentence_chunks(docs)
    print(f"   -> Tổng số câu: {len(chunks)}")

    if not chunks:
        print("Không có chunk nào. Kiểm tra lại thư mục data.")
        return
    texts = [c["text"] for c in chunks]

    print("Đang tạo embeddings ...")
    embeddings = embed_texts(texts, batch_size=16)
    vector_dim = embeddings.shape[1]
    print(f"   -> Embedding shape: {embeddings.shape}")

    print("Đang khởi tạo Qdrant ...")
    client = get_qdrant_client(path="./vector_store")
    init_collection(client, collection_name="documents", vector_size=vector_dim)

    print("Đang upsert embeddings vào Qdrant ...")
    upsert_embeddings(client, "documents", chunks, embeddings)

    print("Test search với câu query demo ...")
    query = "Nội dung gì đó liên quan tới tài liệu của bạn"
    query_emb = embed_texts([query])[0]
    results = search_qdrant(client, "documents", query_emb, top_k=3)
    for r in results:
        print("\n===== KẾT QUẢ =====")
        print("Score:", r.score)
        print("Doc ID:", r.payload.get("doc_id"))
        print("Chunk ID:", r.payload.get("chunk_id"))
        print("Text:", r.payload.get("text")[:200], "...")


if __name__ == "__main__":
    main()
