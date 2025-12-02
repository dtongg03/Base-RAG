from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


def get_qdrant_client(path: str = "../vector_store") -> QdrantClient:
    client = QdrantClient(path=path)
    return client


def init_collection(client: QdrantClient, collection_name: str, vector_size: int):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    print(f"Collection '{collection_name}' đã được tạo với dim = {vector_size}")


def upsert_embeddings(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Dict],
    embeddings,
):
    points: List[PointStruct] = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings), start=1):
        points.append(
            PointStruct(
                id=i,
                vector=emb.tolist(),
                payload={
                    "text": chunk["text"],
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                },
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
    )
    print(f"Đã upsert {len(points)} vectors vào collection '{collection_name}'")


def search_qdrant(
    client: QdrantClient,
    collection_name: str,
    query_embedding,
    top_k: int = 3,
):
    vector = query_embedding.tolist()

    if hasattr(client, "query_points"):
        resp = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=top_k,
        )
        return resp.points

    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=top_k,
        )

    raise RuntimeError("QdrantClient không có query_points() hay search().")


def points_to_context(results) -> str:
    parts = []
    for p in results:
        text = p.payload.get("text", "")
        doc_id = p.payload.get("doc_id", "")
        parts.append(f"[{doc_id}] {text}")
    return "\n\n".join(parts)
