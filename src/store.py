from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, FixedSizeChunker
from .embeddings import _mock_embed, _openai_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _openai_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

     
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
      
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": doc.metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    
        query_embedding = self._embedding_fn(query)
        scored = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append({"content": record["content"], "score": score, "metadata": record["metadata"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # Chunk large documents to fit OpenAI's 8192 token limit (~6000 chars for safety)
        chunker = FixedSizeChunker(chunk_size=6000, overlap=200)
        
        for doc in docs:
            # If content is short, embed directly; otherwise chunk it
            if len(doc.content) <= 6000:
                chunks = [(doc.id, doc.content)]
            else:
                chunk_texts = chunker.chunk(doc.content)
                chunks = [(f"{doc.id}_chunk_{i}", text) for i, text in enumerate(chunk_texts)]
            
            for chunk_id, chunk_text in chunks:
                embedding = self._embedding_fn(chunk_text)
                record = {
                    "id": chunk_id,
                    "content": chunk_text,
                    "embedding": embedding,
                    "metadata": {**doc.metadata, "doc_id": doc.id},
                }
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            records = self._store
        else:
            records = []
            for record in self._store:
                match = True
                for key, value in metadata_filter.items():
                    if record["metadata"].get(key) != value:
                        match = False
                        break
                if match:
                    records.append(record)
        return self._search_records(query, records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        original_len = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id and r["id"] != doc_id]
        return len(self._store) < original_len
