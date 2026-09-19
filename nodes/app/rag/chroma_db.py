from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from shared.embeddings.local import LocalSentenceTransformerEmbeddings
from shared.schemas.document import DocumentChunk, SearchResult


class ChromaStore:
    def __init__(
        self,
        persist_dir: str | Path = "./chroma_storage",
        collection_name: str = "fedrag_corpus",
        embedder: Optional[LocalSentenceTransformerEmbeddings] = None,
    ):
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.embedder = embedder or LocalSentenceTransformerEmbeddings()

        # Initialize persistent disk client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Embeds and persists document chunks into ChromaDB."""
        if not chunks:
            return

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        ids = [chunk.id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[SearchResult], List[List[float]]]:
        """Performs vector search; returns top-k SearchResults and their raw embeddings for TASR feedback."""
        query_emb = self.embedder.embed_query(query_text)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_emb],
            "n_results": top_k,
            "include": ["documents", "distances", "metadatas", "embeddings"],
        }
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)

        search_results: List[SearchResult] = []
        doc_embeddings: List[List[float]] = []

        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        embs = results.get("embeddings", [[]])[0]

        for doc, dist, meta, emb in zip(docs, distances, metas, embs):
            # ChromaDB cosine distance d in [0, 2]; similarity = 1 - (dist / 2)
            score = 1.0 - (float(dist) / 2.0)
            search_results.append(
                SearchResult(content=doc, score=score, metadata=meta or {})
            )
            doc_embeddings.append(emb)

        return search_results, doc_embeddings

    def get_all_embeddings(self) -> List[List[float]]:
        """Fetches all indexed vector embeddings to calculate registration centroids."""
        data = self.collection.get(include=["embeddings"])
        return data.get("embeddings") or []