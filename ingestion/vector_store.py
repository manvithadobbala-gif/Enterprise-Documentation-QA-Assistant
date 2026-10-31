import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from ingestion.text_splitter import DocumentChunk
from ingestion.embeddings import EmbeddingManager

load_dotenv()

DEFAULT_CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
DEFAULT_COLLECTION_NAME = "enterprise_documents"


class VectorStoreManager:
    """Manages persistent ChromaDB vector storage, chunk indexing, and metadata queries."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_CHROMA_DIR)
        self.collection_name = collection_name
        self.embedding_manager = embedding_manager or EmbeddingManager()

        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Use cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def is_document_indexed(self, file_hash: str) -> bool:
        """Checks if a document with the given content hash has already been indexed."""
        results = self.collection.get(
            where={"file_hash": file_hash},
            limit=1
        )
        return len(results.get("ids", [])) > 0

    def add_chunks(self, chunks: List[DocumentChunk], force_update: bool = False) -> int:
        """
        Embeds and stores document chunks in ChromaDB.
        Avoids duplicates unless force_update is True.
        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        # Group chunks by document hash to detect duplicates
        hashes = {c.metadata.get("file_hash") for c in chunks if "file_hash" in c.metadata}

        if not force_update:
            filtered_chunks = []
            for c in chunks:
                f_hash = c.metadata.get("file_hash")
                if f_hash and self.is_document_indexed(f_hash):
                    continue
                filtered_chunks.append(c)
            chunks_to_add = filtered_chunks
        else:
            # If force_update, remove existing chunks with these hashes first
            for f_hash in hashes:
                if f_hash:
                    self.delete_by_hash(f_hash)
            chunks_to_add = chunks

        if not chunks_to_add:
            return 0

        ids = [c.chunk_id for c in chunks_to_add]
        documents = [c.text for c in chunks_to_add]
        metadatas = [c.metadata for c in chunks_to_add]

        # Generate embeddings
        embeddings = self.embedding_manager.embed_documents(documents)

        # Upsert in batches of 100 to stay well within ChromaDB payload limits
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size]
            )

        return len(chunks_to_add)

    def search(
        self,
        query: str,
        top_k: int = 4,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic search for a query and returns retrieved chunks with distance scores.
        Cosine distance in ChromaDB is in [0, 2], where 0 = identical.
        Similarity score = 1.0 - (distance / 2.0).
        """
        if not query or not query.strip():
            return []

        query_vector = self.embedding_manager.embed_query(query)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            ids = results["ids"][0] if "ids" in results else [""] * len(docs)

            for doc_text, meta, dist, chunk_id in zip(docs, metas, dists, ids):
                # Standard cosine similarity: 1.0 - cosine_distance
                similarity = max(0.0, min(1.0, 1.0 - float(dist)))
                item = {
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "metadata": meta,
                    "distance": float(dist),
                    "similarity": round(similarity, 4)
                }
                retrieved.append(item)

        return retrieved

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """Returns summary metadata of all unique documents currently indexed in ChromaDB."""
        data = self.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        if not metadatas:
            return []

        docs_summary: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            doc_name = meta.get("document", "Unknown")
            f_hash = meta.get("file_hash", "")
            key = f"{doc_name}_{f_hash}"

            if key not in docs_summary:
                docs_summary[key] = {
                    "document": doc_name,
                    "file_type": meta.get("file_type", ""),
                    "file_hash": f_hash,
                    "chunk_count": 0,
                    "pages": set()
                }
            docs_summary[key]["chunk_count"] += 1
            if "page" in meta:
                docs_summary[key]["pages"].add(meta["page"])

        summary_list = []
        for v in docs_summary.values():
            summary_list.append({
                "document": v["document"],
                "file_type": v["file_type"],
                "file_hash": v["file_hash"],
                "chunk_count": v["chunk_count"],
                "total_pages": len(v["pages"]) if v["pages"] else 1
            })
        return summary_list

    def delete_document(self, filename: str) -> None:
        """Deletes all chunks belonging to a specific document name."""
        self.collection.delete(where={"document": filename})

    def delete_by_hash(self, file_hash: str) -> None:
        """Deletes all chunks belonging to a specific document hash."""
        self.collection.delete(where={"file_hash": file_hash})

    def count(self) -> int:
        """Returns total number of chunks stored in the collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Deletes the collection and recreates an empty one."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
