import os

# Prevent transformers and huggingface_hub from attempting network calls when cached
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


class EmbeddingManager:
    """
    Manages Sentence Transformer embedding generation with caching,
    configurable model selection, and vector normalization.
    """

    _instances = {}

    def __new__(cls, model_name: Optional[str] = None):
        name = model_name or os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]

    def __init__(self, model_name: Optional[str] = None):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
        self._model = None
        self._initialized = True

    @property
    def model(self):
        """Lazy loader for the SentenceTransformer model to prevent startup latency."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_query(self, query: str) -> List[float]:
        """Generates a normalized embedding vector for a single query string."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates normalized embedding vectors for a list of document strings."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()
