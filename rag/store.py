"""ChromaDB 向量存储管理。"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DB_PATH = Path(__file__).parent / "chroma_db"

_client: chromadb.ClientAPI | None = None
_embedding_fn = DefaultEmbeddingFunction()


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        DB_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(DB_PATH))
    return _client


def close_client() -> None:
    """释放 ChromaDB 连接，避免 --reload 时 SQLite 锁冲突。"""
    global _client
    if _client is not None:
        try:
            _client._sysdb.stop()
        except Exception:
            pass
        _client = None


def get_collection(name: str = "travel_knowledge") -> chromadb.Collection:
    return get_client().get_or_create_collection(
        name=name,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
