"""知识库数据加载器。

数据来源：data/seed_knowledge.json
支持两种方式写入：
  1. add_entry() — 逐条添加
  2. load_seed_data() — 从 JSON 文件批量加载
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from rag.store import get_collection

SEED_FILE = Path(__file__).parent / "seed_knowledge.json"


def add_entry(
    content: str,
    city: str = "",
    country: str = "",
    category: str = "",  # attraction / restaurant / transport / tip / culture
    season: str = "",    # spring / summer / autumn / winter / all
    tags: list[str] | None = None,
    source: str = "",
) -> str:
    """向知识库添加一条旅行知识。返回 document ID。"""
    collection = get_collection()
    doc_id = str(uuid.uuid4())[:8]

    metadata = {}
    if city:
        metadata["city"] = city
    if country:
        metadata["country"] = country
    if category:
        metadata["category"] = category
    if season:
        metadata["season"] = season
    if tags:
        metadata["tags"] = ",".join(tags)
    if source:
        metadata["source"] = source

    collection.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[metadata],
    )
    return doc_id


def load_seed_data() -> int:
    """从 data/seed_knowledge.json 加载种子数据。返回知识库总条目数。"""
    collection = get_collection()

    # 已有数据就跳过
    if collection.count() > 0:
        return collection.count()

    if not SEED_FILE.exists():
        return 0

    with open(SEED_FILE, encoding="utf-8") as f:
        entries = json.load(f)

    for entry in entries:
        add_entry(**entry)

    return collection.count()
