"""RAG 检索接口 — 供 LangGraph agent 调用的工具。"""

from __future__ import annotations

from langchain_core.tools import tool

from rag.store import get_collection


@tool
def search_travel_knowledge(
    query: str,
    city: str = "",
    category: str = "",
    season: str = "",
    n_results: int = 3,
) -> str:
    """从旅行知识库中检索相关信息。当你需要提供具体的、经过验证的旅行建议时使用此工具。

    知识库包含：景点攻略、餐厅推荐、交通指南、季节贴士、文化须知等。
    返回的信息比你自身的知识更具体、更有时效性。

    Args:
        query: 搜索内容，如 "东京赏樱最佳地点", "曼谷防宰客技巧"
        city: 可选，按城市过滤，如 "东京", "曼谷", "巴黎"
        category: 可选，按类别过滤：attraction（景点）/ restaurant（餐厅）/ transport（交通）/ tip（贴士）/ culture（文化）
        season: 可选，按季节过滤：spring / summer / autumn / winter / all
        n_results: 返回结果数量，默认3条
    """
    collection = get_collection()

    if collection.count() == 0:
        return "知识库为空，暂无数据。"

    # 构建元数据过滤条件
    where_conditions = []
    if city:
        where_conditions.append({"city": city})
    if category:
        where_conditions.append({"category": category})
    if season:
        where_conditions.append({"$or": [{"season": season}, {"season": "all"}]})

    where = None
    if len(where_conditions) == 1:
        where = where_conditions[0]
    elif len(where_conditions) > 1:
        where = {"$and": where_conditions}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            where=where,
        )
    except Exception:
        # 如果过滤条件导致无结果，去掉过滤重试
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
        )

    if not results["documents"] or not results["documents"][0]:
        return f"未找到与「{query}」相关的信息。"

    lines = [f"📚 知识库检索结果（{query}）：\n"]
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]), 1
    ):
        city_tag = meta.get("city", "")
        cat_tag = meta.get("category", "")
        season_tag = meta.get("season", "")
        tags = " · ".join(filter(None, [city_tag, cat_tag, season_tag]))
        # 单条结果截断到 500 字符，避免占用过多 token
        doc_text = doc if len(doc) <= 500 else doc[:500] + "..."
        lines.append(f"[{i}] ({tags})")
        lines.append(f"  {doc_text}")
        lines.append("")

    return "\n".join(lines)
