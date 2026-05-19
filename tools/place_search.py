"""地点搜索工具 — 使用 Wikipedia API 查询景点信息。"""

import urllib.parse

import requests
from langchain_core.tools import tool


@tool
def search_place_info(query: str) -> str:
    """搜索城市或景点的详细信息，包括简介、地理位置等。适用于当用户想了解某个目的地或景点时使用。

    Args:
        query: 搜索关键词，如 "浅草寺", "巴厘岛", "卢浮宫"
    """
    # 先搜索中文维基
    for lang in ["zh", "en"]:
        result = _search_wiki(lang, query)
        if result:
            return result

    return f"未找到关于「{query}」的详细信息。"


def _search_wiki(lang: str, query: str) -> str | None:
    api_base = f"https://{lang}.wikipedia.org/api/rest_v1"
    encoded = urllib.parse.quote(query, safe="")
    try:
        resp = requests.get(
            f"{api_base}/page/summary/{encoded}",
            timeout=5,
            headers={"User-Agent": "AITravelPlanner/1.0"},
        )
        if resp.status_code != 200:
            # 尝试搜索
            return _search_and_fetch(lang, query)

        data = resp.json()
        title = data.get("title", query)
        extract = data.get("extract", "")
        coords = data.get("coordinates", {})

        if not extract:
            return None

        lines = [f"📍 {title}", f"  {extract}"]
        if coords:
            lines.append(f"  坐标: {coords.get('lat', '?')}, {coords.get('lon', '?')}")

        return "\n".join(lines)

    except (requests.RequestException, ValueError):
        return None


def _search_and_fetch(lang: str, query: str) -> str | None:
    """当直接查询失败时，用搜索 API 找到最佳匹配。"""
    try:
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        resp = requests.get(
            search_url,
            params={"action": "opensearch", "search": query, "limit": 1, "format": "json"},
            timeout=5,
            headers={"User-Agent": "AITravelPlanner/1.0"},
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        if len(data) < 2 or not data[1]:
            return None

        # 用找到的标题重新获取摘要
        best_title = data[1][0]
        encoded = urllib.parse.quote(best_title, safe="")
        api_base = f"https://{lang}.wikipedia.org/api/rest_v1"
        resp2 = requests.get(
            f"{api_base}/page/summary/{encoded}",
            timeout=5,
            headers={"User-Agent": "AITravelPlanner/1.0"},
        )
        if resp2.status_code != 200:
            return None

        d = resp2.json()
        extract = d.get("extract", "")
        if not extract:
            return None

        return f"📍 {d.get('title', best_title)}\n  {extract}"

    except (requests.RequestException, ValueError):
        return None
