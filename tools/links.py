"""视频推荐 — YouTube Data API v3 + Bilibili 搜索 API。"""

from __future__ import annotations

import logging
import os
import time
import urllib.parse
from dataclasses import dataclass, field

import requests

log = logging.getLogger("travel_planner")


# ── 数据模型 ─────────────────────────────────────────────────────────────────

@dataclass
class Video:
    title: str
    url: str
    author: str
    view_count: int
    published_at: str
    platform: str  # "youtube" | "bilibili"


@dataclass
class RecommendationLinks:
    videos: list[Video] = field(default_factory=list)

    @property
    def has_videos(self) -> bool:
        return len(self.videos) > 0

    @property
    def youtube_videos(self) -> list[Video]:
        return [v for v in self.videos if v.platform == "youtube"]

    @property
    def bilibili_videos(self) -> list[Video]:
        return [v for v in self.videos if v.platform == "bilibili"]


# ── YouTube ──────────────────────────────────────────────────────────────────

def _search_youtube(
    keyword: str,
    api_key: str,
    max_results: int = 2,
) -> list[Video]:
    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": keyword,
                "type": "video",
                "maxResults": max_results,
                "order": "relevance",
                "relevanceLanguage": "en",
                "key": api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except (requests.RequestException, ValueError):
        return []

    if not items:
        return []

    video_ids = [item["id"]["videoId"] for item in items]
    stats = _fetch_youtube_stats(api_key, video_ids)

    videos = []
    for item in items:
        vid = item["id"]["videoId"]
        snippet = item.get("snippet", {})
        stat = stats.get(vid, {})
        videos.append(Video(
            title=snippet.get("title", ""),
            url=f"https://www.youtube.com/watch?v={vid}",
            author=snippet.get("channelTitle", ""),
            view_count=int(stat.get("viewCount", 0)),
            published_at=snippet.get("publishedAt", "")[:10],
            platform="youtube",
        ))

    return videos


def _fetch_youtube_stats(api_key: str, video_ids: list[str]) -> dict:
    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={
                "part": "statistics",
                "id": ",".join(video_ids),
                "key": api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return {}

    return {
        item["id"]: item.get("statistics", {})
        for item in data.get("items", [])
    }


# ── Bilibili ─────────────────────────────────────────────────────────────────

def _search_bilibili(
    keyword: str,
    max_results: int = 2,
) -> list[Video]:
    """使用 Bilibili 搜索 API（公开接口，无需 key）。"""
    try:
        resp = requests.get(
            "https://api.bilibili.com/x/web-interface/search/type",
            params={
                "search_type": "video",
                "keyword": keyword,
                "page": 1,
                "pagesize": max_results,
                "order": "totalrank",  # 综合排序
            },
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Referer": "https://www.bilibili.com",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    raw_results = data.get("data", {}).get("result", [])
    # result 可能是 None 或空
    if not raw_results:
        return []

    # 只取 type=video 的结果
    results = [r for r in raw_results if r.get("type") == "video"]
    if not results:
        results = raw_results  # fallback: 取全部

    videos = []
    for item in results[:max_results]:
        bvid = item.get("bvid", "")
        if not bvid:
            continue
        # Bilibili 搜索结果的 title 带 <em> 高亮标签，需要清理
        title = item.get("title", "").replace("<em class=\"keyword\">", "").replace("</em>", "")
        play = item.get("play", 0)
        # play 可能是字符串 "--" 表示无数据
        if isinstance(play, str):
            play = 0
        videos.append(Video(
            title=title,
            url=f"https://www.bilibili.com/video/{bvid}",
            author=item.get("author", ""),
            view_count=int(play),
            published_at=_format_bilibili_date(item.get("pubdate", 0)),
            platform="bilibili",
        ))

    return videos


def _format_bilibili_date(timestamp: int) -> str:
    """将 Unix 时间戳转为 YYYY-MM-DD。"""
    if not timestamp:
        return ""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")


# ── 统一入口 ─────────────────────────────────────────────────────────────────

def search_videos(
    place_name: str,
    destination: str = "",
    max_results_per_platform: int = 5,
    language: str = "zh",
) -> RecommendationLinks:
    """根据语言搜索视频。中文→B站，英文→YouTube。"""
    all_videos: list[Video] = []

    if language == "zh":
        keyword = f"{place_name} {destination} 旅行攻略".strip()
        all_videos.extend(_search_bilibili(keyword, max_results_per_platform))
    else:
        keyword = f"{place_name} {destination} travel guide".strip()
        yt_key = os.getenv("YOUTUBE_API_KEY", "")
        if yt_key:
            all_videos.extend(_search_youtube(keyword, yt_key, max_results_per_platform))

    all_videos.sort(key=lambda v: v.view_count, reverse=True)

    return RecommendationLinks(videos=all_videos)


def search_videos_batch(
    place_names: list[str],
    destination: str = "",
    max_results_per_platform: int = 5,
    language: str = "zh",
) -> dict[str, RecommendationLinks]:
    """批量并发搜索多个景点的视频。"""
    import concurrent.futures
    t0 = time.time()
    log.info(f"[videos] batch start — {len(place_names)} places")
    results: dict[str, RecommendationLinks] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(search_videos, name, destination, max_results_per_platform, language): name
            for name in place_names
        }
        for fut in concurrent.futures.as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result(timeout=15)
                n = len(results[name].videos)
                log.info(f"[videos]   OK: {name} — {n} videos ({time.time()-t0:.1f}s)")
            except Exception as e:
                results[name] = RecommendationLinks()
                log.warning(f"[videos]   FAIL: {name} — {type(e).__name__}: {e}")
    log.info(f"[videos] batch done in {time.time()-t0:.1f}s")
    return results
