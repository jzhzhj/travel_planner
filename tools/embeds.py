"""TikTok / Instagram / 小红书 短视频嵌入搜索。

流程：DuckDuckGo 搜索 site:xxx → 提取 URL → 调 oEmbed API 获取缩略图和元数据。
中文模式：小红书；英文模式：TikTok + Instagram。
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field

import requests
from ddgs import DDGS

log = logging.getLogger("travel_planner")

HEADERS = {"User-Agent": "AITravelPlanner/1.0"}

# TikTok 视频 URL 正则（允许尾部 query params）
TIKTOK_URL_RE = re.compile(r"https?://(?:www\.)?tiktok\.com/@[\w.]+/video/\d+")
# Instagram Reel URL 正则
INSTA_REEL_RE = re.compile(r"https?://(?:www\.)?instagram\.com/(?:reel|p)/[\w-]+")
# 小红书笔记 URL 正则（explore / discovery/item / note 路径均为笔记）
XHS_URL_RE = re.compile(r"https?://(?:www\.)?xiaohongshu\.com/(?:explore|discovery/item|note)/[\w]+")


@dataclass
class EmbedInfo:
    url: str
    title: str
    author: str
    thumbnail_url: str
    platform: str  # "tiktok" | "instagram" | "xiaohongshu"


@dataclass
class PlaceEmbeds:
    tiktok: list[EmbedInfo] = field(default_factory=list)
    instagram: list[EmbedInfo] = field(default_factory=list)
    xiaohongshu: list[EmbedInfo] = field(default_factory=list)

    @property
    def all(self) -> list[EmbedInfo]:
        return self.tiktok + self.instagram + self.xiaohongshu

    @property
    def has_embeds(self) -> bool:
        return bool(self.tiktok or self.instagram or self.xiaohongshu)


# ── DuckDuckGo 搜索 ─────────────────────────────────────────────────────────

def _search_urls(query: str, site: str, max_results: int = 3) -> list[str]:
    """用 DuckDuckGo 搜索指定 site 的 URL。带重试和延迟。"""
    full_query = f"site:{site} {query}"
    for attempt in range(2):
        try:
            if attempt > 0:
                time.sleep(2)
            results = DDGS(timeout=8).text(full_query, max_results=max_results)
            urls = [r["href"] for r in results if r.get("href")]
            if urls:
                return urls
            log.info(f"[embeds] DDG empty result (attempt {attempt+1}): {full_query[:60]}")
        except Exception as e:
            log.warning(f"[embeds] DDG search failed (attempt {attempt+1}): {type(e).__name__}: {e}")
    return []


def _find_tiktok_urls(place_name: str, destination: str = "") -> list[str]:
    query = f"{place_name} {destination} travel".strip()
    urls = _search_urls(query, "tiktok.com", max_results=10)
    return [u for u in urls if TIKTOK_URL_RE.search(u)][:6]


def _find_instagram_urls(place_name: str, destination: str = "") -> list[str]:
    query = f"{place_name} {destination} travel".strip()
    urls = _search_urls(query, "instagram.com/reel", max_results=10)
    return [u for u in urls if INSTA_REEL_RE.search(u)][:6]


def _find_xiaohongshu_urls(place_name: str, destination: str = "") -> list[str]:
    """搜索某景点的小红书笔记 URL。"""
    query = f"{place_name} {destination} 旅行攻略".strip()
    urls = _search_urls(query, "xiaohongshu.com", max_results=10)
    return [u for u in urls if _is_xhs_note_url(u)][:6]


def _is_xhs_note_url(url: str) -> bool:
    """判断 URL 是否为小红书笔记（排除用户主页、商品页、搜索页等）。"""
    if "xiaohongshu.com" not in url:
        return False
    if XHS_URL_RE.search(url):
        return True
    # 排除非笔记页面
    non_note = ("/user/", "/goods/", "/brand/", "/search/", "/store/", "/cart", "/login")
    return not any(seg in url for seg in non_note)


# ── oEmbed API ───────────────────────────────────────────────────────────────

def _tiktok_oembed(url: str) -> EmbedInfo | None:
    try:
        resp = requests.get(
            "https://www.tiktok.com/oembed",
            params={"url": url},
            timeout=5,
            headers=HEADERS,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return EmbedInfo(
            url=url,
            title=data.get("title", "")[:80],
            author=data.get("author_name", ""),
            thumbnail_url=data.get("thumbnail_url", ""),
            platform="tiktok",
        )
    except (requests.RequestException, ValueError):
        return None


def _instagram_oembed(url: str) -> EmbedInfo | None:
    token = os.environ.get("INSTAGRAM_TOKEN", "")
    if not token:
        return EmbedInfo(
            url=url,
            title="Instagram Reel",
            author="",
            thumbnail_url="",
            platform="instagram",
        )
    try:
        resp = requests.get(
            "https://graph.facebook.com/v18.0/instagram_oembed",
            params={"url": url, "access_token": token},
            timeout=5,
            headers=HEADERS,
        )
        if resp.status_code != 200:
            return EmbedInfo(url=url, title="Instagram Reel", author="", thumbnail_url="", platform="instagram")
        data = resp.json()
        return EmbedInfo(
            url=url,
            title=data.get("title", "Instagram Reel")[:80],
            author=data.get("author_name", ""),
            thumbnail_url=data.get("thumbnail_url", ""),
            platform="instagram",
        )
    except (requests.RequestException, ValueError):
        return None


def _xiaohongshu_embed(url: str, search_title: str = "") -> EmbedInfo:
    """小红书没有 oEmbed API，用搜索标题作为卡片信息。"""
    return EmbedInfo(
        url=url,
        title=search_title or "小红书笔记",
        author="",
        thumbnail_url="",
        platform="xiaohongshu",
    )


# ── 批量搜索 ─────────────────────────────────────────────────────────────────

def search_embeds(place_name: str, destination: str = "", language: str = "zh") -> PlaceEmbeds:
    """搜索某景点的短视频/笔记嵌入。中文→小红书，英文→TikTok + Instagram。"""
    t0 = time.time()
    log.info(f"[embeds] searching: {place_name}...")
    embeds = PlaceEmbeds()

    if language == "zh":
        # 小红书 — 单次查询避免 DuckDuckGo 限流
        queries = [
            f"{place_name} {destination} 旅行攻略",
        ]
        # 相关性关键词：标题必须包含景点名或目的地
        relevance_keys = {place_name}
        if destination:
            relevance_keys.add(destination)

        seen_urls: set[str] = set()
        for query in queries:
            if len(embeds.xiaohongshu) >= 6:
                break
            try:
                results = DDGS(timeout=5).text(f"site:xiaohongshu.com {query.strip()}", max_results=10)
            except Exception:
                continue
            for r in results:
                if len(embeds.xiaohongshu) >= 6:
                    break
                href = r.get("href", "")
                if href in seen_urls:
                    continue
                seen_urls.add(href)
                if not _is_xhs_note_url(href):
                    continue
                title = r.get("title", "")
                body = r.get("body", "")
                text = f"{title} {body}"
                # 标题或摘要必须包含至少一个相关关键词
                if not any(k in text for k in relevance_keys):
                    continue
                embeds.xiaohongshu.append(_xiaohongshu_embed(href, title or "小红书笔记"))
    else:
        # TikTok
        for url in _find_tiktok_urls(place_name, destination):
            info = _tiktok_oembed(url)
            if info:
                embeds.tiktok.append(info)

        # 延迟避免 DuckDuckGo 限流
        if embeds.tiktok:
            time.sleep(1)

        # Instagram
        for url in _find_instagram_urls(place_name, destination):
            info = _instagram_oembed(url)
            if info:
                embeds.instagram.append(info)

    log.info(f"[embeds] done: {place_name} ({time.time()-t0:.1f}s)")
    return embeds


def search_embeds_batch(
    place_names: list[str], destination: str = "", language: str = "zh"
) -> dict[str, PlaceEmbeds]:
    """批量搜索多个景点的短视频嵌入。限制并发避免 DuckDuckGo 限流。"""
    import concurrent.futures
    t0 = time.time()
    log.info(f"[embeds] batch start — {len(place_names)} places")
    results: dict[str, PlaceEmbeds] = {}
    # DuckDuckGo 极易限流，英文模式每个景点 2 次请求，串行化保证成功率
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        futures = {
            pool.submit(search_embeds, name, destination, language): name
            for name in place_names
        }
        for fut in concurrent.futures.as_completed(futures, timeout=30):
            name = futures[fut]
            try:
                results[name] = fut.result(timeout=10)
                log.info(f"[embeds]   OK: {name} ({time.time()-t0:.1f}s)")
            except Exception as e:
                results[name] = PlaceEmbeds()
                log.warning(f"[embeds]   FAIL: {name} — {type(e).__name__}: {e}")
    log.info(f"[embeds] batch done in {time.time()-t0:.1f}s")
    return results
