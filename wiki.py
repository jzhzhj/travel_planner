"""
景点信息获取 — Wikipedia 简介/坐标 + Unsplash 高质量图片
图片优先级：Unsplash（专业摄影） > Wikipedia（百科配图）
"""

from __future__ import annotations

import logging
import os
import time
import urllib.parse
from dataclasses import dataclass

import requests

log = logging.getLogger("travel_planner")

WIKI_API = "https://zh.wikipedia.org/api/rest_v1"
WIKI_API_EN = "https://en.wikipedia.org/api/rest_v1"
UNSPLASH_API = "https://api.unsplash.com"

HEADERS = {"User-Agent": "AITravelPlanner/1.0"}


@dataclass
class PlaceInfo:
    name: str
    summary: str
    image_url: str | None
    lat: float | None = None
    lon: float | None = None


def _search_unsplash(query: str) -> str | None:
    """从 Unsplash 搜索高质量图片，返回 regular 尺寸（1080px）的 URL。"""
    api_key = os.environ.get("UNSPLASH_ACCESS_KEY", "")
    if not api_key:
        return None
    try:
        resp = requests.get(
            f"{UNSPLASH_API}/search/photos",
            params={"query": query, "per_page": 1, "orientation": "landscape"},
            headers={**HEADERS, "Authorization": f"Client-ID {api_key}"},
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        if results:
            return results[0]["urls"].get("regular")
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None


def fetch_place_info(place_name: str, language: str = "zh") -> PlaceInfo:
    """获取地点信息。图片：Unsplash 优先，Wikipedia fallback。"""
    t0 = time.time()
    # 根据语言决定 Wikipedia API 优先顺序
    if language == "en":
        api_order = [WIKI_API_EN, WIKI_API]
    else:
        api_order = [WIKI_API, WIKI_API_EN]

    # 先从 Wikipedia 拿简介、坐标和备用图片
    info = None
    for api_base in api_order:
        log.info(f"[wiki] {place_name}: fetching from {api_base.split('//')[1].split('/')[0]}...")
        t1 = time.time()
        info = _try_fetch(api_base, place_name)
        log.info(f"[wiki] {place_name}: wikipedia {'OK' if info and info.summary else 'MISS'} ({time.time()-t1:.1f}s)")
        if info and (info.summary or info.image_url):
            break

    if info is None:
        info = PlaceInfo(name=place_name, summary="", image_url=None)

    # Unsplash 高质量图片覆盖 Wikipedia 图片
    log.info(f"[wiki] {place_name}: fetching unsplash...")
    t2 = time.time()
    unsplash_url = _search_unsplash(place_name)
    if unsplash_url:
        info.image_url = unsplash_url
        log.info(f"[wiki] {place_name}: unsplash OK ({time.time()-t2:.1f}s)")
    else:
        log.info(f"[wiki] {place_name}: unsplash MISS ({time.time()-t2:.1f}s)")

    # 坐标 fallback
    if info.lat is None:
        log.info(f"[wiki] {place_name}: geocoding via nominatim...")
        t3 = time.time()
        coords = _geocode_nominatim(place_name)
        if coords:
            info.lat, info.lon = coords
            log.info(f"[wiki] {place_name}: nominatim OK ({time.time()-t3:.1f}s)")
        else:
            log.info(f"[wiki] {place_name}: nominatim MISS ({time.time()-t3:.1f}s)")

    log.info(f"[wiki] {place_name}: TOTAL {time.time()-t0:.1f}s")
    return info


def _try_fetch(api_base: str, place_name: str) -> PlaceInfo | None:
    encoded = urllib.parse.quote(place_name, safe="")
    url = f"{api_base}/page/summary/{encoded}"
    try:
        resp = requests.get(url, timeout=5, headers=HEADERS)
        if resp.status_code != 200:
            return None
        data = resp.json()
        summary = data.get("extract", "")
        image_url = data.get("thumbnail", {}).get("source")
        if not image_url:
            image_url = data.get("originalimage", {}).get("source")

        # 提取坐标
        coords = data.get("coordinates", {})
        lat = coords.get("lat")
        lon = coords.get("lon")

        return PlaceInfo(
            name=place_name, summary=summary, image_url=image_url, lat=lat, lon=lon
        )
    except (requests.RequestException, ValueError):
        return None


def _geocode_nominatim(place_name: str) -> tuple[float, float] | None:
    """Nominatim 地理编码 fallback（OpenStreetMap，免费）。"""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place_name, "format": "json", "limit": 1},
            timeout=5,
            headers=HEADERS,
        )
        if resp.status_code == 200:
            results = resp.json()
            if results:
                return float(results[0]["lat"]), float(results[0]["lon"])
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None


def fetch_places_batch(place_names: list[str]) -> dict[str, PlaceInfo]:
    """批量获取多个地点的信息。"""
    results = {}
    for name in place_names:
        results[name] = fetch_place_info(name)
    return results
