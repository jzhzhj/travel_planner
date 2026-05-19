"""Google Routes API — 计算相邻活动间的通行时间。"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


@dataclass
class TransitInfo:
    origin: str          # place name
    destination: str     # place name
    duration_text: str   # "12 mins", "1 hour 5 mins"
    duration_mins: int   # 12, 65
    distance_text: str   # "950 m", "3.2 km"
    mode: str            # "walking" / "driving"


_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"


def _api_key() -> str:
    return os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()


def _format_duration(seconds: int) -> str:
    """秒数 → 人类可读字符串。"""
    mins = seconds // 60
    if mins < 60:
        return f"{mins} mins"
    h, m = divmod(mins, 60)
    return f"{h} hour{'s' if h > 1 else ''} {m} mins" if m else f"{h} hour{'s' if h > 1 else ''}"


def _format_distance(meters: int) -> str:
    """米 → 人类可读字符串。"""
    if meters < 1000:
        return f"{meters} m"
    return f"{meters / 1000:.1f} km"


def _fetch_route(
    origin_ll: tuple[float, float],
    dest_ll: tuple[float, float],
    mode: str = "WALK",
) -> dict | None:
    """调用 Routes API，返回第一条 route，或 None。"""
    key = _api_key()
    if not key:
        return None

    body = {
        "origin": {
            "location": {
                "latLng": {"latitude": origin_ll[0], "longitude": origin_ll[1]}
            }
        },
        "destination": {
            "location": {
                "latLng": {"latitude": dest_ll[0], "longitude": dest_ll[1]}
            }
        },
        "travelMode": mode,
    }
    headers = {
        "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters",
        "Content-Type": "application/json",
    }
    try:
        resp = httpx.post(_ROUTES_URL, json=body, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        routes = data.get("routes", [])
        if routes:
            return routes[0]
    except Exception as e:
        print(f"[directions] {mode} error: {e}")
    return None


def _parse_route(route: dict) -> tuple[int, int]:
    """从 route 中提取 duration_seconds 和 distance_meters。"""
    dur_str = route.get("duration", "0s")  # "300s"
    dur_secs = int(dur_str.rstrip("s")) if isinstance(dur_str, str) else 0
    dist_m = route.get("distanceMeters", 0)
    return dur_secs, dist_m


def _haversine_meters(ll1: tuple[float, float], ll2: tuple[float, float]) -> float:
    """两点间的直线距离（米），基于 Haversine 公式。"""
    import math
    lat1, lon1 = math.radians(ll1[0]), math.radians(ll1[1])
    lat2, lon2 = math.radians(ll2[0]), math.radians(ll2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(a))


def _estimate_transit(
    origin_name: str, dest_name: str,
    origin_ll: tuple[float, float], dest_ll: tuple[float, float],
) -> TransitInfo:
    """无 API 时的直线距离估算。步行 5 km/h，驾车 35 km/h（城市）。"""
    dist_m = _haversine_meters(origin_ll, dest_ll)
    # 城市道路距离约为直线的 1.3 倍
    road_m = dist_m * 1.3
    walk_mins = int(road_m / (5000 / 60))  # 5 km/h
    if walk_mins <= 25:
        return TransitInfo(
            origin=origin_name, destination=dest_name,
            duration_text=_format_duration(walk_mins * 60),
            duration_mins=walk_mins,
            distance_text=f"~{_format_distance(int(road_m))}",
            mode="walking",
        )
    drive_mins = max(1, int(road_m / (35000 / 60)))  # 35 km/h
    return TransitInfo(
        origin=origin_name, destination=dest_name,
        duration_text=f"~{_format_duration(drive_mins * 60)}",
        duration_mins=drive_mins,
        distance_text=f"~{_format_distance(int(road_m))}",
        mode="driving",
    )


def get_transit_between(
    origin_name: str,
    dest_name: str,
    origin_ll: tuple[float, float],
    dest_ll: tuple[float, float],
) -> TransitInfo | None:
    """获取两点间的通行时间。优先用 Routes API，失败时用直线距离估算。"""
    if not origin_ll[0] or not dest_ll[0]:
        return None

    # 无 API key → 直接用估算
    if not _api_key():
        return _estimate_transit(origin_name, dest_name, origin_ll, dest_ll)

    # 先查步行
    walk = _fetch_route(origin_ll, dest_ll, "WALK")
    if walk:
        dur_secs, dist_m = _parse_route(walk)
        walk_mins = dur_secs // 60
        if walk_mins <= 25:
            return TransitInfo(
                origin=origin_name,
                destination=dest_name,
                duration_text=_format_duration(dur_secs),
                duration_mins=walk_mins,
                distance_text=_format_distance(dist_m),
                mode="walking",
            )

    # 步行太远，用驾车
    drive = _fetch_route(origin_ll, dest_ll, "DRIVE")
    if drive:
        dur_secs, dist_m = _parse_route(drive)
        return TransitInfo(
            origin=origin_name,
            destination=dest_name,
            duration_text=_format_duration(dur_secs),
            duration_mins=dur_secs // 60,
            distance_text=_format_distance(dist_m),
            mode="driving",
        )

    # 步行超 25 分钟但驾车查询失败，仍返回步行结果
    if walk:
        dur_secs, dist_m = _parse_route(walk)
        return TransitInfo(
            origin=origin_name,
            destination=dest_name,
            duration_text=_format_duration(dur_secs),
            duration_mins=dur_secs // 60,
            distance_text=_format_distance(dist_m),
            mode="walking",
        )

    # API 全部失败 → 估算 fallback
    return _estimate_transit(origin_name, dest_name, origin_ll, dest_ll)


def get_day_transits(
    activities: list,
    places: dict,
) -> list[TransitInfo | None]:
    """计算一天内相邻活动间的通行时间。返回 len(activities)-1 个结果。"""
    import concurrent.futures

    if len(activities) < 2:
        return []

    pairs = []
    for i in range(len(activities) - 1):
        a = activities[i]
        b = activities[i + 1]
        pa = places.get(a.place_name)
        pb = places.get(b.place_name)
        if pa and pb and pa.lat and pb.lat:
            pairs.append((i, a.place_name, b.place_name,
                          (pa.lat, pa.lon), (pb.lat, pb.lon)))
        else:
            pairs.append(None)

    results: list[TransitInfo | None] = [None] * (len(activities) - 1)

    # 并发查询
    valid = [p for p in pairs if p is not None]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {}
        for idx, on, dn, oll, dll in valid:
            futures[pool.submit(get_transit_between, on, dn, oll, dll)] = idx
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                pass

    return results
