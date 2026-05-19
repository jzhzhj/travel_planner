"""Travelpayouts Data API — 获取真实机票价格数据。

机票: Aviasales prices_for_dates (缓存近期搜索价格)
酒店: Hotellook 已于 2025-10 关停，酒店部分回退到 LLM 推荐 + affiliate 链接
预订链接: Travelpayouts marker 提供 Aviasales / Booking.com affiliate 链接
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx


# ── 配置 ──────────────────────────────────────────────────────────────────


def _token() -> str:
    return os.environ.get("TRAVELPAYOUTS_API_TOKEN", "").strip()


def _marker() -> str:
    return os.environ.get("TRAVELPAYOUTS_MARKER", "").strip()


# ── 数据类（与 renderer.py 接口一致） ──────────────────────────────────────


@dataclass
class FlightDeal:
    airline: str  # IATA carrier code, e.g. "AC"
    airline_name: str
    origin: str
    destination: str
    price: int
    currency: str
    departure_at: str  # "2024-12-01"
    return_at: str
    transfers: int  # 0 = direct
    flight_number: str
    booking_url: str
    logo_url: str  # airline logo
    tier: str = ""  # budget / mid-range / premium


@dataclass
class HotelDeal:
    name: str
    hotel_id: int
    stars: int
    price_per_night: int
    currency: str
    location_name: str  # city/area
    photo_url: str
    booking_url: str
    rating: float = 0.0


@dataclass
class TravelDeals:
    flights: list[FlightDeal] = field(default_factory=list)
    hotels: list[HotelDeal] = field(default_factory=list)


# ── 航空公司名称映射 ────────────────────────────────────────────────────────

AIRLINE_NAMES = {
    "AC": "Air Canada", "NH": "ANA", "JL": "JAL", "CX": "Cathay Pacific",
    "SQ": "Singapore Airlines", "TG": "Thai Airways", "CZ": "China Southern",
    "CA": "Air China", "MU": "China Eastern", "BR": "EVA Air",
    "OZ": "Asiana", "KE": "Korean Air", "UA": "United", "AA": "American",
    "DL": "Delta", "WS": "WestJet", "TR": "Scoot", "MM": "Peach",
    "7C": "Jeju Air", "TW": "T'way Air", "LJ": "Jin Air",
    "HO": "Juneyao Airlines", "MF": "Xiamen Air", "HU": "Hainan Airlines",
    "3U": "Sichuan Airlines", "FM": "Shanghai Airlines",
    "BA": "British Airways", "LH": "Lufthansa", "AF": "Air France",
    "EK": "Emirates", "QR": "Qatar Airways", "EY": "Etihad",
    "AY": "Finnair", "SK": "SAS", "LX": "Swiss",
    "ZG": "Zipair", "IX": "Air India Express", "AI": "Air India",
    "CI": "China Airlines", "PR": "Philippine Airlines", "VN": "Vietnam Airlines",
    "QF": "Qantas", "NZ": "Air New Zealand", "JQ": "Jetstar",
    "FD": "Thai AirAsia", "AK": "AirAsia", "5J": "Cebu Pacific",
    "HA": "Hawaiian Airlines", "AS": "Alaska Airlines", "B6": "JetBlue",
    "WN": "Southwest", "NK": "Spirit", "F8": "Flair Airlines",
    "PD": "Porter Airlines", "TS": "Air Transat", "Y4": "Volaris",
    "KL": "KLM", "IB": "Iberia", "TK": "Turkish Airlines",
}


def _airline_name(code: str) -> str:
    return AIRLINE_NAMES.get(code, code)


# ── 机票 API（Travelpayouts prices_for_dates） ───────────────────────────


_FLIGHT_API = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"


def _fetch_flights_raw(
    origin: str,
    destination: str,
    departure_at: str,
    return_at: str = "",
    currency: str = "cad",
    limit: int = 10,
    direct: bool | None = None,
) -> list[dict]:
    """调用 Travelpayouts prices_for_dates, 返回原始 data 列表。"""
    token = _token()
    if not token:
        return []

    params: dict = {
        "origin": origin.upper(),
        "destination": destination.upper(),
        "departure_at": departure_at,
        "currency": currency.lower(),
        "sorting": "price",
        "limit": limit,
        "token": token,
    }
    if return_at:
        params["return_at"] = return_at
    if direct is not None:
        params["direct"] = str(direct).lower()

    try:
        resp = httpx.get(_FLIGHT_API, params=params, timeout=12)
        resp.raise_for_status()
        body = resp.json()
        if body.get("success"):
            return body.get("data", [])
    except Exception as e:
        print(f"[Travelpayouts] Flight search error: {e}")
    return []


def _fetch_flights_raw_sorted(
    origin: str,
    destination: str,
    departure_at: str,
    return_at: str = "",
    currency: str = "cad",
    limit: int = 30,
    sorting: str = "route",
) -> list[dict]:
    """用不同排序方式调 API，可能返回不同的缓存结果。"""
    token = _token()
    if not token:
        return []
    params: dict = {
        "origin": origin.upper(),
        "destination": destination.upper(),
        "departure_at": departure_at,
        "currency": currency.lower(),
        "sorting": sorting,
        "limit": limit,
        "token": token,
    }
    if return_at:
        params["return_at"] = return_at
    try:
        resp = httpx.get(_FLIGHT_API, params=params, timeout=12)
        resp.raise_for_status()
        body = resp.json()
        if body.get("success"):
            return body.get("data", [])
    except Exception as e:
        print(f"[Travelpayouts] Flight search ({sorting}) error: {e}")
    return []


def _raw_to_deal(item: dict, origin: str, destination: str, currency: str, marker: str) -> FlightDeal:
    """将 API 原始数据转为 FlightDeal。"""
    carrier_code = item.get("airline", "")
    price = int(item.get("price", 0))
    dep_at = item.get("departure_at", "")[:10]
    ret_at = item.get("return_at", "")
    ret_at = ret_at[:10] if ret_at else ""
    transfers = int(item.get("transfers", 0))
    flight_num = item.get("flight_number", "")
    if flight_num and carrier_code:
        flight_num = f"{carrier_code}{flight_num}"

    booking_url = ""
    link = item.get("link", "")
    if link:
        booking_url = (
            f"https://www.aviasales.com{link}"
            + (f"&marker={marker}" if marker else "")
        )
    elif marker and dep_at:
        dep_short = dep_at.replace("-", "")[4:]
        ret_short = ret_at.replace("-", "")[4:] if ret_at else ""
        search_path = f"{origin.upper()}{dep_short}{destination.upper()}{ret_short}1"
        booking_url = (
            f"https://tp.media/r?marker={marker}"
            f"&p=4114&u=https%3A%2F%2Fwww.aviasales.com%2Fsearch%2F{search_path}"
        )

    return FlightDeal(
        airline=carrier_code,
        airline_name=_airline_name(carrier_code),
        origin=origin.upper(),
        destination=destination.upper(),
        price=price,
        currency=currency.upper(),
        departure_at=dep_at,
        return_at=ret_at,
        transfers=transfers,
        flight_number=flight_num,
        booking_url=booking_url,
        logo_url=f"https://pics.avs.io/200/80/{carrier_code}.png",
    )


def _select_diverse(deals: list[FlightDeal], limit: int = 50) -> list[FlightDeal]:
    """标记 tier，去除完全重复项，保留尽可能多的结果供前端筛选。"""
    sorted_deals = sorted(deals, key=lambda d: d.price)

    # 按价格三等分标记 tier
    prices = [d.price for d in sorted_deals]
    if len(prices) >= 3:
        p33 = prices[len(prices) // 3]
        p66 = prices[2 * len(prices) // 3]
    else:
        p33 = p66 = prices[0] if prices else 0

    for d in sorted_deals:
        if d.price <= p33:
            d.tier = "budget"
        elif d.price <= p66:
            d.tier = "mid-range"
        else:
            d.tier = "premium"

    # 去除完全重复（同航司+同日期+同价格+同中转数）
    seen: set[tuple] = set()
    unique: list[FlightDeal] = []
    for d in sorted_deals:
        key = (d.airline, d.departure_at, d.price, d.transfers)
        if key not in seen:
            seen.add(key)
            unique.append(d)

    return unique[:limit]


def _nearby_months(date_str: str, spread: int = 2) -> list[str]:
    """返回日期所在月份及前后各 spread 个月的 YYYY-MM 列表。"""
    if len(date_str) < 7:
        return [date_str]
    year, month_num = int(date_str[:4]), int(date_str[5:7])
    months = []
    for offset in range(-spread, spread + 1):
        m = month_num + offset
        y = year
        while m < 1:
            m += 12
            y -= 1
        while m > 12:
            m -= 12
            y += 1
        months.append(f"{y}-{m:02d}")
    return months


def fetch_flight_deals(
    origin: str,
    destination: str,
    departure_date: str = "",
    return_date: str = "",
    currency: str = "cad",
    limit: int = 50,
) -> list[FlightDeal]:
    """根据用户确定的日期搜索机票，不搜无关月份。并发请求加速。"""
    import concurrent.futures

    if not origin or not destination or not departure_date:
        return []

    marker = _marker()
    all_deals: list[FlightDeal] = []
    seen: set[tuple] = set()

    def _collect(raw: list[dict]) -> None:
        for item in raw:
            d = _raw_to_deal(item, origin, destination, currency, marker)
            key = (d.airline, d.departure_at, d.price)
            if key not in seen:
                seen.add(key)
                all_deals.append(d)

    # ── 第一批：4 个独立查询并发执行 ──
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        f1 = pool.submit(_fetch_flights_raw, origin, destination, departure_date, return_date, currency, 30)
        f2 = pool.submit(_fetch_flights_raw, origin, destination, departure_date, "", currency, 30)
        f3 = pool.submit(
            _fetch_flights_raw, origin, destination,
            departure_date[:7] if len(departure_date) == 10 else departure_date,
            "", currency, 30,
        )
        f4 = pool.submit(_fetch_flights_raw, origin, destination, departure_date, return_date, currency, 30, True)

    for fut in (f1, f2, f3, f4):
        _collect(fut.result())

    # ── 第二批：结果太少时，换排序方式再试 ──
    if len(all_deals) < 15:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            extras = [
                pool.submit(_fetch_flights_raw_sorted,
                            origin, destination, departure_date, return_date, currency, 30, sk)
                for sk in ("route", "distance_unit_price")
            ]
        for fut in extras:
            _collect(fut.result())

    if not all_deals:
        return []

    return _select_diverse(all_deals, limit)


# ── 酒店 API（Hotellook 已关停，返回空列表） ──────────────────────────────
#
# Hotellook 于 2025-10-20 停止服务，所有 API 返回 404。
# 酒店信息由 LLM 在 TravelPlan 中生成（hotel_recommendations + 每天的 hotel 字段），
# renderer.py 在无 API 数据时自动使用 LLM 推荐渲染。


def fetch_hotel_deals(
    destination: str,
    check_in: str = "",
    check_out: str = "",
    currency: str = "cad",
    language: str = "en",
    limit: int = 5,
) -> list[HotelDeal]:
    """酒店 API 已停用，返回空列表。renderer 会回退到 LLM 推荐。"""
    return []


# ── 一次性获取所有 deals ────────────────────────────────────────────────────


def fetch_travel_deals(
    origin: str,
    destination: str,
    start_date: str = "",
    end_date: str = "",
    currency: str = "cad",
    language: str = "en",
) -> TravelDeals:
    """获取机票真实数据 + 酒店（当前为空）。"""
    flights = fetch_flight_deals(
        origin, destination,
        departure_date=start_date, return_date=end_date,
        currency=currency,
    )
    hotels = fetch_hotel_deals(
        destination,
        check_in=start_date, check_out=end_date,
        currency=currency, language=language,
    )
    return TravelDeals(flights=flights, hotels=hotels)
