"""
节假日查询工具

数据源（按优先级）：
  1. Nager.Date — 免费，无需 key，覆盖 121 个国家
  2. Calendarific — 免费额度 1000 次/月，覆盖所有国家（需 API key）
"""

from __future__ import annotations

import os

import requests
from langchain_core.tools import tool

# 常见旅行目的地映射（中文/英文 → ISO 3166-1 alpha-2）
COUNTRY_CODES = {
    "日本": "JP", "japan": "JP",
    "泰国": "TH", "thailand": "TH",
    "韩国": "KR", "korea": "KR", "south korea": "KR",
    "美国": "US", "usa": "US", "united states": "US",
    "法国": "FR", "france": "FR",
    "意大利": "IT", "italy": "IT",
    "西班牙": "ES", "spain": "ES",
    "英国": "GB", "uk": "GB", "united kingdom": "GB",
    "德国": "DE", "germany": "DE",
    "澳大利亚": "AU", "australia": "AU",
    "新西兰": "NZ", "new zealand": "NZ",
    "加拿大": "CA", "canada": "CA",
    "新加坡": "SG", "singapore": "SG",
    "马来西亚": "MY", "malaysia": "MY",
    "印度尼西亚": "ID", "indonesia": "ID",
    "越南": "VN", "vietnam": "VN",
    "土耳其": "TR", "turkey": "TR", "türkiye": "TR",
    "埃及": "EG", "egypt": "EG",
    "墨西哥": "MX", "mexico": "MX",
    "巴西": "BR", "brazil": "BR",
    "荷兰": "NL", "netherlands": "NL",
    "瑞士": "CH", "switzerland": "CH",
    "奥地利": "AT", "austria": "AT",
    "葡萄牙": "PT", "portugal": "PT",
    "希腊": "GR", "greece": "GR",
    "捷克": "CZ", "czech": "CZ",
    "中国": "CN", "china": "CN",
    "台湾": "TW", "taiwan": "TW",
    "香港": "HK", "hong kong": "HK",
    "冰岛": "IS", "iceland": "IS",
    "挪威": "NO", "norway": "NO",
    "瑞典": "SE", "sweden": "SE",
    "芬兰": "FI", "finland": "FI",
    "丹麦": "DK", "denmark": "DK",
    "波兰": "PL", "poland": "PL",
    "匈牙利": "HU", "hungary": "HU",
    "克罗地亚": "HR", "croatia": "HR",
    "摩洛哥": "MA", "morocco": "MA",
    "南非": "ZA", "south africa": "ZA",
    "阿根廷": "AR", "argentina": "AR",
    "秘鲁": "PE", "peru": "PE",
    "哥伦比亚": "CO", "colombia": "CO",
    "菲律宾": "PH", "philippines": "PH",
    "印度": "IN", "india": "IN",
    "斯里兰卡": "LK", "sri lanka": "LK",
    "柬埔寨": "KH", "cambodia": "KH",
    "尼泊尔": "NP", "nepal": "NP",
}


def _resolve_country_code(country: str) -> str | None:
    """将国家名称解析为 ISO 国家代码。"""
    key = country.strip().lower()
    if key in COUNTRY_CODES:
        return COUNTRY_CODES[key]
    if len(key) == 2:
        return key.upper()
    for name, code in COUNTRY_CODES.items():
        if key in name.lower() or name.lower() in key:
            return code
    return None


def _format_holidays(holidays: list[dict], country: str, code: str, year: int) -> str:
    """将假期列表格式化为可读字符串。"""
    if not holidays:
        return f"{country} {year} 年暂无公共假期数据。"

    lines = [f"📅 {country}（{code}）{year} 年公共假期：\n"]
    for h in holidays:
        date = h.get("date", "")
        local_name = h.get("localName") or h.get("local_name", "")
        name = h.get("name", "")
        name_display = f"{local_name}（{name}）" if local_name and local_name != name else name
        lines.append(f"  {date}  {name_display}")

    return "\n".join(lines)


def _fetch_nager(code: str, year: int) -> list[dict] | None:
    """尝试从 Nager.Date 获取假期数据。"""
    try:
        resp = requests.get(
            f"https://date.nager.at/api/v3/PublicHolidays/{year}/{code}",
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, list) and data:
            return data
        return None
    except (requests.RequestException, ValueError):
        return None


def _fetch_calendarific(code: str, year: int) -> list[dict] | None:
    """尝试从 Calendarific 获取假期数据（需要 API key）。"""
    api_key = os.getenv("CALENDARIFIC_API_KEY", "")
    if not api_key:
        return None

    try:
        resp = requests.get(
            "https://calendarific.com/api/v2/holidays",
            params={
                "api_key": api_key,
                "country": code,
                "year": year,
                "type": "national",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    raw = data.get("response", {}).get("holidays", [])
    if not raw:
        return None

    # 转换为统一格式
    return [
        {
            "date": h.get("date", {}).get("iso", "")[:10],
            "name": h.get("name", ""),
            "localName": h.get("name", ""),
        }
        for h in raw
    ]


@tool
def get_holidays(country: str, year: int) -> str:
    """查询指定国家某年的公共假期和节日。适用于帮用户避开或赶上特定节日、了解当地放假安排。

    Args:
        country: 国家名称（中文或英文均可），如 "日本", "France", "泰国"
        year: 年份，如 2025, 2026
    """
    code = _resolve_country_code(country)
    if not code:
        return f"未能识别国家「{country}」。请尝试使用标准国家名称或 ISO 代码（如 JP, TH, FR）。"

    # 优先 Nager.Date（免费），fallback 到 Calendarific
    holidays = _fetch_nager(code, year)
    if holidays is None:
        holidays = _fetch_calendarific(code, year)

    if holidays is None:
        return (
            f"未找到 {country}（{code}）{year} 年的假期数据。"
            f"\n提示：部分国家需要配置 CALENDARIFIC_API_KEY 才能查询。"
        )

    return _format_holidays(holidays, country, code, year)
