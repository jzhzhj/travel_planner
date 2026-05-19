"""天气查询工具 — 使用 wttr.in（免费，无需 API key）。"""

import requests
from langchain_core.tools import tool


@tool
def get_weather(city: str, days: int = 3) -> str:
    """查询指定城市未来几天的天气预报。适用于帮用户决定出行时间、建议穿搭和装备。

    Args:
        city: 城市名称，如 "Tokyo", "Paris", "北京"
        days: 预报天数，1-3天，默认3天
    """
    days = min(max(days, 1), 3)
    try:
        resp = requests.get(
            f"https://wttr.in/{city}",
            params={"format": "j1"},
            timeout=10,
            headers={"User-Agent": "AITravelPlanner/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        return f"无法获取 {city} 的天气信息: {e}"

    current = data.get("current_condition", [{}])[0]
    result_lines = [
        f"📍 {city} 当前天气:",
        f"  温度: {current.get('temp_C', '?')}°C (体感 {current.get('FeelsLikeC', '?')}°C)",
        f"  天气: {current.get('weatherDesc', [{}])[0].get('value', '未知')}",
        f"  湿度: {current.get('humidity', '?')}%",
        f"  风速: {current.get('windspeedKmph', '?')} km/h",
        "",
    ]

    forecasts = data.get("weather", [])[:days]
    for day_data in forecasts:
        date = day_data.get("date", "")
        max_temp = day_data.get("maxtempC", "?")
        min_temp = day_data.get("mintempC", "?")
        desc = day_data.get("hourly", [{}])[4].get("weatherDesc", [{}])[0].get("value", "")
        result_lines.append(f"📅 {date}: {min_temp}°C ~ {max_temp}°C, {desc}")

    return "\n".join(result_lines)
