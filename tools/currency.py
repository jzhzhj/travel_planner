"""汇率换算工具 — 使用 frankfurter.dev（免费，无需 API key）。"""

import requests
from langchain_core.tools import tool


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """将一种货币换算为另一种货币。适用于帮用户估算旅行预算。

    Args:
        amount: 金额
        from_currency: 源货币代码，如 "CNY", "USD", "EUR"
        to_currency: 目标货币代码，如 "JPY", "THB", "KRW"
    """
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    try:
        resp = requests.get(
            "https://api.frankfurter.dev/v1/latest",
            params={"amount": amount, "from": from_currency, "to": to_currency},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        return f"汇率查询失败: {e}"

    rates = data.get("rates", {})
    if to_currency not in rates:
        return f"不支持的货币对: {from_currency} → {to_currency}"

    converted = rates[to_currency]
    rate = converted / amount if amount else 0

    return (
        f"💱 {amount:,.2f} {from_currency} = {converted:,.2f} {to_currency}\n"
        f"  汇率: 1 {from_currency} = {rate:,.4f} {to_currency}\n"
        f"  数据来源: 欧洲央行 (ECB)"
    )
