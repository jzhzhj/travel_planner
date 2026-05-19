"""旅行规划 Agent 的 function calling tools。"""

from rag.retriever import search_travel_knowledge
from tools.currency import convert_currency
from tools.holidays import get_holidays
from tools.place_search import search_place_info
from tools.weather import get_weather

ALL_TOOLS = [
    search_travel_knowledge,  # RAG 知识库检索（优先使用）
    get_weather,
    get_holidays,
    search_place_info,
    convert_currency,
]
