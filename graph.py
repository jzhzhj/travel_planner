"""
AI Travel Planner — LangGraph 实现

图的流程:
  1. chat: 和用户自由对话（可调用工具），收集旅行需求
  2. tool_executor: 如果 chat 调用了工具，执行工具并返回结果
  3. extract_context: 从最新消息中增量提取结构化用户上下文
  4. check_info: 纯 Python 检查必填字段是否齐全（无 LLM 调用）
  5. generate_plan: 生成结构化旅行计划
  6. enrich_plan: 从 Wikipedia 拉取景点简介和图片，渲染 HTML
"""

from __future__ import annotations

import json
import time
import logging

log = logging.getLogger("travel_planner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from models import (
    ExtractedContextPatch, ExtractedIntakeProfile,
    ProceedToPlanParams, ShowSurveyParams,
    TravelPlan, UserContext,
)
from profiles import TravelProfile, build_strategy_prompt
from prompts import prompt_manager
from renderer import render_plan
from tools import ALL_TOOLS
from tools.embeds import search_embeds_batch
from tools.links import search_videos_batch
from tools.directions import TransitInfo, get_day_transits, _haversine_meters
from tools.places import search_destination_pois, format_pois_for_prompt
from tools.travel_api import TravelDeals, fetch_travel_deals
from wiki import PlaceInfo, fetch_place_info

load_dotenv()

# ── 上下文管理配置 ─────────────────────────────────────────────────────────

CHAT_WINDOW_SIZE = 20   # 滑动窗口保留最近 20 条消息
PLAN_RECENT_MSGS = 10   # 生成计划时保留最近 10 条原始消息
RAG_MAX_CHARS = 1500    # 每次 RAG 检索结果最多保留的字符数


# ── 修改意图检测 ──────────────────────────────────────────────────────────────

# 用户想修改已有计划时常见的关键词。LLM 设置 wants_modification 标志不可靠，
# 所以这里用关键词扫描作为后备 / 补充信号。
MODIFY_KEYWORDS = (
    # 中文 — 动词
    "改", "换", "修改", "删", "增加", "去掉", "加上", "加个",
    "替换", "调整", "更新", "重新", "不要", "不想",
    "变成", "变为", "改成", "换成",
    "缩短", "延长", "拉长", "压缩", "精简", "简化",
    "减少", "少一", "少个", "多一", "多个",
    # 中文 — 短语
    "加一天", "减一天", "加两天", "减两天", "少一天", "多一天",
    "两天", "三天", "四天", "五天", "六天", "七天",
    "一天变", "天改", "天变",
    # 英文 — 动词
    "change", "modify", "update", "replace", "remove", "delete",
    "swap", "adjust", "instead", "shorten", "extend", "shrink",
    "condense", "trim", "cut", "make it",
    # 英文 — 短语
    "add a day", "add one more day", "remove a day", "one less day",
    "one more day", "fewer day", "less day",
    "2 day", "3 day", "4 day", "5 day", "two day", "three day",
)


def detect_modification_intent(messages: list[BaseMessage]) -> bool:
    """扫描最近一条用户消息，判断是否含修改已有计划的意图关键词。"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.content:
            text = m.content.lower()
            return any(kw in text for kw in MODIFY_KEYWORDS)
    return False


# ── State ────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan_generated: bool
    plan_data: TravelPlan | None
    html_path: str
    language: str   # "zh" or "en"
    currency: str   # "CAD" or "USD"
    # 结构化用户上下文（替代 conversation_summary + travel_profile）
    user_context: UserContext | None
    # check_info 路由结果（纯 Python，不产生消息）
    check_result: str   # "generate" | "continue"
    # initial_intake 结果
    intake_result: str | None  # "show_survey" | "proceed_to_plan" | None
    # Google Places POI 数据（generate_plan 时拉取，enrich_plan 校验用）
    google_pois: dict | None
    # 富化数据缓存
    enrich_places: dict | None
    enrich_links: dict | None
    enrich_embeds: dict | None
    enrich_deals: TravelDeals | None
    enrich_transits: dict | None


# ── LLM ──────────────────────────────────────────────────────────────────────

def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def get_plan_llm():
    """生成计划专用，用 gpt-4.1 获得更高输出上限 (32k)。"""
    return ChatOpenAI(model="gpt-4.1", temperature=0.7, max_tokens=32768)


def get_llm_with_tools():
    return get_llm().bind_tools(ALL_TOOLS)


# ── Initial Intake（首条消息 LLM tool calling + 关键词并行） ─────────────────

import re
import concurrent.futures


def _keyword_extract(text: str) -> dict:
    """纯关键词/正则提取旅行画像，瞬间返回。与 LLM 并行跑，互补结果。"""
    lower = text.lower()
    result: dict = {}

    def _has(patterns):
        for p in patterns:
            if isinstance(p, re.Pattern):
                if p.search(lower):
                    return True
            elif p in text:
                return True
        return False

    # ── destination: 常见句式 ──
    # "从X去Y" / "X to Y" 优先匹配
    m2 = re.search(r'从\s*(\S{2,8})\s*(?:去|到|飞)\s*([^\s,，。！!?？、玩住呆待逛穷\d]{2,10})', text)
    if not m2:
        # "Vancouver to Seattle" — 排除 go/trip/travel/fly 等动词
        m2 = re.search(r'\b(?!go|trip|travel|fly|heading|get|want|plan|need)([A-Z][a-z]{2,20})\s+to\s+([A-Z][a-z]{2,20})\b', text)
    if m2:
        result["departure_city"] = m2.group(1).strip()
        result["destination"] = m2.group(2).strip()
    else:
        # 中文: "去东京" "想去巴黎"
        m = re.search(r'(?:去|到|飞)\s*([^\s,，。！!?？、玩住呆待逛穷\d]{2,10})', text)
        if m:
            result["destination"] = m.group(1).strip()
        else:
            # 英文: "go to Tokyo" "trip to Paris" "visit Bali"
            m = re.search(r'(?:visit|go\s+to|trip\s+to|travel\s+to|heading\s+to)\s+([A-Za-z][A-Za-z\s]{1,20}?)(?:\s+for\b|\s*,|\s+\d|\s*$)', text, re.I)
            if m:
                result["destination"] = m.group(1).strip()
            else:
                # "in Bali" / "in Paris"
                m = re.search(r'\bin\s+([A-Z][a-z]{2,15})\b', text)
                if m:
                    result["destination"] = m.group(1).strip()

    # ── duration_days ──
    m = re.search(r'(\d{1,2})\s*(?:天|日|days?|nights?)', text, re.I)
    if m:
        result["duration_days"] = int(m.group(1))

    # ── travel_style ──
    if _has([re.compile(r'\b(solo|alone|by myself|1\s*person|1\s*people|just me)\b'), '一个人', '独自', '自己去']):
        result["travel_style"] = "solo"
    elif _has([re.compile(r'\b(girlfriend|boyfriend|wife|husband|partner|honeymoon|couple)\b'),
               '情侣', '夫妻', '男友', '女友', '老公', '老婆', '蜜月', '对象']):
        result["travel_style"] = "couple"
    elif _has([re.compile(r'\b(family|kids?|children|parents)\b'),
               '家人', '家庭', '亲子', '带孩子', '带父母', '一家人']):
        result["travel_style"] = "family"
    elif _has([re.compile(r'\b(friends?|buddies|mates)\b'),
               '朋友', '闺蜜', '兄弟', '同事', '同学']):
        result["travel_style"] = "friends"

    # ── budget_tier ──
    if _has([re.compile(r'\b(luxury|5[- ]?star|high[- ]?end|splurge)\b'),
             '奢华', '豪华', '高端', '五星']):
        result["budget_tier"] = "luxury"
    elif _has([re.compile(r'\b(budget|cheap|backpack|hostel)\b'),
               '穷游', '便宜', '省钱', '背包', '青旅']):
        result["budget_tier"] = "budget"

    # ── pace ──
    if _has([re.compile(r'\b(packed|intensive|see everything)\b'),
             '暴走', '打卡', '紧凑']):
        result["pace"] = "intensive"
    elif _has([re.compile(r'\b(relaxed|slow|chill|laid[- ]?back)\b'),
               '悠闲', '放松', '慢', '度假']):
        result["pace"] = "relaxed"

    # ── interests ──
    interests = []
    interest_map = [
        ("food",      [re.compile(r'\b(food|cuisine|eat|restaurant|dining)\b'), '美食', '吃', '餐厅']),
        ("culture",   [re.compile(r'\b(cultur|histor|museum|temple|shrine)\b'), '文化', '历史', '博物馆']),
        ("nature",    [re.compile(r'\b(nature|hiking?|outdoor|mountain|beach)\b'), '自然', '户外', '徒步', '海边']),
        ("shopping",  [re.compile(r'\b(shop|mall|market)\b'), '购物', '逛街', '买']),
        ("adventure", [re.compile(r'\b(adventure|thrill|diving?|surfing?|skiing?)\b'), '冒险', '潜水', '滑雪']),
        ("instagrammable", [re.compile(r'\b(instagram|photo spot|aesthetic)\b'), '网红', '出片', '拍照']),
    ]
    for key, patterns in interest_map:
        if _has(patterns):
            interests.append(key)
    if interests:
        result["interests"] = interests

    return result


def _merge_keyword_into_profile(profile_dict: dict, kw: dict) -> dict:
    """将关键词提取结果合并到 LLM profile（LLM 优先，关键词补漏）。"""
    for key, val in kw.items():
        if val and not profile_dict.get(key):
            profile_dict[key] = val
    return profile_dict


_PROFILE_FIELDS = ("travel_style", "budget_tier", "interests")
_MIN_PROFILE_COUNT = 1  # 画像字段至少填 1/3 才能直接生成（dest+days 是硬性要求）


def _is_ready_to_proceed(profile: ExtractedIntakeProfile) -> bool:
    """判断是否信息充足可以直接生成计划：dest + days + 至少 2 个画像字段。"""
    if not profile.destination or not profile.duration_days:
        return False
    filled = 0
    if profile.travel_style:
        filled += 1
    if profile.budget_tier:
        filled += 1
    if profile.interests:
        filled += 1
    return filled >= _MIN_PROFILE_COUNT


def _compute_missing(profile: ExtractedIntakeProfile) -> list[str]:
    """计算所有缺失字段列表。"""
    missing = []
    if not profile.destination:
        missing.append("destination")
    if not profile.duration_days:
        missing.append("duration_days")
    if not profile.travel_style:
        missing.append("travel_style")
    if not profile.budget_tier:
        missing.append("budget_tier")
    if not profile.interests:
        missing.append("interests")
    return missing


def _intake_profile_to_context(profile: ExtractedIntakeProfile) -> UserContext:
    """将 LLM 提取的 intake profile 转为 UserContext，缺失可选字段用默认值。"""
    ctx = UserContext(
        destination=profile.destination,
        duration_days=profile.duration_days,
        travel_style=profile.travel_style or "friends",
        budget_tier=profile.budget_tier or "comfort",
        pace=profile.pace or "mixed",
        interests=profile.interests if profile.interests else ["food", "culture"],
        region=profile.region,
        departure_city=profile.departure_city,
        travel_dates=profile.travel_dates,
        budget_amount=profile.budget_amount,
        group_size=profile.group_size,
        special_needs=profile.special_needs or [],
    )
    return ctx


def initial_intake(state: State) -> State:
    """分析用户首条消息：关键词匹配 + LLM tool calling 并行，合并结果。"""
    from langchain_core.tools import StructuredTool

    t0 = time.time()
    lang = state.get("language", "zh")
    writer = get_stream_writer()

    # 取用户首条消息文本
    user_text = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage) and m.content:
            user_text = m.content
            break

    prompt_text = prompt_manager.load("initial_intake", lang=lang)

    # 定义两个工具（函数体是空的，只用 schema）
    def _noop(**kwargs):
        return "ok"

    survey_desc = ("信息不足时调用，弹出问卷让用户补充缺失字段。" if lang == "zh"
                    else "Call when info is insufficient; show a survey for the user to fill in missing fields.")
    proceed_desc = ("信息充足时调用，直接开始生成旅行计划。" if lang == "zh"
                    else "Call when info is sufficient; proceed to generate the travel plan.")
    show_survey_tool = StructuredTool.from_function(
        func=_noop,
        name="show_survey",
        description=survey_desc,
        args_schema=ShowSurveyParams,
    )
    proceed_tool = StructuredTool.from_function(
        func=_noop,
        name="proceed_to_plan",
        description=proceed_desc,
        args_schema=ProceedToPlanParams,
    )

    llm = get_llm().bind_tools(
        [show_survey_tool, proceed_tool],
        tool_choice="required",
    )
    messages = [SystemMessage(content=prompt_text)] + state["messages"]

    # ── 真正并行：关键词提取 + LLM tool calling 同时跑 ──
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        kw_future = pool.submit(_keyword_extract, user_text)
        llm_future = pool.submit(llm.invoke, messages)

    kw_result = kw_future.result()
    log.info(f"[initial_intake] keyword extract: {kw_result}")

    llm_response = None
    llm_error = None
    try:
        llm_response = llm_future.result()
    except Exception as e:
        log.error(f"[initial_intake] LLM error: {e}")
        llm_error = e

    # ── LLM 失败 → 用关键词结果 fallback ──
    if llm_error or not llm_response or not getattr(llm_response, "tool_calls", None):
        log.warning("[initial_intake] LLM failed or no tool call, using keyword fallback")
        profile = ExtractedIntakeProfile.model_validate(kw_result)

        if _is_ready_to_proceed(profile):
            ctx = _intake_profile_to_context(profile)
            for k, v in kw_result.items():
                if v:
                    ctx.source[k] = "keyword"
            greeting = "收到！马上为你生成旅行计划！" if lang == "zh" else "Got it! Generating your travel plan now!"
            log.info(f"[initial_intake] keyword fallback → proceed, dest={ctx.destination}, {time.time()-t0:.1f}s")
            return {
                "messages": [AIMessage(content=greeting)],
                "user_context": ctx,
                "intake_result": "proceed_to_plan",
                "check_result": "generate",
            }
        else:
            missing = _compute_missing(profile)
            writer({
                "type": "show_survey",
                "missing_fields": missing,
                "extracted": kw_result,
            })
            greeting_text = (llm_response.content if llm_response and llm_response.content
                             else ("让我了解一下你的旅行偏好！" if lang == "zh"
                                   else "Let me learn about your travel preferences!"))
            return {
                "messages": [AIMessage(content=greeting_text)],
                "intake_result": "show_survey",
            }

    # ── LLM 成功：合并关键词结果 ──
    tool_call = llm_response.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name == "show_survey":
        params = ShowSurveyParams.model_validate(tool_args)
        # 合并：关键词可能补上 LLM 漏掉的字段
        profile_dict = params.extracted_profile.model_dump()
        _merge_keyword_into_profile(profile_dict, kw_result)
        merged_profile = ExtractedIntakeProfile.model_validate(profile_dict)

        # 合并后信息够了 → 升级为 proceed_to_plan
        if _is_ready_to_proceed(merged_profile):
            ctx = _intake_profile_to_context(merged_profile)
            for k, v in merged_profile.model_dump(exclude_none=True).items():
                if v and v != []:
                    ctx.source[k] = "keyword" if k in kw_result else "llm"
            log.info(f"[initial_intake] keyword+LLM merged → proceed, dest={ctx.destination}, "
                     f"days={ctx.duration_days}, {time.time()-t0:.1f}s")
            return {
                "messages": [AIMessage(content=params.greeting)],
                "user_context": ctx,
                "intake_result": "proceed_to_plan",
                "check_result": "generate",
            }

        # 仍然不够 → 弹问卷
        actual_missing = _compute_missing(merged_profile)
        ctx = _intake_profile_to_context(merged_profile)
        merged_dump = merged_profile.model_dump(exclude_none=True)
        writer({
            "type": "show_survey",
            "missing_fields": actual_missing,
            "extracted": merged_dump,
        })
        log.info(f"[initial_intake] show_survey (merged), missing={actual_missing}, {time.time()-t0:.1f}s")
        return {
            "messages": [AIMessage(content=params.greeting)],
            "user_context": ctx,
            "intake_result": "show_survey",
        }
    else:
        # LLM 说 proceed_to_plan — 合并关键词后再验证一次
        params = ProceedToPlanParams.model_validate(tool_args)
        profile_dict = params.complete_profile.model_dump()
        _merge_keyword_into_profile(profile_dict, kw_result)
        merged_profile = ExtractedIntakeProfile.model_validate(profile_dict)

        # LLM 可能误判，再检查一次
        if not _is_ready_to_proceed(merged_profile):
            actual_missing = _compute_missing(merged_profile)
            ctx = _intake_profile_to_context(merged_profile)
            writer({
                "type": "show_survey",
                "missing_fields": actual_missing,
                "extracted": merged_profile.model_dump(exclude_none=True),
            })
            log.info(f"[initial_intake] LLM said proceed but not ready, missing={actual_missing}, {time.time()-t0:.1f}s")
            return {
                "messages": [AIMessage(content=params.acknowledgment)],
                "user_context": ctx,
                "intake_result": "show_survey",
            }

        ctx = _intake_profile_to_context(merged_profile)
        for k, v in merged_profile.model_dump(exclude_none=True).items():
            if v and v != []:
                ctx.source[k] = "keyword" if k in kw_result else "llm"
        log.info(f"[initial_intake] proceed_to_plan (merged), dest={ctx.destination}, "
                 f"days={ctx.duration_days}, {time.time()-t0:.1f}s")
        return {
            "messages": [AIMessage(content=params.acknowledgment)],
            "user_context": ctx,
            "intake_result": "proceed_to_plan",
            "check_result": "generate",
        }


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _trim_messages(messages: list[BaseMessage], max_count: int) -> list[BaseMessage]:
    """滑动窗口截断，确保不会从 ToolMessage 开头。"""
    if len(messages) <= max_count:
        return messages
    trimmed = messages[-max_count:]
    while trimmed and isinstance(trimmed[0], ToolMessage):
        trimmed = trimmed[1:]
    return trimmed


def _extract_rag_context(messages: list[BaseMessage]) -> str:
    """从所有 ToolMessage 中提取 RAG 检索结果，去重合并。"""
    rag_pieces: list[str] = []
    seen = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content:
            content = msg.content
            if "知识库检索结果" in content or "Knowledge base" in content:
                if len(content) > RAG_MAX_CHARS:
                    content = content[:RAG_MAX_CHARS] + "\n...(已截断)"
                key = content[:200]
                if key not in seen:
                    seen.add(key)
                    rag_pieces.append(content)
    return "\n\n".join(rag_pieces) if rag_pieces else ""


def _context_to_dict(ctx: UserContext | None) -> dict:
    """UserContext → 只包含已填字段的精简 dict（给 prompt 用）。"""
    if ctx is None:
        return {}
    d = ctx.model_dump(exclude={"source", "wants_modification", "narrative_summary"})
    return {k: v for k, v in d.items() if v is not None and v != [] and v != ""}


def _profile_from_context(ctx: UserContext) -> TravelProfile:
    """从 UserContext 构建 TravelProfile（用于策略注入）。"""
    return TravelProfile(
        travel_style=ctx.travel_style or "friends",
        budget_tier=ctx.budget_tier or "comfort",
        pace=ctx.pace or "mixed",
        interests=ctx.interests or ["food", "culture"],
        duration_days=ctx.duration_days or 5,
        region=ctx.region or "",
    )


def _format_context_block(ctx: UserContext, lang: str) -> str:
    """将 UserContext 格式化为可注入 prompt 的文本块。"""
    d = _context_to_dict(ctx)
    if not d:
        return ""

    if lang == "zh":
        label_map = {
            "destination": "目的地", "duration_days": "天数",
            "travel_style": "出行方式", "budget_tier": "预算档次",
            "pace": "节奏", "interests": "兴趣",
            "region": "区域", "departure_city": "出发城市",
            "travel_dates": "出行日期", "budget_amount": "预算金额",
            "group_size": "人数", "special_needs": "特殊需求",
        }
        header = "用户需求（结构化）"
    else:
        label_map = {
            "destination": "Destination", "duration_days": "Duration (days)",
            "travel_style": "Travel style", "budget_tier": "Budget tier",
            "pace": "Pace", "interests": "Interests",
            "region": "Region", "departure_city": "Departure city",
            "travel_dates": "Travel dates", "budget_amount": "Budget amount",
            "group_size": "Group size", "special_needs": "Special needs",
        }
        header = "User Requirements (Structured)"

    lines = [f"[{header}]"]
    for key, label in label_map.items():
        val = d.get(key)
        if val:
            if isinstance(val, list):
                val = ", ".join(val)
            lines.append(f"- {label}: {val}")

    if ctx.narrative_summary:
        extra_header = "补充信息" if lang == "zh" else "Additional Context"
        lines.append(f"\n[{extra_header}]")
        lines.append(ctx.narrative_summary)

    return "\n".join(lines)


def _format_plan_summary(plan: TravelPlan, lang: str) -> str:
    """将 TravelPlan 压缩为文本摘要，注入 chat 上下文供追问使用。"""
    zh = lang == "zh"
    lines = [f"[{'当前旅行计划' if zh else 'Current Travel Plan'}]"]
    lines.append(f"{'标题' if zh else 'Title'}: {plan.title}")
    lines.append(f"{'目的地' if zh else 'Destination'}: {plan.destination}")
    if plan.start_date:
        lines.append(f"{'日期' if zh else 'Dates'}: {plan.start_date} ~ {plan.end_date}")
    lines.append(f"{'天数' if zh else 'Duration'}: {plan.duration}")
    lines.append("")

    for day in plan.daily_plans:
        header = f"Day {day.day} — {day.theme}"
        activities = "; ".join(
            f"{a.time_slot}: {a.place_name}" for a in day.activities
        )
        lines.append(f"{header}: {activities}")
        if day.hotel:
            h = day.hotel
            hotel_label = "住宿" if zh else "Hotel"
            lines.append(f"  {hotel_label}: {h.name} ({h.area}, {h.price_per_night}, {h.tier})")

    if plan.flight_recommendations:
        lines.append("")
        lines.append("✈️ " + ("航班推荐" if zh else "Flights") + ":")
        for f in plan.flight_recommendations:
            lines.append(f"  - {f.airline} {f.route} {f.price_estimate} ({f.tier})")

    if plan.budget_summary:
        lines.append("")
        lines.append(("💰 预算" if zh else "💰 Budget") + ": " + plan.budget_summary[:200])

    return "\n".join(lines)


def _compress_context(overflow_msgs: list[BaseMessage], ctx: UserContext, lang: str) -> str:
    """增量压缩：将溢出窗口的旧消息中非结构化信息合并到叙事摘要。"""
    human_ai = [
        m for m in overflow_msgs
        if isinstance(m, (HumanMessage, AIMessage)) and m.content
        and not m.content.startswith("[")
    ]
    if not human_ai:
        return ctx.narrative_summary

    structured = json.dumps(_context_to_dict(ctx), ensure_ascii=False)
    prev_summary = ctx.narrative_summary or ("无" if lang == "zh" else "None")
    prompt = prompt_manager.load(
        "summarize", lang=lang,
        structured_fields=structured,
        previous_summary=prev_summary,
    )

    try:
        resp = get_llm().invoke([SystemMessage(content=prompt)] + human_ai)
        text = resp.content.strip()
        if text.lower() in ("无", "none", "n/a", ""):
            return ctx.narrative_summary or ""
        return text
    except Exception:
        return ctx.narrative_summary


# ── _build_plan_messages ──────────────────────────────────────────────────────

def _build_plan_messages(state: State) -> list[BaseMessage]:
    """为 generate_plan 构建精简的消息列表:
    SystemMessage + [策略指令] + [结构化上下文] + [RAG] + 最近 N 条消息
    """
    from datetime import date

    lang = state.get("language", "zh")
    currency = state.get("currency", "CAD")
    today = date.today().isoformat()
    system_prompt = prompt_manager.load("plan_system", lang=lang, currency=currency, today=today)

    # 注入旅行画像策略
    ctx = state.get("user_context")
    if ctx is not None:
        profile = _profile_from_context(ctx)
        strategy = build_strategy_prompt(profile, language=lang)
        system_prompt += "\n\n" + strategy

    all_msgs = state["messages"]
    result: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    # 1. 注入结构化用户上下文
    if ctx is not None:
        ctx_block = _format_context_block(ctx, lang)
        if ctx_block:
            result.append(HumanMessage(content=ctx_block))

    # 2. 已有计划 — 修改时传完整 plan JSON，让 LLM 在原始结构上精确调整
    plan = state.get("plan_data")
    if plan is not None and state.get("plan_generated"):
        plan_json = plan.model_dump_json(indent=2, exclude_none=True)
        modify_hint = (
            "以下是用户当前旅行计划的完整 JSON。用户要求修改此计划。\n"
            "请在此 JSON 基础上调整，输出完整的新 JSON。\n"
            "规则：\n"
            "1. 用户没有要求修改的部分（行程、酒店、活动描述、tips 等）必须原样保留\n"
            "2. 只改动用户明确要求的部分\n"
            "3. 如果用户要求加天数，在已有行程后追加新的 daily_plan\n"
            "4. 如果用户要求删天数，删除对应天并重新编号\n"
            "5. 保持所有字段的结构和格式不变"
            if lang == "zh" else
            "Below is the user's current travel plan as complete JSON. The user wants to modify it.\n"
            "Adjust based on this JSON and output a complete new JSON.\n"
            "Rules:\n"
            "1. Keep ALL parts the user did NOT ask to change (activities, hotels, descriptions, tips, etc.) exactly as-is\n"
            "2. Only modify what the user explicitly requested\n"
            "3. If adding days, append new daily_plans after existing ones\n"
            "4. If removing days, delete them and renumber\n"
            "5. Preserve the structure and format of all fields"
        )
        result.append(HumanMessage(content=f"{modify_hint}\n\n```json\n{plan_json}\n```"))

    # 3. RAG 上下文
    rag_ctx = _extract_rag_context(all_msgs)
    if rag_ctx:
        label = "知识库参考资料" if lang == "zh" else "Knowledge Base Reference"
        result.append(HumanMessage(content=f"[{label}]\n{rag_ctx}"))

    # 3.5 Google Places 真实 POI 数据 — 让 LLM 从真实景点中选择
    destination = ctx.destination if ctx else None
    if destination and not (plan is not None and state.get("plan_generated")):
        # 仅首次生成时拉取，修改计划时不重新拉
        try:
            pois = search_destination_pois(destination, language=lang)
            pois_text = format_pois_for_prompt(pois, language=lang)
            if pois_text:
                result.append(HumanMessage(content=pois_text))
                log.info(f"[build_plan_messages] injected Google Places POIs for {destination}")
            # 存入 state 供 enrich_plan 后验校验
            state["google_pois"] = pois
        except Exception as e:
            log.warning(f"[build_plan_messages] Places POI fetch failed: {e}")

    # 4. 最近 N 条消息
    recent = _trim_messages(all_msgs, PLAN_RECENT_MSGS)
    result.extend(recent)

    return result


# ── Graph nodes ──────────────────────────────────────────────────────────────

def chat(state: State) -> State:
    """和用户对话（带工具调用能力），收集旅行需求。"""
    t0 = time.time()
    lang = state.get("language", "zh")
    currency = state.get("currency", "CAD")
    prompt = prompt_manager.load("chat_system", lang=lang, currency=currency)

    all_msgs = state["messages"]

    # 注入上下文：结构化需求 + 已有计划摘要
    context_msgs: list[BaseMessage] = []
    ctx = state.get("user_context")

    # 有任何已知上下文（survey/对话提取/压缩摘要）时，注入结构化上下文
    if ctx is not None:
        ctx_block = _format_context_block(ctx, lang)
        if ctx_block:
            context_msgs.append(HumanMessage(content=ctx_block))

    # 滑动窗口截断
    if len(all_msgs) > CHAT_WINDOW_SIZE:
        all_msgs = _trim_messages(all_msgs, CHAT_WINDOW_SIZE)

    # 已有计划时注入摘要，让 LLM 能回答追问
    plan = state.get("plan_data")
    if plan is not None and state.get("plan_generated"):
        plan_summary = _format_plan_summary(plan, lang)
        context_msgs.append(HumanMessage(content=plan_summary))

    messages = [SystemMessage(content=prompt)] + context_msgs + all_msgs
    response = get_llm_with_tools().invoke(messages)
    log.info(f"[chat] done in {time.time()-t0:.1f}s")

    return {"messages": [response], "plan_generated": state.get("plan_generated", False)}


# 使用 langgraph 内置的 ToolNode 自动执行工具调用
tool_executor = ToolNode(ALL_TOOLS)


def extract_context(state: State) -> State:
    """从最新消息中增量提取结构化用户上下文。单次 LLM 调用。"""
    t0 = time.time()
    lang = state.get("language", "zh")
    ctx = state.get("user_context") or UserContext()

    # 找到最近的用户消息
    recent_human = [
        m for m in state["messages"]
        if isinstance(m, HumanMessage) and m.content
        and not m.content.startswith("[")
    ]
    if not recent_human:
        return {"user_context": ctx}

    # 构建 prompt
    current = json.dumps(_context_to_dict(ctx), ensure_ascii=False, indent=2)
    prompt = prompt_manager.load("extract_context", lang=lang, current_context=current)

    # 只取最近 4 条消息（用户+AI），足够 LLM 理解最新需求
    recent = [
        m for m in state["messages"]
        if isinstance(m, (HumanMessage, AIMessage)) and m.content
        and not m.content.startswith("[")
    ][-4:]

    print(f"[extract_context] existing ctx: {current}")
    print(f"[extract_context] {len(recent)} recent msgs, last human: "
          f"{[m.content[:80] for m in recent if isinstance(m, HumanMessage)]}")

    parsed: ExtractedContextPatch | None = None
    for attempt in range(2):
        try:
            llm = get_llm().with_structured_output(ExtractedContextPatch, method="function_calling")
            parsed = llm.invoke([SystemMessage(content=prompt)] + recent)
            print(f"[extract_context] structured output (attempt {attempt+1}): "
                  f"{parsed.model_dump_json(exclude_none=True)[:500]}")
            break
        except Exception as e:
            print(f"[extract_context] attempt {attempt+1} FAILED: {type(e).__name__}: {e}")
            if attempt == 1:
                return {"user_context": ctx}

    if parsed is None:
        return {"user_context": ctx}

    # wants_modification 直接设到 ctx 上（schema 强制填充，不会缺）
    if parsed.wants_modification:
        ctx.wants_modification = True
        print("[extract_context] wants_modification=True (LLM)")

    # 转 dict 做合并，排除 confidence 和 wants_modification
    patch = parsed.model_dump(exclude_none=True, exclude={"confidence", "wants_modification"})
    confidence = parsed.confidence

    # 合并 patch 到 ctx：survey 来源的字段不被低 confidence 覆盖
    for key, val in patch.items():
        if key in ("confidence", "source", "narrative_summary"):
            continue
        if val is None or val == "" or val == []:
            continue

        existing_source = ctx.source.get(key, "")
        patch_conf = confidence.get(key, 0.8)

        # survey 来源只有 confidence=1.0（用户明确否定）才覆盖
        if existing_source == "survey" and patch_conf < 1.0:
            continue

        setattr(ctx, key, val)
        ctx.source[key] = "user" if patch_conf >= 1.0 else "inferred"

    print(f"[extract_context] after merge: dest={ctx.destination}, days={ctx.duration_days}, "
          f"interests={ctx.interests}, departure={ctx.departure_city}, budget={ctx.budget_amount}")

    # 上下文压缩：消息超出窗口时，压缩溢出部分到叙事摘要
    all_msgs = state["messages"]
    if len(all_msgs) > CHAT_WINDOW_SIZE:
        overflow = all_msgs[:-CHAT_WINDOW_SIZE]
        ctx.narrative_summary = _compress_context(overflow, ctx, lang)

    log.info(f"[extract_context] done in {time.time()-t0:.1f}s")
    return {"user_context": ctx}


def check_info(state: State) -> State:
    """纯 Python 检查：必填字段是否齐全。不调 LLM，不产生消息。"""
    ctx = state.get("user_context")
    plan_status = state.get("plan_generated", False)

    # 已有计划：检查是否要修改
    if plan_status:
        if ctx and ctx.wants_modification:
            ctx.wants_modification = False  # 重置
            print("[check_info] → generate (modification request via flag)")
            return {"check_result": "generate", "user_context": ctx}

        # 后备检测：即使 extract_context 没设 wants_modification，
        # 也扫最近一条用户消息看是否含修改意图关键词
        if detect_modification_intent(state.get("messages", [])):
            print("[check_info] → generate (keyword fallback)")
            return {"check_result": "generate"}

        print("[check_info] → continue (plan exists, no modification)")
        return {"check_result": "continue"}

    # 尚未生成计划：检查必填项
    if ctx is None:
        print("[check_info] → continue (no user_context)")
        return {"check_result": "continue"}

    has_dest = ctx.destination is not None
    has_days = ctx.duration_days is not None
    has_interests = len(ctx.interests) > 0

    print(f"[check_info] dest={ctx.destination}({has_dest}), "
          f"days={ctx.duration_days}({has_days}), "
          f"interests={ctx.interests}({has_interests})")

    if has_dest and has_days and has_interests:
        print("[check_info] → generate (all fields present)")
        return {"check_result": "generate"}
    elif has_dest and has_days:
        # 有目的地和天数但没兴趣 → 用默认兴趣，直接生成
        ctx.interests = ["food", "culture"]
        print("[check_info] → generate (dest+days, default interests)")
        return {"check_result": "generate", "user_context": ctx}
    else:
        print(f"[check_info] → continue (missing: "
              f"{'dest ' if not has_dest else ''}"
              f"{'days ' if not has_days else ''}"
              f"{'interests' if not has_interests else ''})")
        return {"check_result": "continue"}


def generate_plan(state: State) -> State:
    """生成结构化旅行计划。流式输出：边生成边通过 stream writer 推送部分 JSON。"""
    lang = state.get("language", "zh")

    t0 = time.time()
    messages = _build_plan_messages(state)
    log.info(f"[generate_plan] input tokens ~{sum(len(m.content) for m in messages)} chars, {len(messages)} msgs")

    parser = JsonOutputParser(pydantic_object=TravelPlan)
    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(
            content=messages[0].content + "\n\n" + parser.get_format_instructions()
        )

    llm = get_plan_llm().bind(response_format={"type": "json_object"})
    chain = llm | parser

    try:
        writer = get_stream_writer()
    except Exception:
        writer = None

    def _should_emit(curr: dict, prev: dict | None) -> bool:
        if prev is None:
            return True
        for k in ("title", "destination", "overview"):
            if curr.get(k) and not prev.get(k):
                return True
        if len(curr.get("daily_plans") or []) != len(prev.get("daily_plans") or []):
            return True
        curr_days = curr.get("daily_plans") or []
        prev_days = prev.get("daily_plans") or []
        if curr_days and prev_days:
            curr_last = curr_days[-1] or {}
            prev_last = prev_days[-1] or {}
            if len(curr_last.get("activities") or []) != len(prev_last.get("activities") or []):
                return True
            if (curr_last.get("theme") or "") != (prev_last.get("theme") or ""):
                return True
        return False

    last_partial: dict | None = None
    last_emitted: dict | None = None
    for partial in chain.stream(messages):
        if not isinstance(partial, dict):
            continue
        last_partial = partial
        if writer is not None and _should_emit(partial, last_emitted):
            try:
                writer({"type": "plan_partial", "data": partial})
                last_emitted = partial
            except Exception:
                pass

    if last_partial is None:
        raise ValueError("Plan generation produced no output")

    plan = TravelPlan.model_validate(last_partial)
    log.info(f"[generate_plan] done in {time.time()-t0:.1f}s, {len(plan.daily_plans)} days")

    if lang == "en":
        summary = f'I\'ve generated "{plan.title}"! Fetching photos and details, please wait...'
    else:
        summary = f"我已经为你生成了「{plan.title}」！正在获取景点图片和详细介绍，稍等片刻..."
    return {
        "messages": [AIMessage(content=summary)],
        "plan_generated": True,
        "plan_data": plan,
    }


def _validate_plan_places(
    plan: TravelPlan,
    google_pois: dict[str, list] | None,
    lang: str,
) -> TravelPlan:
    """Post-validate: check each activity's place_name exists via Google Places.

    If a place was NOT in the pre-fetched POI list, do a quick search to verify
    it actually exists. If it doesn't, replace with the best-rated unused POI
    from the list.
    """
    if not google_pois:
        return plan

    from tools.places import search_places, PlacePOI

    # Build set of known-good names from Google Places results
    known_names: set[str] = set()
    all_pois: list[PlacePOI] = []
    for category in ("attractions", "restaurants"):
        for poi in google_pois.get(category, []):
            known_names.add(poi.display_name.lower())
            all_pois.append(poi)

    # Track names already used in plan
    used_names = {
        act.place_name.lower()
        for day in plan.daily_plans
        for act in day.activities
    }

    replacements_made = 0
    for day in plan.daily_plans:
        for act in day.activities:
            name_lower = act.place_name.lower()
            if name_lower in known_names:
                continue  # Already a verified Google Places result

            # Not in our pre-fetched list — verify it exists via quick search
            try:
                results = search_places(act.place_name, max_results=1, language="en")
                if results and results[0].display_name.lower() != name_lower:
                    # Search returned a different name — use the corrected name
                    log.info(
                        f"[place_validate] correcting '{act.place_name}' → "
                        f"'{results[0].display_name}'"
                    )
                    act.place_name = results[0].display_name
                    replacements_made += 1
                elif results:
                    # Found with same name — it's real, keep it
                    pass
                else:
                    # Not found at all — replace with best unused POI
                    is_rest = getattr(act, "is_restaurant", False)
                    category = "restaurants" if is_rest else "attractions"
                    replacement = _pick_unused_poi(
                        google_pois.get(category, []), used_names
                    )
                    if replacement:
                        log.warning(
                            f"[place_validate] '{act.place_name}' not found, "
                            f"replacing with '{replacement.display_name}'"
                        )
                        act.place_name = replacement.display_name
                        if replacement.editorial_summary:
                            act.description = replacement.editorial_summary
                        used_names.add(replacement.display_name.lower())
                        replacements_made += 1
            except Exception as e:
                log.warning(f"[place_validate] verification failed for '{act.place_name}': {e}")

            used_names.add(act.place_name.lower())

    log.info(f"[place_validate] {replacements_made} corrections made")
    return plan


def _pick_unused_poi(pois: list, used_names: set[str]):
    """Pick the highest-rated POI not already used in the plan."""
    for poi in sorted(pois, key=lambda p: (-p.rating, -p.user_ratings_total)):
        if poi.display_name.lower() not in used_names:
            return poi
    return None


MAX_DISTANCE_KM = 80  # 景点距目的地中心最大允许距离（km）


def _validate_place_distances(
    plan: TravelPlan,
    places: dict[str, PlaceInfo],
    lang: str,
) -> TravelPlan:
    """检查每个活动景点与目的地的距离，过远的用 LLM 替换为合理景点。"""
    dest_info = places.get(plan.destination)
    if not dest_info or not dest_info.lat or not dest_info.lon:
        log.warning("[distance_check] no coordinates for destination, skipping")
        return plan

    dest_ll = (dest_info.lat, dest_info.lon)
    outliers: list[tuple[int, int, str, float]] = []  # (day_idx, act_idx, name, dist_km)

    for di, day in enumerate(plan.daily_plans):
        for ai, act in enumerate(day.activities):
            pi = places.get(act.place_name)
            if not pi or not pi.lat or not pi.lon:
                continue
            dist_km = _haversine_meters(dest_ll, (pi.lat, pi.lon)) / 1000
            if dist_km > MAX_DISTANCE_KM:
                outliers.append((di, ai, act.place_name, dist_km))
                log.warning(
                    f"[distance_check] OUTLIER: {act.place_name} is {dist_km:.0f}km from {plan.destination}"
                )

    if not outliers:
        log.info("[distance_check] all places within range")
        return plan

    # 用 LLM 批量获取替代景点
    outlier_names = [o[2] for o in outliers]
    existing = {act.place_name for day in plan.daily_plans for act in day.activities}
    existing_str = "、".join(existing)

    if lang == "zh":
        prompt = (
            f"以下景点不在{plan.destination}附近，请为每个提供一个位于{plan.destination}市区内的替代景点。\n"
            f"需要替换的：{', '.join(outlier_names)}\n"
            f"已有的（不要重复）：{existing_str}\n"
            f"只输出 JSON 对象，key 是原景点名，value 是一个对象 "
            f'{{\"name\": \"替代景点标准名称\", \"description\": \"2-3句描述\", \"duration_minutes\": 数字}}。\n'
            f"不要输出其他内容。"
        )
    else:
        prompt = (
            f"The following places are NOT near {plan.destination}. Provide a replacement within {plan.destination} city for each.\n"
            f"To replace: {', '.join(outlier_names)}\n"
            f"Already in plan (do NOT repeat): {existing_str}\n"
            f"Output ONLY a JSON object where each key is the original name and value is "
            f'{{\"name\": \"replacement standard name\", \"description\": \"2-3 sentence description\", \"duration_minutes\": number}}.\n'
            f"No other text."
        )

    try:
        llm = get_llm()
        resp = llm.invoke([SystemMessage(content=prompt)])
        text = resp.content.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            replacements = json.loads(text[start : end + 1])
        else:
            replacements = {}
    except Exception as e:
        log.warning(f"[distance_check] LLM replacement failed: {e}")
        replacements = {}

    replaced = 0
    for di, ai, old_name, dist_km in outliers:
        repl = replacements.get(old_name)
        if not repl:
            continue
        act = plan.daily_plans[di].activities[ai]
        log.info(
            f"[distance_check] replacing {old_name} ({dist_km:.0f}km away) → {repl['name']}"
        )
        act.place_name = repl["name"]
        act.description = repl.get("description", act.description)
        act.duration_minutes = repl.get("duration_minutes", act.duration_minutes)
        replaced += 1

    log.info(f"[distance_check] replaced {replaced}/{len(outliers)} outliers")
    return plan


def enrich_plan(state: State) -> State:
    """从 Wikipedia 获取景点信息，渲染并保存 HTML。使用并发加速。"""
    import concurrent.futures

    t0 = time.time()
    lang = state.get("language", "zh")
    plan: TravelPlan = state["plan_data"]

    try:
        writer = get_stream_writer()
    except Exception:
        writer = None

    def _progress(pct: int, status: str) -> None:
        if writer:
            try:
                writer({"type": "enrich_progress", "pct": pct, "status": status})
            except Exception:
                pass

    place_names = []
    seen = set()
    if plan.destination not in seen:
        place_names.append(plan.destination)
        seen.add(plan.destination)
    for day in plan.daily_plans:
        for act in day.activities:
            if act.place_name not in seen:
                place_names.append(act.place_name)
                seen.add(act.place_name)
    log.info(f"[enrich_plan] start — {len(place_names)} places to enrich")

    _progress(72, "搜索景点信息..." if lang == "zh" else "Fetching place info...")

    currency = state.get("currency", "CAD").lower()
    origin = getattr(plan, "departure_iata", "") or ""
    dest = getattr(plan, "destination_iata", "") or ""
    start_date = getattr(plan, "start_date", "") or ""
    end_date = getattr(plan, "end_date", "") or ""

    # ── 并发执行所有 enrichment 任务 ──
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        # 1. Wikipedia 景点信息 — 每个景点一个任务
        place_futures = {
            name: pool.submit(fetch_place_info, name, language=lang)
            for name in place_names
        }
        # 2. 视频搜索（整批）
        links_future = pool.submit(
            search_videos_batch, place_names,
            destination=plan.destination, language=lang,
        )
        # 3. 嵌入搜索（整批）
        embeds_future = pool.submit(
            search_embeds_batch, place_names,
            destination=plan.destination, language=lang,
        )
        # 4. 机票
        deals_future = pool.submit(
            fetch_travel_deals, origin, dest,
            start_date=start_date, end_date=end_date,
            currency=currency, language=lang,
        )

        # 收集结果 — 谁先完成谁先收，同时推送进度
        log.info("[enrich_plan] waiting for wikipedia results...")
        t1 = time.time()
        places: dict[str, PlaceInfo] = {}
        for name, fut in place_futures.items():
            try:
                places[name] = fut.result(timeout=15)
                log.info(f"[enrich_plan]   wiki OK: {name} ({time.time()-t1:.1f}s)")
            except Exception as e:
                places[name] = PlaceInfo(name=name, summary="", image_url="", lat=0, lon=0)
                log.warning(f"[enrich_plan]   wiki FAIL: {name} — {type(e).__name__}: {e}")
        log.info(f"[enrich_plan] wikipedia done in {time.time()-t1:.1f}s")

        # ── 景点真实性校验：检查 LLM 生成的景点是否真实存在 ──
        _progress(74, "校验景点信息..." if lang == "zh" else "Validating places...")
        google_pois = state.get("google_pois")
        plan = _validate_plan_places(plan, google_pois, lang)

        # ── 距离校验：检查是否有景点离目的地过远 ──
        _progress(75, "校验景点距离..." if lang == "zh" else "Validating place distances...")
        old_names = {act.place_name for day in plan.daily_plans for act in day.activities}
        plan = _validate_place_distances(plan, places, lang)
        new_names = {act.place_name for day in plan.daily_plans for act in day.activities}
        added_names = new_names - old_names
        if added_names:
            log.info(f"[enrich_plan] fetching info for {len(added_names)} replacement places")
            for name in added_names:
                try:
                    places[name] = fetch_place_info(name, language=lang)
                    place_names.append(name)
                except Exception as e:
                    places[name] = PlaceInfo(name=name, summary="", image_url=None)
                    log.warning(f"[enrich_plan]   replacement wiki FAIL: {name} — {e}")

        _progress(78, "搜索推荐视频..." if lang == "zh" else "Searching videos...")

        log.info("[enrich_plan] waiting for videos...")
        t2 = time.time()
        try:
            links = links_future.result(timeout=30)
            log.info(f"[enrich_plan] videos done in {time.time()-t2:.1f}s — {len(links)} results")
        except Exception as e:
            links = {}
            log.warning(f"[enrich_plan] videos FAIL in {time.time()-t2:.1f}s — {type(e).__name__}: {e}")
        _progress(83, "搜索短视频攻略..." if lang == "zh" else "Searching short videos...")

        log.info("[enrich_plan] waiting for embeds...")
        t3 = time.time()
        try:
            embeds = embeds_future.result(timeout=30)
            log.info(f"[enrich_plan] embeds done in {time.time()-t3:.1f}s — {len(embeds)} results")
        except Exception as e:
            embeds = {}
            log.warning(f"[enrich_plan] embeds FAIL in {time.time()-t3:.1f}s — {type(e).__name__}: {e}")
        _progress(88, "搜索机票数据..." if lang == "zh" else "Searching flights...")

        log.info("[enrich_plan] waiting for flights...")
        t4 = time.time()
        try:
            travel_deals = deals_future.result(timeout=20)
            log.info(f"[enrich_plan] flights done in {time.time()-t4:.1f}s")
        except Exception as e:
            travel_deals = None
            log.warning(f"[enrich_plan] flights FAIL in {time.time()-t4:.1f}s — {type(e).__name__}: {e}")
        _progress(90, "计算景点间通行时间..." if lang == "zh" else "Calculating transit times...")

        log.info("[enrich_plan] calculating transit times...")
        t_transit = time.time()
        day_transits: dict[int, list[TransitInfo | None]] = {}
        for day in plan.daily_plans:
            all_acts = day.activities
            if len(all_acts) >= 2:
                day_transits[day.day] = get_day_transits(all_acts, places)
        log.info(f"[enrich_plan] transit times done in {time.time()-t_transit:.1f}s")

        _progress(94, "渲染计划页面..." if lang == "zh" else "Rendering plan...")

    t5 = time.time()
    html = render_plan(plan, places, links, embeds, travel_deals=travel_deals,
                       day_transits=day_transits, language=lang)
    path = Path("travel_plan.html").resolve()
    path.write_text(html, encoding="utf-8")
    log.info(f"[enrich_plan] render done in {time.time()-t5:.1f}s")
    log.info(f"[enrich_plan] TOTAL {time.time()-t0:.1f}s")

    if lang == "en":
        done_msg = f'Your travel plan "{plan.title}" is ready! Check it out on the right.'
    else:
        done_msg = f"旅行计划「{plan.title}」已生成！请在右侧查看完整计划。"
    return {
        "messages": [AIMessage(content=done_msg)],
        "plan_generated": True,
        "html_path": str(path),
        "enrich_places": places,
        "enrich_links": links,
        "enrich_embeds": embeds,
        "enrich_deals": travel_deals,
        "enrich_transits": day_transits,
    }


# ── Routing ──────────────────────────────────────────────────────────────────

from langgraph.graph import START


def route_entry_point(state: State) -> Literal["initial_intake", "chat", "generate_plan"]:
    """首条消息（无上下文、无计划）→ initial_intake；
    问卷提交后（check_result=generate 且尚无计划）→ 直接生成计划；
    后续 → chat。"""
    if not state.get("plan_generated") and state.get("user_context") is None:
        return "initial_intake"
    if not state.get("plan_generated") and state.get("check_result") == "generate":
        return "generate_plan"
    return "chat"


def route_after_intake(state: State) -> Literal["generate_plan", "__end__"]:
    """intake 后：proceed_to_plan → 生成计划；show_survey → 结束等问卷。"""
    if state.get("intake_result") == "proceed_to_plan":
        return "generate_plan"
    return END


def route_after_chat(state: State) -> Literal["tool_executor", "extract_context"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_executor"
    return "extract_context"


def route_after_check(state: State) -> Literal["generate_plan", "__end__"]:
    if state.get("check_result") == "generate":
        return "generate_plan"
    return END


# ── Build graph ──────────────────────────────────────────────────────────────
#
# 流程:
#   首条消息: → initial_intake → generate_plan / END(问卷)
#   后续消息: → chat → extract_context → check_info → generate_plan → enrich_plan
#                ↕
#           tool_executor

def build_graph():
    graph = StateGraph(State)

    graph.add_node("initial_intake", initial_intake)
    graph.add_node("chat", chat)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("extract_context", extract_context)
    graph.add_node("check_info", check_info)
    graph.add_node("generate_plan", generate_plan)
    graph.add_node("enrich_plan", enrich_plan)

    # 入口：条件路由
    graph.add_conditional_edges(START, route_entry_point)

    # initial_intake 路由
    graph.add_conditional_edges("initial_intake", route_after_intake)

    # 常规 chat 流程
    graph.add_conditional_edges("chat", route_after_chat)
    graph.add_edge("tool_executor", "chat")
    graph.add_edge("extract_context", "check_info")
    graph.add_conditional_edges("check_info", route_after_check)
    graph.add_edge("generate_plan", "enrich_plan")
    graph.add_edge("enrich_plan", END)

    return graph.compile()
