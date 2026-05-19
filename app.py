"""
AI Travel Planner — FastAPI Web 服务
用法: uvicorn app:app --reload
"""

from __future__ import annotations

import json
import threading
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from graph import CHAT_WINDOW_SIZE, build_graph
from models import UserContext
from rag.loader import load_seed_data
from rag.store import close_client as _close_chroma
from renderer import render_flight_cards, render_plan
from tools.travel_api import fetch_flight_deals

load_dotenv()

app = FastAPI(title="AI Travel Planner")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 内存中保存会话状态 {session_id: state}
_sessions: dict[str, dict] = {}

# 启动时加载知识库
_seed_count = 0
_graph = None

# 关闭信号：让 SSE generator 尽快退出
_shutdown_event = threading.Event()


@app.on_event("startup")
def startup():
    global _seed_count, _graph
    _shutdown_event.clear()
    _seed_count = load_seed_data()
    _graph = build_graph()


@app.on_event("shutdown")
def shutdown():
    _shutdown_event.set()
    _close_chroma()


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    language: str = "zh"
    currency: str = "CAD"


# 节点名 → 进度百分比 + 双语状态文案
PROGRESS_STEPS = {
    "chat": {
        "pct": 10,
        "zh": "正在理解你的需求...",
        "en": "Understanding your request...",
    },
    "tool_executor": {
        "pct": 20,
        "zh": "正在查询相关信息...",
        "en": "Looking up information...",
    },
    "extract_context": {
        "pct": 30,
        "zh": "正在理解你的旅行偏好...",
        "en": "Analyzing your preferences...",
    },
    "generate_plan": {
        "pct": 50,
        "zh": "正在生成旅行计划...",
        "en": "Generating travel plan...",
    },
    "enrich_plan": {
        "pct": 75,
        "zh": "正在获取图片、视频和地图数据...",
        "en": "Fetching photos, videos & map data...",
    },
}


def _is_user_facing(msg) -> bool:
    """判断消息是否应该展示给用户。"""
    if isinstance(msg, ToolMessage):
        return False
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        return False
    if isinstance(msg, AIMessage) and not msg.content:
        return False
    return True


def _sse(event: str, data: dict) -> str:
    """格式化 SSE 事件。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/api/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())[:8]

    if session_id not in _sessions:
        _sessions[session_id] = {
            "messages": [],
            "plan_generated": False,
            "plan_data": None,
            "html_path": "",
            "language": req.language,
            "currency": req.currency,
            "user_context": None,
            "check_result": "continue",
            "intake_result": None,
            "enrich_places": None,
            "enrich_links": None,
            "enrich_embeds": None,
            "enrich_deals": None,
            "enrich_transits": None,
        }

    state = _sessions[session_id]
    state["language"] = req.language
    state["currency"] = req.currency
    state["messages"].append(HumanMessage(content=req.message))

    lang = req.language

    def generate_sse():
        import queue

        prev_msg_count = len(state["messages"])
        plan_regenerating = False

        q: queue.Queue = queue.Queue()

        def _run_graph():
            try:
                for mode, chunk in _graph.stream(state, stream_mode=["values", "custom"]):
                    if _shutdown_event.is_set():
                        return
                    q.put(("chunk", mode, chunk))
                q.put(("done", None, None))
            except Exception as e:
                q.put(("error", None, e))

        t = threading.Thread(target=_run_graph, daemon=True)
        t.start()

        try:
            final_result = None
            deadline = 120  # 总超时秒数
            while deadline > 0:
                if _shutdown_event.is_set():
                    return
                try:
                    msg_type, mode, chunk = q.get(timeout=2)
                except queue.Empty:
                    deadline -= 2
                    continue
                if msg_type == "done":
                    break
                if msg_type == "error":
                    raise chunk  # chunk is the exception
                if mode == "custom":
                    if isinstance(chunk, dict):
                        ctype = chunk.get("type", "")
                        if ctype == "plan_partial":
                            yield _sse("plan_partial", {"plan": chunk.get("data", {})})
                        elif ctype == "enrich_progress":
                            yield _sse("progress", {
                                "pct": chunk.get("pct", 75),
                                "status": chunk.get("status", ""),
                            })
                        elif ctype == "show_survey":
                            yield _sse("show_survey", {
                                "missing_fields": chunk.get("missing_fields", []),
                                "extracted": chunk.get("extracted", {}),
                            })
                    continue

                final_result = chunk

                # check_info 决定生成计划时才标记
                if chunk.get("check_result") == "generate":
                    plan_regenerating = True

                msgs = chunk.get("messages", [])
                if len(msgs) > prev_msg_count:
                    last_msg = msgs[-1]
                    if plan_regenerating and chunk.get("html_path") and chunk.get("plan_data") is not None:
                        yield _sse("progress", {
                            "pct": 95,
                            "status": PROGRESS_STEPS["enrich_plan"][lang],
                        })
                    elif plan_regenerating and chunk.get("plan_data") is not None:
                        yield _sse("progress", {
                            "pct": 70,
                            "status": PROGRESS_STEPS["generate_plan"][lang],
                        })
                    elif isinstance(last_msg, ToolMessage):
                        yield _sse("progress", {
                            "pct": 20,
                            "status": PROGRESS_STEPS["tool_executor"][lang],
                        })
                    elif isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
                        yield _sse("progress", {
                            "pct": 15,
                            "status": PROGRESS_STEPS["chat"][lang],
                        })

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = (
                f"生成过程出错：{type(e).__name__}: {e}"
                if lang == "zh"
                else f"Error during generation: {type(e).__name__}: {e}"
            )
            yield _sse("done", {
                "session_id": session_id,
                "reply": error_msg,
                "plan_html": None,
            })
            return

        if final_result is None:
            yield _sse("done", {
                "session_id": session_id,
                "reply": "正在思考中..." if lang == "zh" else "Thinking...",
                "plan_html": None,
            })
            return

        # 提取面向用户的消息
        all_msgs = final_result.get("messages", [])
        new_messages = all_msgs[prev_msg_count:]
        user_facing = [m for m in new_messages if _is_user_facing(m)]
        reply = "\n\n".join(m.content for m in user_facing if m.content)
        if not reply:
            reply = "正在思考中..." if lang == "zh" else "Thinking..."

        # 读取 HTML — 重新生成时从新路径读，否则尝试已缓存路径
        plan_html = None
        html_path = final_result.get("html_path", "") or state.get("html_path", "")
        if plan_regenerating and html_path and final_result.get("plan_data") is not None:
            try:
                with open(html_path, encoding="utf-8") as f:
                    plan_html = f.read()
            except FileNotFoundError:
                pass
        elif not plan_regenerating and html_path and state.get("plan_generated"):
            # 非重新生成但已有计划 — 把已有 HTML 也发回，防止前端空白
            try:
                with open(html_path, encoding="utf-8") as f:
                    plan_html = f.read()
            except FileNotFoundError:
                pass

        # 裁剪消息：压缩后的旧消息已存入 narrative_summary，可安全丢弃
        if len(all_msgs) > CHAT_WINDOW_SIZE:
            all_msgs = all_msgs[-CHAT_WINDOW_SIZE:]
            while all_msgs and isinstance(all_msgs[0], ToolMessage):
                all_msgs = all_msgs[1:]

        # 保存 state
        _sessions[session_id] = {
            "messages": all_msgs,
            "plan_generated": final_result.get("plan_generated", False),
            "plan_data": final_result.get("plan_data"),
            "html_path": final_result.get("html_path", ""),
            "language": req.language,
            "currency": req.currency,
            "user_context": final_result.get("user_context"),
            "check_result": "continue",
            "intake_result": final_result.get("intake_result"),
            "enrich_places": final_result.get("enrich_places"),
            "enrich_links": final_result.get("enrich_links"),
            "enrich_embeds": final_result.get("enrich_embeds"),
            "enrich_deals": final_result.get("enrich_deals"),
            "enrich_transits": final_result.get("enrich_transits"),
        }

        if plan_regenerating:
            yield _sse("progress", {"pct": 100, "status": "Done!" if lang == "en" else "完成！"})
        yield _sse("done", {
            "session_id": session_id,
            "reply": reply,
            "plan_html": plan_html,
        })

    return StreamingResponse(generate_sse(), media_type="text/event-stream")


class RerenderRequest(BaseModel):
    session_id: str
    language: str = "zh"


@app.post("/api/rerender")
def rerender(req: RerenderRequest):
    """切换语言后，用缓存的富化数据重新渲染计划 HTML。"""
    state = _sessions.get(req.session_id)
    if not state or not state.get("plan_data"):
        return JSONResponse({"plan_html": None}, status_code=404)

    plan = state["plan_data"]
    places = state.get("enrich_places") or {}
    links = state.get("enrich_links")
    embeds = state.get("enrich_embeds")
    deals = state.get("enrich_deals")

    transits = state.get("enrich_transits")

    html = render_plan(plan, places, links, embeds, travel_deals=deals,
                       day_transits=transits, language=req.language)

    state["language"] = req.language
    from pathlib import Path
    path = Path("travel_plan.html").resolve()
    path.write_text(html, encoding="utf-8")
    state["html_path"] = str(path)

    return {"plan_html": html}


class FlightSearchRequest(BaseModel):
    session_id: str
    airline: str = "all"      # IATA code or "all"
    max_stops: int = -1       # -1 = any, 0 = direct, 1 = ≤1 stop
    max_price: int = 0        # 0 = no limit
    refresh: bool = False     # True = 强制重新调 API


@app.post("/api/flights")
def search_flights(req: FlightSearchRequest):
    """筛选机票：优先用缓存数据，refresh=True 时重新调 API。"""
    state = _sessions.get(req.session_id)
    if not state or not state.get("plan_data"):
        return JSONResponse({"cards_html": None, "count": 0}, status_code=404)

    plan = state["plan_data"]
    lang = state.get("language", "zh")
    currency = state.get("currency", "CAD").lower()

    # 优先用 enrich_plan 阶段缓存的全量数据
    deals = state.get("enrich_deals")
    flights = deals.flights if deals and deals.flights else []

    # refresh 或无缓存时重新拉 API
    if req.refresh or not flights:
        origin = getattr(plan, "departure_iata", "") or ""
        dest = getattr(plan, "destination_iata", "") or ""
        start_date = getattr(plan, "start_date", "") or ""
        end_date = getattr(plan, "end_date", "") or ""
        if not origin or not dest:
            return {"cards_html": None, "count": 0}
        flights = fetch_flight_deals(
            origin, dest,
            departure_date=start_date, return_date=end_date,
            currency=currency, limit=50,
        )
        # 更新缓存
        from tools.travel_api import TravelDeals
        state["enrich_deals"] = TravelDeals(
            flights=flights,
            hotels=(deals.hotels if deals else []),
        )

    # 应用筛选条件
    filtered = []
    for f in flights:
        if req.airline != "all" and f.airline != req.airline:
            continue
        if req.max_stops >= 0 and f.transfers > req.max_stops:
            continue
        if req.max_price > 0 and f.price > req.max_price:
            continue
        filtered.append(f)

    filtered.sort(key=lambda f: f.price)

    # 航司列表（用未筛选的全集）
    airlines_map: dict[str, str] = {}
    for f in flights:
        if f.airline_name and f.airline not in airlines_map:
            airlines_map[f.airline] = f.airline_name
    airlines_list = sorted(airlines_map.items(), key=lambda x: x[1])

    if not filtered:
        return {"cards_html": None, "count": 0, "airlines": airlines_list}

    cards_html = render_flight_cards(filtered, lang)
    return {
        "cards_html": cards_html,
        "count": len(filtered),
        "airlines": airlines_list,
    }


class EmbedRefreshRequest(BaseModel):
    session_id: str
    place_name: str


@app.post("/api/embeds")
def refresh_embeds(req: EmbedRefreshRequest):
    """为某个景点重新搜索 TikTok/Instagram/小红书 视频。"""
    from tools.embeds import search_embeds

    state = _sessions.get(req.session_id)
    if not state or not state.get("plan_data"):
        return JSONResponse({"embeds": []}, status_code=404)

    plan = state["plan_data"]
    lang = state.get("language", "zh")
    destination = plan.destination or ""

    result = search_embeds(req.place_name, destination, lang)
    embeds_list = [
        {
            "url": e.url,
            "title": e.title or "",
            "author": e.author or "",
            "thumbnail_url": e.thumbnail_url or "",
            "platform": e.platform,
        }
        for e in result.all
    ]
    return {"embeds": embeds_list}


class SuggestRequest(BaseModel):
    session_id: str
    category: str = "activities"  # "activities" or "restaurants"
    exclude: list[str] = []  # names already suggested, to avoid repeats


@app.post("/api/suggest")
def suggest_more(req: SuggestRequest):
    """用 LLM 生成更多景点或餐厅推荐，返回名称列表。"""
    from langchain_core.messages import SystemMessage as SysMsg

    from graph import get_llm

    state = _sessions.get(req.session_id)
    if not state or not state.get("plan_data"):
        return JSONResponse({"suggestions": []}, status_code=404)

    plan = state["plan_data"]
    lang = state.get("language", "zh")
    destination = plan.destination or ""

    # 收集计划中已有 + 前端已推荐过的名称，避免重复
    existing = set(req.exclude)
    for day in plan.daily_plans:
        for act in day.activities:
            existing.add(act.place_name)

    existing_str = "、".join(existing) if existing else "无"

    if req.category == "restaurants":
        if lang == "zh":
            prompt = (
                f"请为去{destination}旅行的游客推荐 3 家当地特色餐厅。\n"
                f"⚠️ 餐厅必须位于{destination}市区内，绝对不要推荐其他城市的餐厅。\n"
                f"已有（不要重复）：{existing_str}\n"
                f"只输出 JSON 数组，每个元素是一个对象 {{\"name\": \"餐厅真实名称\", \"desc\": \"一句话描述特色\"}}。\n"
                f"不要输出其他内容。"
            )
        else:
            prompt = (
                f"Recommend 3 local restaurants in {destination} for tourists.\n"
                f"⚠️ All restaurants MUST be located within {destination} city. Do NOT suggest restaurants in other cities.\n"
                f"Already included (do NOT repeat): {existing_str}\n"
                f"Output ONLY a JSON array, each element: {{\"name\": \"real restaurant name\", \"desc\": \"one-line description\"}}.\n"
                f"No other text."
            )
    else:
        if lang == "zh":
            prompt = (
                f"请为去{destination}旅行的游客推荐 3 个值得去的景点或活动。\n"
                f"⚠️ 景点必须位于{destination}市区或可当天往返的近郊，绝对不要推荐其他城市的景点。\n"
                f"已有（不要重复）：{existing_str}\n"
                f"只输出 JSON 数组，每个元素是一个对象 {{\"name\": \"景点标准名称\", \"desc\": \"一句话描述亮点\"}}。\n"
                f"不要输出其他内容。"
            )
        else:
            prompt = (
                f"Recommend 3 attractions or activities in {destination} for tourists.\n"
                f"⚠️ All places MUST be within {destination} city or nearby day-trip suburbs. Do NOT suggest places in other cities.\n"
                f"Already included (do NOT repeat): {existing_str}\n"
                f"Output ONLY a JSON array, each element: {{\"name\": \"standard place name\", \"desc\": \"one-line description\"}}.\n"
                f"No other text."
            )

    try:
        llm = get_llm()
        resp = llm.invoke([SysMsg(content=prompt)])
        text = resp.content.strip()
        # 提取 JSON 数组
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            suggestions = json.loads(text[start : end + 1])
        else:
            suggestions = []
    except Exception as e:
        print(f"[suggest] error: {e}")
        suggestions = []

    return {"suggestions": suggestions}


class SurveySubmitRequest(BaseModel):
    session_id: str
    language: str = "zh"
    currency: str = "CAD"
    # 必填字段（问卷可能需要用户填写）
    destination: str = ""
    duration_days: int | None = None
    # 画像字段
    travel_style: str = ""
    budget_tier: str = ""
    pace: str = ""
    interests: list[str] = []


@app.post("/api/survey")
def submit_survey(req: SurveySubmitRequest):
    """问卷提交：合并用户选择到 user_context，然后走 generate_plan 流程。"""
    state = _sessions.get(req.session_id)
    if not state:
        return JSONResponse({"error": "session not found"}, status_code=404)

    lang = req.language
    state["language"] = lang
    state["currency"] = req.currency

    # 合并问卷数据到 user_context
    ctx = state.get("user_context") or UserContext()
    if req.destination:
        ctx.destination = req.destination
        ctx.source["destination"] = "survey"
    if req.duration_days:
        ctx.duration_days = req.duration_days
        ctx.source["duration_days"] = "survey"
    if req.travel_style:
        ctx.travel_style = req.travel_style
        ctx.source["travel_style"] = "survey"
    if req.budget_tier:
        ctx.budget_tier = req.budget_tier
        ctx.source["budget_tier"] = "survey"
    if req.pace:
        ctx.pace = req.pace
        ctx.source["pace"] = "survey"
    if req.interests:
        ctx.interests = req.interests
        ctx.source["interests"] = "survey"
    else:
        ctx.interests = ctx.interests or ["food", "culture"]

    # 填充默认值
    ctx.travel_style = ctx.travel_style or "friends"
    ctx.budget_tier = ctx.budget_tier or "comfort"
    ctx.pace = ctx.pace or "mixed"

    state["user_context"] = ctx
    # 标记为直接生成计划
    state["intake_result"] = "proceed_to_plan"
    state["check_result"] = "generate"

    def generate_sse():
        import queue

        prev_msg_count = len(state["messages"])

        q: queue.Queue = queue.Queue()

        def _run_graph():
            try:
                for mode, chunk in _graph.stream(state, stream_mode=["values", "custom"]):
                    if _shutdown_event.is_set():
                        return
                    q.put(("chunk", mode, chunk))
                q.put(("done", None, None))
            except Exception as e:
                q.put(("error", None, e))

        t = threading.Thread(target=_run_graph, daemon=True)
        t.start()

        try:
            final_result = None
            deadline = 120
            while deadline > 0:
                if _shutdown_event.is_set():
                    return
                try:
                    msg_type, mode, chunk = q.get(timeout=2)
                except queue.Empty:
                    deadline -= 2
                    continue
                if msg_type == "done":
                    break
                if msg_type == "error":
                    raise chunk
                if mode == "custom":
                    if isinstance(chunk, dict):
                        ctype = chunk.get("type", "")
                        if ctype == "plan_partial":
                            yield _sse("plan_partial", {"plan": chunk.get("data", {})})
                        elif ctype == "enrich_progress":
                            yield _sse("progress", {
                                "pct": chunk.get("pct", 75),
                                "status": chunk.get("status", ""),
                            })
                    continue

                final_result = chunk

                msgs = chunk.get("messages", [])
                if len(msgs) > prev_msg_count:
                    last_msg = msgs[-1]
                    if chunk.get("html_path") and chunk.get("plan_data") is not None:
                        yield _sse("progress", {
                            "pct": 95,
                            "status": PROGRESS_STEPS["enrich_plan"][lang],
                        })
                    elif chunk.get("plan_data") is not None:
                        yield _sse("progress", {
                            "pct": 70,
                            "status": PROGRESS_STEPS["generate_plan"][lang],
                        })

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = (
                f"生成过程出错：{type(e).__name__}: {e}"
                if lang == "zh"
                else f"Error during generation: {type(e).__name__}: {e}"
            )
            yield _sse("done", {
                "session_id": req.session_id,
                "reply": error_msg,
                "plan_html": None,
            })
            return

        if final_result is None:
            yield _sse("done", {
                "session_id": req.session_id,
                "reply": "正在思考中..." if lang == "zh" else "Thinking...",
                "plan_html": None,
            })
            return

        all_msgs = final_result.get("messages", [])
        new_messages = all_msgs[prev_msg_count:]
        user_facing = [m for m in new_messages if _is_user_facing(m)]
        reply = "\n\n".join(m.content for m in user_facing if m.content)
        if not reply:
            reply = "正在思考中..." if lang == "zh" else "Thinking..."

        plan_html = None
        html_path = final_result.get("html_path", "") or state.get("html_path", "")
        if html_path and final_result.get("plan_data") is not None:
            try:
                with open(html_path, encoding="utf-8") as f:
                    plan_html = f.read()
            except FileNotFoundError:
                pass

        if len(all_msgs) > CHAT_WINDOW_SIZE:
            all_msgs = all_msgs[-CHAT_WINDOW_SIZE:]
            while all_msgs and isinstance(all_msgs[0], ToolMessage):
                all_msgs = all_msgs[1:]

        _sessions[req.session_id] = {
            "messages": all_msgs,
            "plan_generated": final_result.get("plan_generated", False),
            "plan_data": final_result.get("plan_data"),
            "html_path": final_result.get("html_path", ""),
            "language": req.language,
            "currency": req.currency,
            "user_context": final_result.get("user_context"),
            "check_result": "continue",
            "intake_result": final_result.get("intake_result"),
            "enrich_places": final_result.get("enrich_places"),
            "enrich_links": final_result.get("enrich_links"),
            "enrich_embeds": final_result.get("enrich_embeds"),
            "enrich_deals": final_result.get("enrich_deals"),
            "enrich_transits": final_result.get("enrich_transits"),
        }

        yield _sse("progress", {"pct": 100, "status": "Done!" if lang == "en" else "完成！"})
        yield _sse("done", {
            "session_id": req.session_id,
            "reply": reply,
            "plan_html": plan_html,
        })

    return StreamingResponse(generate_sse(), media_type="text/event-stream")


@app.get("/api/health")
def health():
    return {"status": "ok", "knowledge_count": _seed_count}
