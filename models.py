"""旅行计划的 Pydantic 数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── 用户上下文（结构化提取，增量更新） ─────────────────────────────────────


class UserContext(BaseModel):
    """从对话中结构化提取的用户旅行需求，每轮增量更新。"""

    # 必填 — 触发计划生成的门槛
    destination: str | None = None
    duration_days: int | None = None

    # 画像 — 合并了原 classify_profile + region_detect
    travel_style: str | None = None      # solo / couple / family / friends
    budget_tier: str | None = None       # budget / comfort / luxury
    pace: str | None = None              # intensive / relaxed / mixed
    interests: list[str] = Field(default_factory=list)
    region: str | None = None            # asia / europe / americas / oceania / africa / middle_east

    # 补充信息
    departure_city: str | None = None    # e.g. "Vancouver"
    travel_dates: str | None = None      # e.g. "2026-07-01 to 2026-07-07"
    budget_amount: str | None = None     # e.g. "$3000 CAD"
    group_size: int | None = None
    special_needs: list[str] = Field(default_factory=list)

    # 压缩后的叙事摘要（只存结构化字段之外的信息）
    narrative_summary: str = ""

    # 来源追踪：字段名 → "user" | "survey" | "inferred"
    source: dict[str, str] = Field(default_factory=dict)

    # 计划修改意图（extract_context 设置，check_info 读取）
    wants_modification: bool = False


class ExtractedContextPatch(BaseModel):
    """extract_context LLM 的输出 schema。用 with_structured_output 强制填充。
    所有字段 Optional（只输出有变化的），但 wants_modification 必填。"""

    destination: str | None = Field(None, description="用户要去的目的地")
    departure_city: str | None = Field(None, description="出发城市")
    duration_days: int | None = Field(None, description="旅行天数")
    travel_style: str | None = Field(None, description="solo/couple/family/friends")
    budget_tier: str | None = Field(None, description="budget/comfort/luxury")
    pace: str | None = Field(None, description="intensive/relaxed/mixed")
    interests: list[str] | None = Field(None, description="food/culture/nature/shopping/adventure/instagrammable")
    region: str | None = Field(None, description="asia/europe/americas/oceania/africa/middle_east")
    travel_dates: str | None = Field(None, description="旅行日期")
    budget_amount: str | None = Field(None, description="预算金额")
    group_size: int | None = Field(None, description="出行人数")
    special_needs: list[str] | None = Field(None, description="特殊需求")
    wants_modification: bool = Field(
        default=False,
        description="仅当用户明确要求修改已有计划时为 true（如'改成两天''换个酒店'）。"
                    "新建计划、首次提需求、普通聊天时必须为 false。",
    )
    confidence: dict[str, float] = Field(default_factory=dict, description="每个字段的置信度")


# ── Initial Intake（首条消息 LLM tool calling） ──────────────────────────────


class ExtractedIntakeProfile(BaseModel):
    """从用户首条消息中提取的旅行画像。"""
    destination: str | None = Field(None, description="目的地")
    duration_days: int | None = Field(None, description="旅行天数")
    departure_city: str | None = Field(None, description="出发城市")
    travel_style: str | None = Field(None, description="solo/couple/family/friends")
    budget_tier: str | None = Field(None, description="budget/comfort/luxury")
    pace: str | None = Field(None, description="intensive/relaxed/mixed")
    interests: list[str] = Field(default_factory=list, description="food/culture/nature/shopping/adventure/instagrammable")
    region: str | None = Field(None, description="asia/europe/americas/oceania/africa/middle_east")
    travel_dates: str | None = Field(None, description="旅行日期")
    budget_amount: str | None = Field(None, description="预算金额")
    group_size: int | None = Field(None, description="出行人数")
    special_needs: list[str] = Field(default_factory=list, description="特殊需求")


class ShowSurveyParams(BaseModel):
    """Call this tool when info is insufficient and a survey is needed."""
    extracted_profile: ExtractedIntakeProfile
    missing_fields: list[str] = Field(
        description="List of missing field names. Options: destination, duration_days, travel_style, budget_tier, pace, interests"
    )
    greeting: str = Field(
        description="A short friendly response in the USER's language, acknowledging what they said and letting them know you need a few more details."
    )


class ProceedToPlanParams(BaseModel):
    """Call this tool when info is sufficient to generate the plan."""
    complete_profile: ExtractedIntakeProfile
    acknowledgment: str = Field(
        description="A short enthusiastic confirmation in the USER's language, letting them know the travel plan is being generated."
    )


class Activity(BaseModel):
    time_slot: str = Field(description="Start time in HH:MM 24-hour format, e.g. '09:00', '12:30', '18:00'")
    place_name: str = Field(description="景点或餐厅的标准名称，用于 Wikipedia 和社交媒体搜索")
    duration_minutes: int = Field(default=60, description="建议游玩/用餐时长（分钟）")
    description: str = Field(description="活动的详细描述，2-3句话")
    food_recommendation: str = Field(default="", description="推荐菜品，餐厅活动必填")
    estimated_cost: str = Field(default="", description="预估花费")
    is_restaurant: bool = Field(default=False, description="是否为餐厅推荐。中午和晚上的用餐活动必须设为 true")


class HotelRecommendation(BaseModel):
    name: str = Field(description="酒店名称，如 Hotel Gracery Shinjuku")
    area: str = Field(description="所在区域，如 Shinjuku / 新宿")
    price_per_night: str = Field(description="每晚价格估算，如 $150 CAD")
    stars: int = Field(default=3, description="星级 1-5", ge=1, le=5)
    highlight: str = Field(default="", description="亮点：靠近地铁、含早餐、网红打卡等")
    tier: str = Field(description="档次：budget / mid-range / premium")


class DayPlan(BaseModel):
    day: int = Field(description="第几天")
    theme: str = Field(description="当日主题，如'文化探索日''美食之旅'")
    activities: list[Activity] = Field(description="当天的活动列表")
    hotel: HotelRecommendation | None = Field(default=None, description="当晚推荐入住的酒店。最后一天可以不填（不需要住宿）")


class FlightRecommendation(BaseModel):
    airline: str = Field(description="航空公司名称，如 Air Canada, ANA, Japan Airlines")
    route: str = Field(description="航线，如 YVR → NRT")
    price_estimate: str = Field(description="往返价格估算，如 $800 - $1200 CAD")
    tier: str = Field(description="档次：budget / mid-range / premium")
    note: str = Field(default="", description="备注：直飞/转机、飞行时长、推荐原因等")


class TravelPlan(BaseModel):
    title: str = Field(description="旅行计划标题")
    destination: str = Field(description="目的地城市/地区")
    departure_iata: str = Field(default="", description="出发城市的 IATA 机场代码，如 YVR、PEK、LAX。如果用户未提供出发地则留空")
    destination_iata: str = Field(default="", description="目的地的 IATA 机场代码，如 NRT、CDG、BKK")
    start_date: str = Field(default="", description="出发日期，格式 YYYY-MM-DD。如果用户没给具体日期，则根据用户描述推断一个合理的日期（如'下个月'→下月15号，'暑假'→7月1日）")
    end_date: str = Field(default="", description="返程日期，格式 YYYY-MM-DD。根据 start_date + 天数计算")
    duration: str = Field(description="旅行天数")
    overview: str = Field(description="旅行概览，3-5句话总结这次旅行的亮点和特色")
    daily_plans: list[DayPlan] = Field(description="每日行程")
    flight_recommendations: list[FlightRecommendation] = Field(default_factory=list, description="3个机票推荐，分别为 budget / mid-range / premium 三个档次")
    tips: list[str] = Field(default_factory=list, description="实用贴士列表")
    budget_summary: str = Field(default="", description="预算总结，Markdown 表格格式")
