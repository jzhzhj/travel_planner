"""旅行画像分类 + 策略注入系统。

通过对话自然收集的信息，确定用户的旅行画像（profile），
然后加载对应的策略规则注入到计划生成 prompt 中。

画像 = 出行方式 × 预算档次 × 节奏偏好 × 核心兴趣
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── 画像维度 ──────────────────────────────────────────────────────────────────

# 出行方式
TRAVEL_STYLES = {
    "solo": {"zh": "独自旅行", "en": "Solo Travel"},
    "couple": {"zh": "情侣/蜜月", "en": "Couple / Honeymoon"},
    "family": {"zh": "亲子家庭", "en": "Family with Kids"},
    "friends": {"zh": "朋友结伴", "en": "Friends Group"},
}

# 预算档次
BUDGET_TIERS = {
    "budget": {"zh": "穷游", "en": "Budget"},
    "comfort": {"zh": "舒适", "en": "Comfort"},
    "luxury": {"zh": "豪华", "en": "Luxury"},
}

# 节奏偏好
PACE_STYLES = {
    "intensive": {"zh": "暴走打卡型", "en": "Fast-paced Sightseeing"},
    "relaxed": {"zh": "悠闲深度型", "en": "Relaxed & In-depth"},
    "mixed": {"zh": "混合型", "en": "Balanced Mix"},
}

# 核心兴趣（可多选）
INTERESTS = {
    "food": {"zh": "美食", "en": "Food & Cuisine"},
    "culture": {"zh": "文化历史", "en": "Culture & History"},
    "nature": {"zh": "自然户外", "en": "Nature & Outdoors"},
    "shopping": {"zh": "购物", "en": "Shopping"},
    "adventure": {"zh": "冒险刺激", "en": "Adventure & Thrills"},
    "instagrammable": {"zh": "网红打卡", "en": "Instagrammable Spots"},
}


# ── 旅行画像 ──────────────────────────────────────────────────────────────────

@dataclass
class TravelProfile:
    travel_style: str = "friends"       # solo / couple / family / friends
    budget_tier: str = "comfort"        # budget / comfort / luxury
    pace: str = "mixed"                 # intensive / relaxed / mixed
    interests: list[str] = field(default_factory=lambda: ["food", "culture"])
    duration_days: int = 5
    region: str = ""                    # asia / europe / americas / oceania / africa


# ── 策略规则库 ────────────────────────────────────────────────────────────────

# 每日活动上限
PACING_RULES = {
    "intensive": {"max_activities": 5, "rest_day_every": 0, "min_free_time": "1h"},
    "relaxed":   {"max_activities": 2, "rest_day_every": 2, "min_free_time": "3h"},
    "mixed":     {"max_activities": 3, "rest_day_every": 3, "min_free_time": "2h"},
}

# 家庭/亲子额外约束
FAMILY_RULES = {
    "max_activities": 3,
    "nap_break": True,
    "kid_friendly_required": True,
    "no_late_nights": True,
    "stroller_accessible": True,
}

# 预算分配模板 (百分比)
BUDGET_ALLOCATION = {
    "budget": {
        "asia":        {"accommodation": 25, "transport": 20, "food": 30, "activities": 15, "shopping": 10},
        "europe":      {"accommodation": 30, "transport": 30, "food": 20, "activities": 15, "shopping": 5},
        "americas":    {"accommodation": 30, "transport": 25, "food": 25, "activities": 12, "shopping": 8},
        "oceania":     {"accommodation": 30, "transport": 25, "food": 25, "activities": 12, "shopping": 8},
        "africa":      {"accommodation": 25, "transport": 25, "food": 20, "activities": 20, "shopping": 10},
        "middle_east": {"accommodation": 30, "transport": 20, "food": 25, "activities": 15, "shopping": 10},
        "default":     {"accommodation": 25, "transport": 25, "food": 25, "activities": 15, "shopping": 10},
    },
    "comfort": {
        "asia":        {"accommodation": 35, "transport": 20, "food": 25, "activities": 12, "shopping": 8},
        "europe":      {"accommodation": 35, "transport": 25, "food": 20, "activities": 12, "shopping": 8},
        "americas":    {"accommodation": 35, "transport": 22, "food": 22, "activities": 13, "shopping": 8},
        "oceania":     {"accommodation": 35, "transport": 22, "food": 22, "activities": 13, "shopping": 8},
        "africa":      {"accommodation": 30, "transport": 25, "food": 20, "activities": 17, "shopping": 8},
        "middle_east": {"accommodation": 35, "transport": 20, "food": 22, "activities": 15, "shopping": 8},
        "default":     {"accommodation": 35, "transport": 22, "food": 23, "activities": 12, "shopping": 8},
    },
    "luxury": {
        "accommodation": 40, "transport": 20, "food": 20, "activities": 10, "shopping": 10,
    },
}

# 住宿偏好
ACCOMMODATION_STYLE = {
    ("solo", "budget"):     "hostel / capsule hotel / guesthouse",
    ("solo", "comfort"):    "boutique hotel / well-reviewed 3-star",
    ("solo", "luxury"):     "design hotel / 4-star+",
    ("couple", "budget"):   "Airbnb / budget hotel with good reviews",
    ("couple", "comfort"):  "boutique hotel / romantic 4-star",
    ("couple", "luxury"):   "5-star resort / luxury ryokan / iconic hotel",
    ("family", "budget"):   "family room / apartment / Airbnb",
    ("family", "comfort"):  "family-friendly 3-4 star with pool",
    ("family", "luxury"):   "resort with kids club / suite hotel",
    ("friends", "budget"):  "hostel / shared Airbnb",
    ("friends", "comfort"): "apartment / 3-star central location",
    ("friends", "luxury"):  "villa / premium apartment / 4-star+",
}

# 反模式规则（通用 + 画像特定）
ANTI_PATTERNS = {
    "_universal": [
        "连续安排2个大型博物馆（博物馆疲劳）",
        "最后一天安排远郊景点（赶飞机风险）",
        "红眼航班到达当天安排高强度行程",
        "把所有购物集中在最后一天（行李超重风险）",
    ],
    "family": [
        "连续3天以上无儿童友好景点",
        "每天步行超过8公里（小孩走不动）",
        "安排晚上9点以后的活动",
        "全天无午休/下午茶休息",
    ],
    "couple": [
        "每天都是人山人海的热门景点（没有浪漫氛围）",
        "完全没有两人独处的餐厅/咖啡时间",
    ],
    "budget": [
        "推荐高档餐厅而不提供平价替代",
        "忽略免费景点和免费日",
        "没有提到当地交通通票/省钱技巧",
    ],
    "intensive": [
        "景点之间通勤超过1小时却没有说明",
        "没有标注每个景点的游玩时间",
    ],
    "relaxed": [
        "一天塞超过3个景点",
        "没有安排休闲/放空时间",
    ],
}

# 兴趣 → 活动类型权重
INTEREST_ACTIVITY_MIX = {
    "food":           "每天至少安排1个美食体验（当地市场、特色餐厅、烹饪课、街头小吃）",
    "culture":        "每天至少1个文化/历史景点（博物馆、寺庙、古迹、传统体验）",
    "nature":         "每2天至少1个自然景点（公园、山、海滩、徒步），避免连续3天纯室内",
    "shopping":       "安排专门的购物时段和区域，标注营业时间和退税信息",
    "adventure":      "每2天安排1个刺激活动（潜水、滑翔、攀岩等），注意体力恢复",
    "instagrammable": "每天至少1个出片地点，标注最佳拍照时间和角度",
}


# ── 分类 prompt 已迁移到 prompts/classify_system/ ──────────────────────────


# ── 策略生成 ──────────────────────────────────────────────────────────────────

def build_strategy_prompt(profile: TravelProfile, language: str = "zh") -> str:
    """根据旅行画像生成策略指令，注入到 generate_plan 的 system prompt 中。"""

    lang = language
    sections: list[str] = []

    # ── 画像概述
    style_name = TRAVEL_STYLES.get(profile.travel_style, {}).get(lang, profile.travel_style)
    budget_name = BUDGET_TIERS.get(profile.budget_tier, {}).get(lang, profile.budget_tier)
    pace_name = PACE_STYLES.get(profile.pace, {}).get(lang, profile.pace)
    interest_names = [INTERESTS.get(i, {}).get(lang, i) for i in profile.interests]

    if lang == "zh":
        sections.append(f"## 用户旅行画像\n出行方式：{style_name} | 预算：{budget_name} | 节奏：{pace_name} | 兴趣：{'、'.join(interest_names)}")
    else:
        sections.append(f"## Travel Profile\nStyle: {style_name} | Budget: {budget_name} | Pace: {pace_name} | Interests: {', '.join(interest_names)}")

    # ── 节奏规则
    pacing = PACING_RULES.get(profile.pace, PACING_RULES["mixed"])
    if lang == "zh":
        pace_rules = f"## 节奏规则\n- 每天最多 {pacing['max_activities']} 个景点/活动"
        if pacing["rest_day_every"]:
            pace_rules += f"\n- 每连续 {pacing['rest_day_every']} 天高强度后安排 1 天轻松日（spa/咖啡/闲逛）"
        pace_rules += f"\n- 每天保留至少 {pacing['min_free_time']} 自由时间"
    else:
        pace_rules = f"## Pacing Rules\n- Max {pacing['max_activities']} attractions/activities per day"
        if pacing["rest_day_every"]:
            pace_rules += f"\n- After {pacing['rest_day_every']} intensive days, schedule 1 relaxation day (spa/café/stroll)"
        pace_rules += f"\n- Reserve at least {pacing['min_free_time']} free time daily"
    sections.append(pace_rules)

    # ── 家庭特殊规则
    if profile.travel_style == "family":
        if lang == "zh":
            sections.append(
                "## 亲子特殊规则\n"
                "- 每天最多 3 个活动，必须包含至少 1 个儿童友好景点\n"
                "- 下午安排午休/零食时间（14:00-15:00）\n"
                "- 不安排 21:00 以后的活动\n"
                "- 优先选择有电梯/无障碍通道的景点（推婴儿车）\n"
                "- 酒店优先选有家庭房/套间/泳池的"
            )
        else:
            sections.append(
                "## Family Rules\n"
                "- Max 3 activities/day, at least 1 must be kid-friendly\n"
                "- Schedule nap/snack break around 14:00-15:00\n"
                "- No activities after 9 PM\n"
                "- Prefer stroller-accessible venues\n"
                "- Hotels: family rooms / suites / with pool preferred"
            )

    # ── 预算分配
    alloc = BUDGET_ALLOCATION.get(profile.budget_tier, BUDGET_ALLOCATION["comfort"])
    if isinstance(alloc, dict) and "accommodation" not in alloc:
        # 有按区域细分的
        alloc = alloc.get(profile.region, alloc.get("default", alloc))

    if isinstance(alloc, dict) and "accommodation" in alloc:
        if lang == "zh":
            lines = ["## 预算分配建议"]
            for k, v in alloc.items():
                names = {"accommodation": "住宿", "transport": "交通", "food": "餐饮", "activities": "门票/活动", "shopping": "购物"}
                lines.append(f"- {names.get(k, k)}: ~{v}%")
            sections.append("\n".join(lines))
        else:
            lines = ["## Budget Allocation"]
            for k, v in alloc.items():
                lines.append(f"- {k.title()}: ~{v}%")
            sections.append("\n".join(lines))

    # ── 住宿偏好
    acc_key = (profile.travel_style, profile.budget_tier)
    acc_style = ACCOMMODATION_STYLE.get(acc_key, "3-4 star hotel")
    if lang == "zh":
        sections.append(f"## 住宿偏好\n推荐类型：{acc_style}")
    else:
        sections.append(f"## Accommodation Preference\nRecommended type: {acc_style}")

    # ── 兴趣活动比例
    interest_rules = []
    for interest in profile.interests:
        rule = INTEREST_ACTIVITY_MIX.get(interest)
        if rule:
            interest_rules.append(f"- {rule}")
    if interest_rules:
        header = "## 兴趣活动要求" if lang == "zh" else "## Interest-based Activity Mix"
        sections.append(header + "\n" + "\n".join(interest_rules))

    # ── 反模式
    anti = list(ANTI_PATTERNS.get("_universal", []))
    anti.extend(ANTI_PATTERNS.get(profile.travel_style, []))
    anti.extend(ANTI_PATTERNS.get(profile.budget_tier, []))
    anti.extend(ANTI_PATTERNS.get(profile.pace, []))
    if anti:
        header = "## 避免以下问题" if lang == "zh" else "## Anti-patterns to Avoid"
        sections.append(header + "\n" + "\n".join(f"- ❌ {a}" for a in anti))

    return "\n\n".join(sections)
