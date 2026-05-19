# AI Travel Planner — 项目架构文档

## 目录

1. [项目概览](#1-项目概览)
2. [目录结构](#2-目录结构)
3. [核心流程 — LangGraph 状态图](#3-核心流程--langgraph-状态图)
4. [State 数据结构](#4-state-数据结构)

5. [Prompt 管理系统](#5-prompt-管理系统)
6. [用户画像分类系统](#6-用户画像分类系统)
7. [策略注入机制](#7-策略注入机制)
8. [上下文管理（3 层）](#8-上下文管理3-层)
9. [Pydantic 数据模型](#9-pydantic-数据模型)
10. [工具系统 tools/](#10-工具系统-tools)
11. [RAG 知识检索系统](#11-rag-知识检索系统)
12. [HTML 渲染器](#12-html-渲染器)
13. [外部 API 集成](#13-外部-api-集成)
14. [前端架构](#14-前端架构)
15. [Web 服务 app.py](#15-web-服务-apppy)
16. [环境变量](#16-环境变量)
17. [依赖清单](#17-依赖清单)

---

## 1. 项目概览

基于 LangGraph + FastAPI 的 AI 旅行规划助手。通过自然对话收集用户需求，自动分类用户画像，生成个性化结构化旅行计划，并渲染为包含地图、视频、实时机票酒店数据的交互式 HTML 页面。

**技术栈：**

| 层 | 技术 |
|---|------|
| LLM | OpenAI GPT-4o-mini（对话）+ GPT-4.1（计划生成，32k 输出） |
| Agent 框架 | LangGraph StateGraph + ToolNode |
| 知识检索 | ChromaDB 向量库 + 余弦相似度 |
| Web 框架 | FastAPI + SSE 流式响应 |
| 前端 | 原生 HTML/CSS/JS 单页应用 |
| 地图 | Apple MapKit JS / Google Maps（双供应商） |
| 机票酒店 | Amadeus Self-Service API（OAuth2） |
| 视频 | YouTube Data API / Bilibili 公开 API |
| 短视频 | TikTok oEmbed / Instagram oEmbed / 小红书 |

---

## 2. 目录结构

```
ai_travel_planner/
│
├── app.py                        # FastAPI Web 入口（SSE 流式响应）
├── main.py                       # CLI 命令行入口
├── graph.py                      # LangGraph 状态图（核心流程编排）
├── models.py                     # Pydantic 数据模型
├── profiles.py                   # 用户画像 + 策略规则库
├── renderer.py                   # HTML 旅行计划渲染器
├── wiki.py                       # Wikipedia + Unsplash 景点信息
├── .env                          # API 密钥（不入版本管理）
│
├── prompts/                      # Prompt 版本管理
│   ├── __init__.py               #   导出 PromptManager, prompt_manager
│   ├── config.yaml               #   版本配置（改这里切版本）
│   ├── manager.py                #   PromptManager 类
│   ├── chat_system/              #   对话收集节点 prompt
│   │   ├── v1.zh.md
│   │   └── v1.en.md
│   ├── check_info_system/        #   流程控制节点 prompt
│   │   ├── v1.zh.md
│   │   └── v1.en.md
│   ├── plan_system/              #   计划生成节点 prompt
│   │   ├── v1.zh.md
│   │   └── v1.en.md
│   ├── classify_system/          #   画像分类节点 prompt
│   │   ├── v1.zh.md
│   │   └── v1.en.md
│   ├── summarize/                #   对话摘要 prompt
│   │   ├── v1.zh.md
│   │   └── v1.en.md
│   └── region_detect/            #   区域检测 prompt
│       └── v1.md
│
├── tools/                        # LangChain 工具集
│   ├── __init__.py               #   ALL_TOOLS 注册列表
│   ├── travel_api.py             #   Amadeus 机票/酒店 API
│   ├── embeds.py                 #   TikTok/Instagram/小红书 嵌入搜索
│   ├── links.py                  #   YouTube/Bilibili 视频推荐
│   ├── place_search.py           #   Wikipedia 景点搜索工具
│   ├── weather.py                #   wttr.in 天气查询工具
│   ├── currency.py               #   汇率转换工具
│   └── holidays.py               #   节假日查询工具
│
├── rag/                          # RAG 知识检索
│   ├── store.py                  #   ChromaDB 初始化
│   ├── loader.py                 #   种子数据加载器
│   ├── retriever.py              #   检索工具（向量搜索 + 元数据过滤）
│   ├── seed_knowledge.json       #   种子知识库（JSON）
│   └── chroma_db/                #   ChromaDB 持久化目录
│
└── static/                       # 前端静态文件
    └── index.html                #   SPA 单页应用（含 survey 表单）
```

---

## 3. 核心流程 — LangGraph 状态图

**文件：** `graph.py`

```
用户消息
    │
    ▼
┌─────────────────────────────────┐
│  chat                           │  Prompt: chat_system
│  自然对话 + 工具调用             │  LLM: gpt-4o-mini (temp=0.7)
│  滑动窗口 20 条 + 摘要注入       │  工具: ALL_TOOLS (5个)
└────────┬───────────┬────────────┘
         │           │
    有 tool_calls    无 tool_calls
         │           │
         ▼           ▼
   tool_executor  check_info ─────── Prompt: check_info_system
   (ToolNode)     判断信息是否充足     输出: {"action":"generate"|"continue_chat"}
         │           │
         │      ┌────┴────┐
         │      │         │
         │  continue   generate
         │   _chat        │
         │      │         ▼
         │     END   classify_profile ─── Prompt: classify_system
         │           (用户画像分类)        或用前端表单跳过
         │                │                + region_detect 补区域
         └→ 回到 chat      ▼
                     generate_plan ────── Prompt: plan_system + 策略注入
                     结构化计划生成         LLM: gpt-4.1 (temp=0.7, 32k)
                     → TravelPlan          → Pydantic 结构化输出
                           │
                           ▼
                     enrich_plan
                     ├── Wikipedia 图片/摘要/坐标
                     ├── YouTube/Bilibili 视频
                     ├── TikTok/Instagram/小红书 短视频
                     ├── Amadeus 实时机票/酒店
                     └── render_plan → HTML
                           │
                           ▼
                          END
```

**LLM 配置：**

| 节点 | 模型 | 温度 | max_tokens | 用途 |
|------|------|------|-----------|------|
| chat | gpt-4o-mini | 0.7 | 默认 | 对话、工具调用 |
| check_info | gpt-4o-mini | 0.7 | 默认 | JSON 判断 |
| classify_profile | gpt-4o-mini | 0.7 | 默认 | 画像分类 JSON |
| generate_plan | gpt-4.1 | 0.7 | 32768 | 结构化计划（需大输出） |
| _summarize_conversation | gpt-4o-mini | 0.7 | 默认 | 对话摘要 |

**路由逻辑：**

| 路由函数 | 输入 | 条件 | 去向 |
|---------|------|------|------|
| `route_after_chat` | chat 输出的最后一条消息 | `last_msg.tool_calls` 非空 | `tool_executor` |
| | | 否则 | `check_info` |
| `route_after_check` | check_info 输出的 JSON | `action == "generate"` | `classify_profile` |
| | | 否则 | `END` |

---

## 4. State 数据结构

```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 完整对话历史
    plan_generated: bool                                   # 是否已生成过计划
    plan_data: TravelPlan | None                          # 结构化计划数据
    html_path: str                                         # 渲染后的 HTML 路径
    language: str                                          # "zh" | "en"
    currency: str                                          # "CAD" | "USD" 等
    conversation_summary: str                              # LLM 生成的对话摘要
    travel_profile: TravelProfile | None                  # 用户旅行画像
```

**消息类型在 `messages` 中的角色：**

| 类型 | 来源 | 用途 |
|------|------|------|
| `HumanMessage` | 用户输入 | 对话内容 |
| `AIMessage` | LLM 回复 | 对话回复或工具调用请求 |
| `AIMessage.tool_calls` | LLM | 请求执行工具（不面向用户展示） |
| `ToolMessage` | 工具返回 | 天气/RAG/景点等查询结果（不面向用户展示） |

---

## 5. Prompt 管理系统

**文件：** `prompts/manager.py`、`prompts/config.yaml`

### 5.1 PromptManager 类

```python
from prompts import prompt_manager

# 加载 prompt（版本从 config.yaml 读取）
text = prompt_manager.load("chat_system", lang="zh", currency="CAD")

# 强制指定版本
text = prompt_manager.load("plan_system", lang="en", version="v2", currency="USD", today="2026-07-01")

# 查看本次会话使用的版本
print(prompt_manager.usage_log)  # {"chat_system": "v1", "plan_system": "v2"}

# 列出可用版本
prompt_manager.list_versions("chat_system")  # ["v1", "v2"]

# 热更新配置（改了 config.yaml 后不用重启）
prompt_manager.reload_config()
```

**API：**

| 方法 | 说明 |
|------|------|
| `load(name, lang, version, **vars)` | 加载并渲染模板，记录版本 |
| `get_version(name)` | 读取 config.yaml 中的当前版本 |
| `list_versions(name)` | 列出某 prompt 所有可用版本 |
| `reload_config()` | 热更新配置 + 清缓存 |
| `usage_log` (property) | 返回 `{name: version}` 使用记录 |

### 5.2 模板语法

- 变量占位符：`${variable}`（不与 JSON `{}` 冲突）
- 文件命名：`{version}.{lang}.md`（如 `v1.zh.md`）或 `{version}.md`（无语言区分）
- 查找顺序：先找 `v1.zh.md`，找不到则回退到 `v1.md`

### 5.3 版本切换流程

```
1. 新建文件:  prompts/chat_system/v2.zh.md
              prompts/chat_system/v2.en.md
2. 改配置:    prompts/config.yaml → chat_system.version: v2
3. 零代码改动，立即生效
```

### 5.4 所有 Prompt 清单

| 名称 | 变量 | 对应节点 | 核心指令 |
|------|------|---------|---------|
| `chat_system` | `${currency}` | chat | 自然对话收集 6 项需求（目的地/时间/预算/人数/兴趣/特殊需求），信息够了别追问，主动用工具辅助 |
| `check_info_system` | 无 | check_info | 3 条必要信息（目的地+天数+偏好）全满足→`"generate"`，否则→`"continue_chat"`，输出严格 JSON |
| `plan_system` | `${currency}` `${today}` | generate_plan | 生成完整 TravelPlan：IATA 代码、日期推断、`duration_minutes`、3 档机票推荐、每日酒店、Markdown 预算表 |
| `classify_system` | 无 | classify_profile | 从对话关键词推断 `travel_style/budget_tier/pace/interests/region` 五个字段，输出严格 JSON |
| `summarize` | 无 | _summarize_conversation | 用要点概括对话中所有旅行需求信息，不遗漏 |
| `region_detect` | 无 | classify_profile (补充) | 只返回一个区域词：`asia/europe/americas/oceania/africa/middle_east` |

### 5.5 Prompt 内容详情

#### chat_system（对话收集）

**作用：** chat 节点的 system prompt，指导 LLM 通过自然聊天收集旅行需求。

**核心指令：**
- 收集 6 项关键信息：目的地、出行时间/天数、预算范围、同行人数和关系、兴趣偏好、特殊需求
- 自然聊天，不要像问卷逐条询问
- 信息已够（目的地+天数+至少一个偏好）→ 不再追问，直接确认
- 主动使用工具：`search_travel_knowledge`（优先）、`get_weather`、`get_holidays`、`search_place_info`、`convert_currency`
- 已生成过计划后：修改需求/回答问题/规划新旅行 三种模式

#### check_info_system（流程控制）

**作用：** 判断是否已收集到足够信息来触发计划生成。

**判断规则：**
- `plan_generated = false` 时：目的地（具体城市/地区/国家）+ 天数 + 至少一种兴趣 → 三项全满足才 generate
- `plan_generated = true` 时：用户要求修改 → generate，否则 → continue_chat
- 输出格式：`{"action": "generate"|"continue_chat", "reason": "..."}`

#### plan_system（计划生成）

**作用：** generate_plan 节点的 system prompt，指导 LLM 输出完整的 TravelPlan 结构化数据。

**核心指令：**
- IATA 机场代码：出发地和目的地
- 日期推断："下个月"→下月15号，"暑假"→7月1日，没说→一个月后
- 活动 `duration_minutes`：根据景点规模估算（咖啡厅=30min，博物馆=120-180min，乐园=300-480min）
- `place_name` 用标准名称（用于 Wikipedia 搜索）
- 3 档机票推荐：budget / mid-range / premium
- 每日酒店推荐：就近原则，含星级/价格/亮点
- 预算表 Markdown 格式

**策略注入点：** `build_strategy_prompt(profile)` 的输出会追加在此 prompt 之后，形成完整的系统指令。

#### classify_system（画像分类）

**作用：** 从对话内容推断用户的 5 维旅行画像。

**推断规则（关键词映射）：**
- 蜜月/老婆/男女朋友 → couple；孩子/宝宝 → family；朋友/同学 → friends；无同行者 → solo
- 穷游/省钱/青旅 → budget；五星/奢华/商务舱 → luxury；默认 → comfort
- 打卡/景点多 → intensive；悠闲/放松/度假 → relaxed；默认 → mixed
- `interests` 从对话提取，至少一个
- `region` 从目的地推断

#### summarize（对话摘要）

**作用：** 当对话超过 `CHAT_WINDOW_SIZE`（20 条）时，用 LLM 压缩历史对话为要点。

**触发条件：** `len(messages) > CHAT_WINDOW_SIZE` 且 `conversation_summary` 为空

#### region_detect（区域检测）

**作用：** 当用户通过前端表单提交画像但缺少 `region` 字段时，用 LLM 从对话推断目的地所属区域。

**可选值：** `asia / europe / americas / oceania / africa / middle_east`

---

## 6. 用户画像分类系统

**文件：** `profiles.py`

### 6.1 画像数据结构

```python
@dataclass
class TravelProfile:
    travel_style: str = "friends"       # solo / couple / family / friends
    budget_tier: str = "comfort"        # budget / comfort / luxury
    pace: str = "mixed"                 # intensive / relaxed / mixed
    interests: list[str] = ["food", "culture"]  # 可多选
    duration_days: int = 5
    region: str = ""                    # asia / europe / americas / oceania / africa / middle_east
```

### 6.2 四个维度

**维度一：出行方式 `travel_style`**

| 值 | 中文 | 英文 | 触发词 |
|---|------|------|--------|
| `solo` | 独自旅行 | Solo Travel | 无同行者提及 |
| `couple` | 情侣/蜜月 | Couple / Honeymoon | 蜜月、老婆、男/女朋友 |
| `family` | 亲子家庭 | Family with Kids | 孩子、宝宝、儿子、女儿 |
| `friends` | 朋友结伴 | Friends Group | 朋友、同学、同事、一群 |

**维度二：预算档次 `budget_tier`**

| 值 | 中文 | 英文 | 触发词 |
|---|------|------|--------|
| `budget` | 穷游 | Budget | 穷游、省钱、便宜、背包、青旅 |
| `comfort` | 舒适 | Comfort | 默认（无特殊提及） |
| `luxury` | 豪华 | Luxury | 五星、奢华、豪华、商务舱 |

**维度三：节奏偏好 `pace`**

| 值 | 中文 | 英文 | 触发词 |
|---|------|------|--------|
| `intensive` | 暴走打卡型 | Fast-paced | 打卡、拍照、必去、景点多 |
| `relaxed` | 悠闲深度型 | Relaxed | 悠闲、慢旅行、不赶、放松 |
| `mixed` | 混合型 | Balanced | 默认 |

**维度四：核心兴趣 `interests`（可多选）**

| 值 | 中文 | 英文 |
|---|------|------|
| `food` | 美食 | Food & Cuisine |
| `culture` | 文化历史 | Culture & History |
| `nature` | 自然户外 | Nature & Outdoors |
| `shopping` | 购物 | Shopping |
| `adventure` | 冒险刺激 | Adventure & Thrills |
| `instagrammable` | 网红打卡 | Instagrammable Spots |

### 6.3 分类流程

```
用户打开页面
       │
  ┌────┴────────────────┐
  │  填了前端 survey 表单？ │
  └────┬──────────┬─────┘
       是          否
       │           │
       ▼           ▼
  app.py 转换       classify_profile 节点
  ProfileData →     LLM 从对话推断全部 5 个字段
  TravelProfile     使用 classify_system prompt
       │           │
       ├───────────┘
       ▼
  region 为空？ → 是 → LLM 补充 region（region_detect prompt）
       │
       ▼
  完整 TravelProfile → 注入 state["travel_profile"]
```

**前端表单到后端的数据流：**

```
前端 survey → ChatRequest.profile (ProfileData)
    → app.py: TravelProfile(travel_style=..., budget_tier=..., pace=..., interests=...)
    → state["travel_profile"]
    → classify_profile 检测到已有 profile → 跳过 LLM 分类，只补 region
    → generate_plan 读取 profile → build_strategy_prompt() → 注入 system prompt
```

---

## 7. 策略注入机制

**文件：** `profiles.py` — `build_strategy_prompt(profile, language)`

画像分类后，`build_strategy_prompt()` 根据用户 `TravelProfile` 生成一段策略文本，追加到 `plan_system` prompt 之后，作为 generate_plan 的完整 system prompt。

### 7.1 策略组成

```
plan_system prompt（基础模板）
    +
build_strategy_prompt 输出：
    ├── § 用户旅行画像概述
    ├── § 节奏规则
    ├── § 家庭特殊规则（仅 family）
    ├── § 预算分配建议
    ├── § 住宿偏好
    ├── § 兴趣活动要求
    └── § 反模式检查
```

### 7.2 节奏规则 `PACING_RULES`

| pace | 每天最多活动 | 休息日频率 | 每天最少自由时间 |
|------|------------|----------|---------------|
| `intensive` | 5 个 | 不安排休息日 | 1 小时 |
| `mixed` | 3 个 | 每连续 3 天安排 1 天轻松日 | 2 小时 |
| `relaxed` | 2 个 | 每连续 2 天安排 1 天轻松日 | 3 小时 |

### 7.3 家庭特殊规则 `FAMILY_RULES`（仅 `travel_style = family` 触发）

- 每天最多 3 个活动，至少 1 个儿童友好
- 下午 14:00-15:00 午休/零食时间
- 不安排 21:00 以后的活动
- 优先无障碍通道（推婴儿车）
- 酒店优先家庭房/套间/泳池

### 7.4 预算分配 `BUDGET_ALLOCATION[budget_tier][region]`

按 `budget_tier × region` 交叉查表，返回 5 项百分比：

| 区域 | 适用 tier | 住宿% | 交通% | 餐饮% | 活动% | 购物% |
|------|----------|-------|-------|-------|-------|-------|
| `asia` | budget | 25 | 20 | 30 | 15 | 10 |
| `asia` | comfort | 35 | 20 | 25 | 12 | 8 |
| `europe` | budget | 30 | 30 | 20 | 15 | 5 |
| `europe` | comfort | 35 | 25 | 20 | 12 | 8 |
| `americas` | budget | 30 | 25 | 25 | 12 | 8 |
| `americas` | comfort | 35 | 22 | 22 | 13 | 8 |
| `oceania` | budget | 30 | 25 | 25 | 12 | 8 |
| `oceania` | comfort | 35 | 22 | 22 | 13 | 8 |
| `africa` | budget | 25 | 25 | 20 | 20 | 10 |
| `africa` | comfort | 30 | 25 | 20 | 17 | 8 |
| `middle_east` | budget | 30 | 20 | 25 | 15 | 10 |
| `middle_east` | comfort | 35 | 20 | 22 | 15 | 8 |
| `luxury`（所有区域统一） | — | 40 | 20 | 20 | 10 | 10 |

### 7.5 住宿偏好 `ACCOMMODATION_STYLE[(travel_style, budget_tier)]`

12 种组合：

| 出行方式 × 预算 | 推荐住宿类型 |
|---------------|------------|
| solo + budget | hostel / capsule hotel / guesthouse |
| solo + comfort | boutique hotel / well-reviewed 3-star |
| solo + luxury | design hotel / 4-star+ |
| couple + budget | Airbnb / budget hotel with good reviews |
| couple + comfort | boutique hotel / romantic 4-star |
| couple + luxury | 5-star resort / luxury ryokan / iconic hotel |
| family + budget | family room / apartment / Airbnb |
| family + comfort | family-friendly 3-4 star with pool |
| family + luxury | resort with kids club / suite hotel |
| friends + budget | hostel / shared Airbnb |
| friends + comfort | apartment / 3-star central location |
| friends + luxury | villa / premium apartment / 4-star+ |

### 7.6 兴趣活动配比 `INTEREST_ACTIVITY_MIX`

| 兴趣 | 规则 |
|------|------|
| `food` | 每天至少 1 个美食体验（市场、餐厅、烹饪课、街头小吃） |
| `culture` | 每天至少 1 个文化/历史景点（博物馆、寺庙、古迹） |
| `nature` | 每 2 天至少 1 个自然景点，避免连续 3 天纯室内 |
| `shopping` | 安排专门购物时段和区域，标注营业时间和退税信息 |
| `adventure` | 每 2 天 1 个刺激活动（潜水、滑翔、攀岩），注意体力恢复 |
| `instagrammable` | 每天至少 1 个出片地点，标注最佳拍照时间和角度 |

### 7.7 反模式检查 `ANTI_PATTERNS`

**通用（所有画像）：**
- 连续安排 2 个大型博物馆（博物馆疲劳）
- 最后一天安排远郊景点（赶飞机风险）
- 红眼航班到达当天安排高强度行程
- 所有购物集中在最后一天（行李超重风险）

**family 特有：**
- 连续 3 天以上无儿童友好景点
- 每天步行超过 8 公里
- 安排晚上 9 点以后的活动
- 全天无午休/下午茶休息

**couple 特有：**
- 每天都是人山人海的热门景点
- 完全没有两人独处的餐厅/咖啡时间

**budget 特有：**
- 推荐高档餐厅而不提供平价替代
- 忽略免费景点和免费日
- 没有提到当地交通通票/省钱技巧

**intensive 特有：**
- 景点之间通勤超过 1 小时却没有说明
- 没有标注每个景点的游玩时间

**relaxed 特有：**
- 一天塞超过 3 个景点
- 没有安排休闲/放空时间

---

## 8. 上下文管理（3 层）

**文件：** `graph.py` 中的辅助函数

### 8.1 滑动窗口 `_trim_messages()`

```
对话长度 ≤ 20 条 → 保持原样
对话长度 > 20 条 → 只保留最近 20 条
    + ToolMessage 不能作为第一条（确保 AIMessage tool_calls 在前）
```

- **应用位置：** `chat` 节点（每次对话前截断）、`_build_plan_messages()`（取最近 10 条）
- **配置：** `CHAT_WINDOW_SIZE = 20`、`PLAN_RECENT_MSGS = 10`

### 8.2 对话摘要 `_summarize_conversation()`

```
对话长度 > CHAT_WINDOW_SIZE 且 conversation_summary 为空
    → 用 gpt-4o-mini 将最近 20 条 Human/AI 消息压缩为要点摘要
    → 保存到 state["conversation_summary"]
    → 后续 chat/generate_plan 节点注入摘要作为上下文
```

- **注入方式：** 作为 `HumanMessage(content="[对话摘要] ...")` 插入消息列表头部
- **只处理：** `HumanMessage` 和 `AIMessage`（跳过 Tool 消息和 `[context]` 开头的消息）

### 8.3 RAG 上下文提取 `_extract_rag_context()`

```
遍历所有 ToolMessage
    → 筛选含 "知识库检索结果" 或 "Knowledge base" 的内容
    → 每条截断到 RAG_MAX_CHARS (1500) 字符
    → 用前 200 字符去重
    → 合并为独立上下文块
```

- **应用位置：** `_build_plan_messages()` — 注入到 generate_plan 的消息列表中
- **注入方式：** 作为 `HumanMessage(content="[知识库参考资料] ...")` 独立插入

### 8.4 generate_plan 的消息构建

```python
_build_plan_messages(state) → [
    SystemMessage(plan_system + strategy),   # 基础 prompt + 画像策略
    HumanMessage("[对话摘要] ..."),            # 可选：长对话时的摘要
    HumanMessage("[知识库参考资料] ..."),       # 可选：RAG 检索结果
    ...最近 10 条原始消息...                    # 保留细节
]
```

---

## 9. Pydantic 数据模型

**文件：** `models.py`

```python
class Activity(BaseModel):
    time_slot: str        # "上午" / "下午" / "晚上" / "清晨"
    place_name: str       # 标准名称（用于 Wikipedia 搜索）
    duration_minutes: int # 建议游玩时长（分钟）
    description: str      # 详细描述，2-3 句话
    food_recommendation: str  # 具体餐厅/菜品名
    estimated_cost: str   # "${currency}" 格式

class HotelRecommendation(BaseModel):
    name: str             # 酒店真实名称
    area: str             # 所在区域
    price_per_night: str  # 每晚价格
    stars: int            # 1-5 星
    highlight: str        # 推荐亮点
    tier: str             # "budget" / "mid-range" / "premium"

class DayPlan(BaseModel):
    day: int              # 第几天
    theme: str            # 当日主题
    activities: list[Activity]
    hotel: HotelRecommendation | None  # 当晚住宿（最后一天可空）

class FlightRecommendation(BaseModel):
    airline: str          # 航空公司名称
    route: str            # 航线（如 YVR → NRT）
    price_estimate: str   # 往返价格范围
    tier: str             # "budget" / "mid-range" / "premium"
    note: str             # 直飞/转机、飞行时长

class TravelPlan(BaseModel):
    title: str            # 计划标题
    destination: str      # 目的地
    departure_iata: str   # 出发机场 IATA
    destination_iata: str # 目的地机场 IATA
    start_date: str       # YYYY-MM-DD
    end_date: str         # YYYY-MM-DD
    duration_days: int
    overview: str         # 概览描述
    daily_plans: list[DayPlan]
    flight_recommendations: list[FlightRecommendation]
    tips: list[str]       # 旅行贴士
    budget_summary: str   # Markdown 表格
```

---

## 10. 工具系统 tools/

**文件：** `tools/__init__.py` 注册 `ALL_TOOLS` 列表

| 工具名 | 文件 | 外部 API | 描述 |
|-------|------|---------|------|
| `search_travel_knowledge` | `rag/retriever.py` | ChromaDB 本地 | 从知识库检索经验证的旅行攻略 |
| `get_weather` | `tools/weather.py` | wttr.in | 查询城市天气和未来预报 |
| `get_holidays` | `tools/holidays.py` | Nager.Date / Calendarific | 查询国家节假日 |
| `search_place_info` | `tools/place_search.py` | Wikipedia REST API | 查询景点详细信息和坐标 |
| `convert_currency` | `tools/currency.py` | frankfurter.dev | 汇率换算 |

**非 LangChain 工具（enrich_plan 直接调用）：**

| 模块 | 外部 API | 描述 |
|------|---------|------|
| `tools/travel_api.py` | Amadeus OAuth2 | 实时机票/酒店报价 |
| `tools/links.py` | YouTube / Bilibili | 视频推荐 |
| `tools/embeds.py` | TikTok / Instagram / 小红书 | 短视频嵌入 |
| `wiki.py` | Wikipedia + Unsplash + Nominatim | 景点信息/图片/坐标 |

### 10.1 travel_api.py — Amadeus 集成

**认证：** OAuth2 `client_credentials`，token 缓存 + 自动刷新（<60s 剩余时重取）

**数据流：**

```
fetch_travel_deals(origin, dest, start_date, end_date, currency, language)
    ├── fetch_flight_deals()
    │     └── POST /v2/shopping/flight-offers
    │         → FlightDeal[](airline, price, transfers, booking_url)
    │
    └── fetch_hotel_deals()
          ├── GET /v1/reference-data/locations/hotels/by-city?cityCode=XXX
          │     → hotel_ids[]
          └── GET /v3/shopping/hotel-offers?hotelIds=...
                → HotelDeal[](name, stars, price, booking_url)
```

**Booking 链接（Travelpayouts 联盟）：**
- 机票：`tp.media/r?marker={MARKER}&u=aviasales.com/search/{路线}`
- 酒店：`hotellook.com/hotels/{目的地}`

**航空公司映射：** 100+ IATA 代码 → 全名（如 `AC` → `Air Canada`）

### 10.2 links.py — 视频推荐

| 语言 | 平台 | API | 需要 Key |
|------|------|-----|---------|
| zh | Bilibili | `api.bilibili.com/x/web-interface/search/type` | 不需要 |
| en | YouTube | YouTube Data API v3 (search.list + videos.list) | 需要 |

### 10.3 embeds.py — 短视频嵌入

| 语言 | 平台 | 搜索方式 | 元数据获取 |
|------|------|---------|----------|
| zh | 小红书 | DuckDuckGo → URL 提取 | 搜索结果标题 |
| en | TikTok | DuckDuckGo → URL 提取 | oEmbed API |
| en | Instagram | DuckDuckGo → URL 提取 | oEmbed API (需 token) |

### 10.4 holidays.py — 节假日

- 121 个国家的中英文名称 → ISO 代码映射
- 优先 Nager.Date（免费）→ 回退 Calendarific（需 API key）

---

## 11. RAG 知识检索系统

**文件：** `rag/store.py`、`rag/loader.py`、`rag/retriever.py`

### 11.1 架构

```
rag/seed_knowledge.json
    → loader.py (add_entry) → ChromaDB (travel_knowledge collection)
    → retriever.py (search_travel_knowledge) → 工具结果 → LLM
```

### 11.2 ChromaDB 配置

| 参数 | 值 |
|------|---|
| 持久化路径 | `rag/chroma_db/` |
| Collection 名 | `travel_knowledge` |
| Embedding | `DefaultEmbeddingFunction()`（Chroma 内置） |
| 距离度量 | cosine |

### 11.3 文档元数据

```python
{
    "city": "东京",
    "country": "日本",
    "category": "attraction",   # attraction / restaurant / transport / tip / culture
    "season": "spring",         # spring / summer / autumn / winter / all
    "tags": "寺庙,浅草,历史",
    "source": "编辑整理"
}
```

### 11.4 检索逻辑

```python
search_travel_knowledge(query, city=None, category=None, season=None, n_results=3)
    → ChromaDB.query(query_texts=[query], n_results, where=filters)
    → 每条结果截断到 500 字符
    → 返回 "📚 知识库检索结果" 格式
    → 如果带过滤无结果 → 回退到无过滤检索
```

### 11.5 种子数据示例

当前种子数据以日本为主（东京、京都），包含：
- 浅草寺、筑地市场、秋叶原等景点介绍
- 东京地铁票券指南
- 樱花赏花季节攻略
- 伏见稻荷大社、京都红叶等

---

## 12. HTML 渲染器

**文件：** `renderer.py`

### 12.1 输入数据

```python
render_plan(
    plan: TravelPlan,
    places: dict[str, PlaceInfo],       # 景点信息（摘要/图片/坐标）
    links: dict[str, RecommendationLinks],  # 视频推荐
    embeds: dict[str, PlaceEmbeds],     # 短视频嵌入
    travel_deals: TravelDeals,          # 实时机票/酒店数据
    language: str = "zh"
) → str  # HTML
```

### 12.2 HTML 结构

```
<html>
├── <head> (CSS 样式)
├── <body>
│   ├── 标题 + 概览
│   ├── 机票推荐区
│   │   ├── LLM 推荐（3 档）
│   │   └── Amadeus 实时数据（航司 logo + 价格 + 预订链接）
│   │
│   ├── 每日行程（循环 DayPlan）
│   │   ├── 日期 + 主题
│   │   ├── 地图（当日所有景点标记 + 路线）
│   │   ├── 活动卡片（循环 Activity）
│   │   │   ├── Wikipedia 图片
│   │   │   ├── 景点名称 + 描述
│   │   │   ├── ⏱ 游玩时长（duration_minutes）
│   │   │   ├── 🍜 美食推荐
│   │   │   ├── 💰 费用估算
│   │   │   └── 📍 坐标
│   │   ├── 视频推荐（YouTube/Bilibili 卡片）
│   │   ├── 短视频嵌入（TikTok/Instagram/小红书）
│   │   └── 当日酒店推荐
│   │       ├── LLM 推荐
│   │       └── Amadeus 实时酒店数据
│   │
│   ├── 旅行贴士
│   ├── 预算汇总表
│   └── 全局地图（所有天全部景点）
│
└── <script>
    ├── Apple MapKit JS 或 Google Maps JS
    ├── 彩色标记（7 种颜色按天循环）
    ├── 日地图 + 全局地图
    └── 路线连线（polyline）
```

### 12.3 地图供应商

| 优先级 | 供应商 | 环境变量 | 功能 |
|--------|--------|---------|------|
| 1 | Apple MapKit JS | `APPLE_MAPKIT_TOKEN` | 标记 + 路线 + 信息窗口 |
| 2 | Google Maps | `GOOGLE_MAPS_API_KEY` | 编号标记 + 信息窗口 |
| 3 | 无地图 | — | 只显示坐标文字 |

---

## 13. 外部 API 集成

| 服务 | API | 用途 | 免费额度 | 需要 Key |
|------|-----|------|---------|---------|
| OpenAI | Chat Completions | LLM 核心 | $5 试用 | 是 |
| Amadeus | OAuth2 + Flight/Hotel Search | 实时机票酒店 | 2000 调用/月 | 是 |
| Wikipedia | REST API v1 | 景点信息/图片 | 无限 | 否 |
| Unsplash | Search Photos | 高质量图片 | 50 次/小时 | 是 |
| wttr.in | Weather | 天气预报 | 无限 | 否 |
| frankfurter.dev | Exchange Rates | 汇率换算 | 无限 | 否 |
| Nager.Date | Public Holidays | 节假日 | 无限 | 否 |
| Calendarific | Holidays | 节假日（备用） | 1000 次/月 | 是 |
| YouTube Data API v3 | Search + Videos | 视频推荐 | 10k units/天 | 是 |
| Bilibili | Web Search | 中文视频 | 无限 | 否 |
| DuckDuckGo | Text Search | 短视频 URL 发现 | 无限 | 否 |
| TikTok oEmbed | `tiktok.com/oembed` | 视频元数据 | 无限 | 否 |
| Instagram oEmbed | Facebook Graph | Reel 元数据 | 需 token | 是 |
| Nominatim (OSM) | Geocoding | 坐标回退 | 1 次/秒 | 否 |
| Travelpayouts | 联盟链接 | 预订跳转+佣金 | 无限 | 是(marker) |
| Apple MapKit JS | Maps SDK | 交互式地图 | 25 万/天 | 是(token) |
| Google Maps JS | Maps SDK | 交互式地图（备用） | $200/月 | 是 |

---

## 14. 前端架构

**文件：** `static/index.html`（单页应用）

### 14.1 布局

```
┌───────────────────────────────────────────────────────┐
│  Header: Logo | 语言切换(中/英) | 货币选择 | 状态指示    │
├─────────────────────┬─────────────────────────────────┤
│                     │                                 │
│   聊天面板 (40%)     │      预览面板 (60%)              │
│                     │                                 │
│  ┌───────────────┐  │  ┌────────────────────────────┐ │
│  │ 消息列表       │  │  │                            │ │
│  │  用户消息      │  │  │   生成前: 占位图 + 引导文字  │ │
│  │  助手回复      │  │  │                            │ │
│  │  进度条        │  │  │   生成后: iframe 嵌入 HTML   │ │
│  │  ...          │  │  │   (旅行计划完整页面)          │ │
│  └───────────────┘  │  │                            │ │
│                     │  └────────────────────────────┘ │
│  ┌───────────────┐  │                                 │
│  │ 快速开始示例    │  │                                 │
│  └───────────────┘  │                                 │
│                     │                                 │
│  ┌───────────────┐  │                                 │
│  │ 输入框 + 发送  │  │                                 │
│  └───────────────┘  │                                 │
├─────────────────────┴─────────────────────────────────┤
│  Survey 表单弹窗（首次使用时显示）                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │ 出行方式: [独自] [情侣] [家庭] [朋友]  (chip 单选)  │  │
│  │ 预算档次: [穷游] [舒适] [豪华]          (chip 单选)  │  │
│  │ 旅行节奏: [暴走打卡] [悠闲深度] [混合]   (chip 单选)  │  │
│  │ 核心兴趣: [美食] [文化] [自然] [购物] ... (chip 多选) │  │
│  │                                                   │  │
│  │              [ 开始规划旅行 ]                        │  │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

### 14.2 通信协议

```
POST /api/chat
    Request Body (JSON):
    {
        session_id: "abc123" | null,
        message: "我想去东京5天",
        language: "zh",
        currency: "CAD",
        profile: {
            travel_style: "couple",
            budget_tier: "comfort",
            pace: "mixed",
            interests: ["food", "culture"]
        }
    }

    Response: SSE (text/event-stream)
    event: progress
    data: {"pct": 50, "status": "正在生成旅行计划..."}

    event: done
    data: {"session_id": "abc123", "reply": "...", "plan_html": "<html>..." | null}
```

### 14.3 进度节点

| 节点 | 百分比 | 中文状态 | 英文状态 |
|------|--------|---------|---------|
| chat | 10% | 正在理解你的需求... | Understanding your request... |
| tool_executor | 20% | 正在查询相关信息... | Looking up information... |
| check_info | 30% | 正在分析信息完整度... | Analyzing collected info... |
| classify_profile | 40% | 正在分析你的旅行偏好... | Analyzing your travel style... |
| generate_plan | 50% | 正在生成旅行计划... | Generating travel plan... |
| enrich_plan | 75% | 正在获取图片、视频和地图数据... | Fetching photos, videos & map data... |
| 完成 | 100% | 完成！ | Done! |

---

## 15. Web 服务 app.py

### 15.1 路由

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/` | 返回 `static/index.html` |
| POST | `/api/chat` | SSE 流式对话（核心接口） |
| GET | `/api/health` | 健康检查 `{"status":"ok", "knowledge_count": N}` |

### 15.2 会话管理

```python
_sessions: dict[str, dict] = {}  # 内存存储，进程重启丢失

session_state = {
    "messages": [],              # BaseMessage 列表
    "plan_generated": False,
    "plan_data": None,           # TravelPlan
    "html_path": "",
    "language": "zh",
    "currency": "CAD",
    "conversation_summary": "",
    "travel_profile": None,      # TravelProfile
}
```

### 15.3 消息过滤 `_is_user_facing()`

以下消息不展示给用户：
- `ToolMessage`（工具返回结果）
- `AIMessage` 带 `tool_calls`（工具调用请求）
- `AIMessage` 内容为 JSON 且包含 `action` + `reason`（check_info 内部判断）
- 空内容的 `AIMessage`

### 15.4 启动流程

```python
@app.on_event("startup")
def startup():
    _seed_count = load_seed_data()   # 加载 RAG 种子知识库
    _graph = build_graph()           # 编译 LangGraph
```

---

## 16. 环境变量

**文件：** `.env`

```bash
# ── 必需 ──
OPENAI_API_KEY=sk-...              # OpenAI API（核心 LLM）

# ── Amadeus 机票/酒店 ──
AMADEUS_API_KEY=                    # 注册: developers.amadeus.com
AMADEUS_API_SECRET=
AMADEUS_BASE_URL=https://test.api.amadeus.com  # 测试环境

# ── 地图（二选一） ──
APPLE_MAPKIT_TOKEN=                 # Apple MapKit JS JWT token
GOOGLE_MAPS_API_KEY=                # Google Maps JavaScript API

# ── 媒体 ──
UNSPLASH_ACCESS_KEY=                # 景点高清图片
YOUTUBE_API_KEY=                    # 英文视频推荐
INSTAGRAM_TOKEN=                    # Instagram oEmbed（可选）

# ── 联盟营销 ──
TRAVELPAYOUTS_MARKER=               # 预订链接佣金标记
TRAVELPAYOUTS_API_TOKEN=

# ── 可选 ──
CALENDARIFIC_API_KEY=               # 节假日 API 备用
```

---

## 17. 依赖清单

### Python 包

| 包 | 用途 |
|---|------|
| `langgraph` | 状态图编排 |
| `langchain-openai` | OpenAI LLM 集成 |
| `langchain-core` | BaseMessage, ToolNode |
| `python-dotenv` | .env 加载 |
| `chromadb` | 向量数据库 |
| `fastapi` | Web 框架 |
| `uvicorn` | ASGI 服务器 |
| `pydantic` | 数据模型验证 |
| `httpx` | HTTP 客户端（Amadeus API） |
| `pyyaml` | Prompt config.yaml 解析 |
| `ddgs` | DuckDuckGo 搜索（短视频 URL 发现） |

### 启动命令

```bash
# Web 服务
uvicorn app:app --reload

# CLI 模式
python main.py
```
