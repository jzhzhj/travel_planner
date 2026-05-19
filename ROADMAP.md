# AI Travel Planner — 优化路线图

基于代码审计的完整优化计划，按优先级和依赖关系排序。每个阶段的任务可独立完成。

---

## 阶段 0：紧急修复（立即）

> 当前代码中存在会导致崩溃或数据损坏的问题，必须先修。

### 0.1 generate_plan 结构化输出崩溃保护
- **问题：** `graph.py` 中 `generate_plan()` 直接调用 `get_plan_llm().with_structured_output(TravelPlan).invoke(messages)`，如果 LLM 返回的 JSON 不符合 TravelPlan schema，整个 graph 崩溃，前端只能收到一条泛泛的错误信息
- **修复：** 加 try/except + 重试逻辑（最多 2 次），失败后返回用户可读的错误提示而不是崩溃
- **文件：** `graph.py` → `generate_plan()`

### 0.2 并发会话竞态条件
- **问题：** `app.py` 的 `_sessions` 字典在多个请求同时操作同一 session_id 时会出现数据竞争（`state["messages"].append()` 和 `_sessions[session_id] = {...}` 没有锁）
- **修复：** 给每个 session 加 `threading.Lock()`，或改用 `asyncio.Lock()`
- **文件：** `app.py`

### 0.3 HTML 文件名冲突
- **问题：** 所有用户的计划都写入 `travel_plan.html`，多用户同时使用会互相覆盖
- **修复：** 文件名加 session_id，如 `travel_plan_{session_id}.html`
- **文件：** `graph.py` → `enrich_plan()`，`app.py` → HTML 读取

### 0.4 Amadeus token 刷新失败后空授权
- **问题：** `travel_api.py` 中如果 token 刷新请求失败，`_get_access_token()` 返回空字符串，后续 API 调用带空 `Authorization` header 会得到 401 但没有有意义的处理
- **修复：** 刷新失败时抛异常或返回 None，调用方检查后跳过 API 请求
- **文件：** `tools/travel_api.py`

---

## 阶段 1：性能优化（1-2 周）

> `enrich_plan` 节点是整个流程的瓶颈，当前 10+ 个景点的计划需要 2-3 分钟。

### 1.1 并行化 enrich_plan 中的 API 调用
- **现状：** Wikipedia、视频、短视频、机票、酒店全部串行调用
- **方案：** 用 `concurrent.futures.ThreadPoolExecutor` 并行化
  ```
  现在:  wiki(A) → wiki(B) → wiki(C) → videos → embeds → flights → hotels
  优化后: [wiki(A), wiki(B), wiki(C)] 并行 → [videos, embeds] 并行 → [flights, hotels] 并行
  ```
- **预期效果：** 从 2-3 分钟降到 30-60 秒
- **文件：** `graph.py` → `enrich_plan()`，`wiki.py` → 新增 `fetch_places_concurrent()`

### 1.2 添加结果缓存层
- **现状：** 每次请求都重新调 Wikipedia/天气/节假日 API，无缓存
- **方案：** 加一个简单的 TTL 缓存（`functools.lru_cache` 或用 `cachetools.TTLCache`）
  - Wikipedia 景点信息：缓存 24 小时
  - 天气数据：缓存 3 小时
  - 节假日数据：缓存 7 天
  - 汇率：缓存 1 小时
- **文件：** `wiki.py`、`tools/weather.py`、`tools/holidays.py`、`tools/currency.py`

### 1.3 SSE 流式优化
- **现状：** `stream_mode="values"` 每次返回完整 state 快照，消息历史长了以后序列化开销大
- **方案：** 改用 `stream_mode="updates"` 只传增量，减少网络和序列化开销
- **文件：** `app.py` → `generate_sse()`

---

## 阶段 2：稳定性 & 错误处理（1-2 周）

> 让所有外部 API 失败都有优雅降级，用户始终能得到结果。

### 2.1 全链路错误处理
- **每个外部调用加 try/except + 降级策略：**

  | API | 失败后降级 |
  |-----|----------|
  | Amadeus 机票 | 只显示 LLM 推荐的 3 档机票，不显示实时价格 |
  | Amadeus 酒店 | 只显示 LLM 推荐的酒店，不显示实时价格 |
  | Wikipedia | 只显示景点名称，无摘要/图片 |
  | Unsplash | 回退到 Wikipedia 图片，再失败则无图 |
  | YouTube/Bilibili | 视频推荐区域隐藏 |
  | TikTok/Instagram | 短视频区域隐藏 |
  | 天气 API | 返回"暂无天气数据" |

- **文件：** `graph.py`、`wiki.py`、`tools/` 下所有文件、`renderer.py`

### 2.2 LLM 调用重试与回退
- 对 `check_info`、`classify_profile`、`generate_plan` 的 LLM 调用加重试（最多 2 次）
- JSON 解析失败时的兜底：`check_info` 默认 continue_chat，`classify_profile` 默认 profile
- `generate_plan` 如果结构化输出两次都失败，尝试普通 text 输出 + 手动解析
- **文件：** `graph.py`

### 2.3 日志系统
- **现状：** 全靠 `print()` 和 `traceback.print_exc()`
- **方案：** 用标准 `logging` 模块，分级别（DEBUG/INFO/WARNING/ERROR）
- 关键记录点：
  - 每个 LLM 调用的 token 用量和耗时
  - 每个外部 API 调用的耗时和状态码
  - prompt_manager 的版本使用记录
  - 会话创建/销毁
- **文件：** 所有 Python 文件

---

## 阶段 3：RAG 知识库增强（2-3 周）

> 当前种子数据仅覆盖日本，但项目定位是全球旅行规划。

### 3.1 扩展种子数据覆盖
- **当前：** ~100 条，全是日本（东京、京都）
- **目标：** 至少覆盖 TOP 20 热门目的地（每个 10-20 条）：
  - 亚洲：东京、曼谷、首尔、新加坡、巴厘岛
  - 欧洲：巴黎、罗马、巴塞罗那、伦敦、阿姆斯特丹
  - 美洲：纽约、洛杉矶、墨西哥城、布宜诺斯艾利斯、温哥华
  - 大洋洲：悉尼、奥克兰
  - 中东/非洲：迪拜、开罗、开普敦
- **数据类型：** 景点、餐厅、交通贴士、文化礼仪、省钱攻略、季节建议
- **文件：** `rag/seed_knowledge.json`（或拆分为多个 JSON）

### 3.2 添加 Reranker
- **现状：** ChromaDB 余弦相似度直接返回 top-N，无重排序
- **方案：** 检索 top-10 → Cross-encoder 重排序 → 取 top-3
- 推荐模型：`cross-encoder/ms-marco-MiniLM-L-6-v2`（轻量，<100ms）
- **文件：** `rag/retriever.py`、新增 `rag/reranker.py`

### 3.3 Embedding 模型升级
- **现状：** ChromaDB `DefaultEmbeddingFunction()`（通用 all-MiniLM-L6-v2）
- **方案：** 换成更适合多语言旅行场景的模型
  - 选项 A：`BAAI/bge-small-zh-v1.5`（中文优化）
  - 选项 B：`intfloat/multilingual-e5-small`（多语言）
- **文件：** `rag/store.py`

### 3.4 用户反馈回写 RAG
- 用户对计划的修改意见可以作为新知识写回 RAG
- 例如用户说"浅草寺不需要那么久，1小时就够了" → 更新浅草寺的建议时长
- **文件：** `rag/loader.py`（新增 `add_user_feedback()`），`graph.py`

---

## 阶段 4：会话管理 & 持久化（1-2 周）

> 解决重启丢数据和内存泄漏问题。

### 4.1 Session 持久化
- **现状：** 纯内存 dict，重启全丢
- **方案（从简到繁选一个）：**
  - Level 1：SQLite + JSON 序列化（最简单，单机够用）
  - Level 2：Redis（支持多实例部署 + 自动过期）
  - Level 3：PostgreSQL（需要查询历史计划时）
- **需要序列化的字段：** messages（LangChain BaseMessage）、plan_data（Pydantic）、travel_profile
- **文件：** 新增 `session_store.py`，修改 `app.py`

### 4.2 Session 过期清理
- 添加 TTL：超过 24 小时未活动的会话自动清理
- 限制最大会话数（如 1000）
- **文件：** `app.py` 或 `session_store.py`

### 4.3 前端 localStorage
- 保存 session_id 到 localStorage，刷新页面后恢复对话
- 保存 survey 表单选择，不用每次重填
- 保存最近的计划 HTML，离线也能查看
- **文件：** `static/index.html`

---

## 阶段 5：前端体验升级（2-3 周）

### 5.1 错误状态 UI
- API 调用失败时显示具体错误（而不是泛泛的"出错了"）
- 添加重试按钮
- 网络断开时显示离线提示
- **文件：** `static/index.html`

### 5.2 移动端适配
- **现状：** 有 900px 断点但不够用
- 移动端改为单栏布局（聊天和预览切换显示）
- 按钮和输入框适配触控
- Survey 表单卡片在小屏幕自动换行
- **文件：** `static/index.html`

### 5.3 计划交互增强
- 在 iframe 内的计划 HTML 中添加"修改这天"按钮
- 点击后自动在聊天框填入修改请求
- 活动卡片可拖拽排序（高级）
- **文件：** `renderer.py`、`static/index.html`

### 5.4 导出功能
- PDF 导出（用 `weasyprint` 或浏览器 `window.print()` 优化）
- 日历导出（生成 .ics 文件，可导入 Google Calendar / Apple Calendar）
- 分享链接（生成唯一 URL）
- **文件：** 新增 `export.py`，`renderer.py`，`app.py` 新增路由

---

## 阶段 6：测试 & CI/CD（2 周）

### 6.1 单元测试
- **优先级高的测试目标：**
  - `profiles.py`：`build_strategy_prompt()` 对每种画像组合的输出
  - `prompts/manager.py`：版本加载、变量渲染、缓存
  - `models.py`：TravelPlan schema 验证
  - `rag/retriever.py`：检索 + 元数据过滤
  - `graph.py`：路由逻辑 `route_after_chat`、`route_after_check`
  - `tools/travel_api.py`：token 缓存、API 响应解析
- **Mock 策略：** 所有外部 API 用 `unittest.mock.patch` 或 `responses` 库 mock
- **文件：** 新建 `tests/` 目录

### 6.2 集成测试
- 测试完整 graph 流程（mock LLM 返回固定回复）
- 测试 SSE 端点（FastAPI TestClient）
- 测试 prompt 加载 + 变量渲染端到端
- **文件：** `tests/test_graph.py`、`tests/test_api.py`

### 6.3 Dockerfile
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
- **文件：** 新增 `Dockerfile`、`.dockerignore`

### 6.4 GitHub Actions CI
- PR 时自动运行：
  - `pytest`（单元测试）
  - `ruff check`（lint）
  - `mypy`（类型检查）
  - 语法检查（`python -c "import ast; ast.parse()"` 所有 .py）
- **文件：** 新增 `.github/workflows/ci.yml`

---

## 阶段 7：LLM 调用优化（1-2 周）

### 7.1 Token 用量监控
- 在每次 LLM 调用后记录 `response.usage_metadata`（prompt_tokens, completion_tokens）
- 汇总到 session 级别，前端可展示（可选）
- 用于分析哪个节点最费 token，指导优化
- **文件：** `graph.py` 所有 LLM 调用点

### 7.2 动态模型选择
- **现状：** 所有对话用 gpt-4o-mini，所有计划用 gpt-4.1，硬编码
- **方案：** 根据用户画像或请求复杂度动态选模型
  - 简单问候/闲聊 → gpt-4o-mini
  - 复杂多城市计划 → gpt-4.1
  - 计划修改（小改动）→ gpt-4o-mini
- **文件：** `graph.py`、`prompts/config.yaml`（可加 model 配置）

### 7.3 Prompt 压缩
- `plan_system` 是最长的 prompt（~800 tokens），加上策略注入可达 1500+ tokens
- 考虑把固定部分缓存为 system message prefix（如果 OpenAI 支持 prompt caching）
- 或者根据画像只注入相关的策略段落，而非全部
- **文件：** `profiles.py` → `build_strategy_prompt()`

---

## 阶段 8：功能扩展（长期）

### 8.1 多城市行程支持
- **现状：** `TravelPlan` 只有一个 `destination`，不支持 A→B→C
- **方案：** 添加 `cities: list[CityStop]`，每个 CityStop 包含 city、days、transport_to_next
- 路线优化：根据地理位置排序城市
- **文件：** `models.py`、`graph.py`、`renderer.py`、`prompts/plan_system/`

### 8.2 计划编辑与版本对比
- 用户说"把第二天的寺庙换成购物"→ 只重新生成第二天，不重做整个计划
- 保存计划的多个版本，可对比差异
- **文件：** `graph.py`（新增 `modify_plan` 节点）、`app.py`

### 8.3 用户账户系统
- 注册/登录（OAuth：Google、WeChat）
- 持久化旅行偏好（不用每次填表单）
- 历史计划存档
- **文件：** 新增 `auth.py`、数据库 schema

### 8.4 协作规划
- 多人同时编辑同一个计划
- 投票选景点
- 费用 AA 计算
- **文件：** WebSocket 支持、新增协作模块

### 8.5 智能推荐引擎
- 根据用户历史计划推荐目的地
- 根据季节、预算、当前热度推荐
- "和你相似的用户还去了..."
- **文件：** 新增 `recommendation.py`

---

## 进度追踪

| 阶段 | 任务 | 状态 | 备注 |
|------|------|------|------|
| 0.1 | generate_plan 崩溃保护 | ⬜ 待开始 | |
| 0.2 | 会话竞态条件修复 | ⬜ 待开始 | |
| 0.3 | HTML 文件名冲突 | ⬜ 待开始 | |
| 0.4 | Amadeus token 刷新修复 | ⬜ 待开始 | |
| 1.1 | enrich_plan 并行化 | ⬜ 待开始 | |
| 1.2 | 结果缓存层 | ⬜ 待开始 | |
| 1.3 | SSE 流式优化 | ⬜ 待开始 | |
| 2.1 | 全链路错误处理 | ⬜ 待开始 | |
| 2.2 | LLM 调用重试与回退 | ⬜ 待开始 | |
| 2.3 | 日志系统 | ⬜ 待开始 | |
| 3.1 | 扩展种子数据 | ⬜ 待开始 | |
| 3.2 | 添加 Reranker | ⬜ 待开始 | |
| 3.3 | Embedding 升级 | ⬜ 待开始 | |
| 3.4 | 用户反馈回写 RAG | ⬜ 待开始 | |
| 4.1 | Session 持久化 | ⬜ 待开始 | |
| 4.2 | Session 过期清理 | ⬜ 待开始 | |
| 4.3 | 前端 localStorage | ⬜ 待开始 | |
| 5.1 | 错误状态 UI | ⬜ 待开始 | |
| 5.2 | 移动端适配 | ⬜ 待开始 | |
| 5.3 | 计划交互增强 | ⬜ 待开始 | |
| 5.4 | 导出功能 | ⬜ 待开始 | |
| 6.1 | 单元测试 | ⬜ 待开始 | |
| 6.2 | 集成测试 | ⬜ 待开始 | |
| 6.3 | Dockerfile | ⬜ 待开始 | |
| 6.4 | GitHub Actions CI | ⬜ 待开始 | |
| 7.1 | Token 用量监控 | ⬜ 待开始 | |
| 7.2 | 动态模型选择 | ⬜ 待开始 | |
| 7.3 | Prompt 压缩 | ⬜ 待开始 | |
| 8.1 | 多城市行程 | ⬜ 待开始 | |
| 8.2 | 计划编辑与版本对比 | ⬜ 待开始 | |
| 8.3 | 用户账户系统 | ⬜ 待开始 | |
| 8.4 | 协作规划 | ⬜ 待开始 | |
| 8.5 | 智能推荐引擎 | ⬜ 待开始 | |
