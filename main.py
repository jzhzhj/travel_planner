"""
AI Travel Planner — 命令行入口

用法: python main.py
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph import CHAT_WINDOW_SIZE, build_graph
from rag.loader import load_seed_data


def is_user_facing(msg) -> bool:
    """判断消息是否应该展示给用户。"""
    if isinstance(msg, ToolMessage):
        return False
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        return False
    if isinstance(msg, AIMessage) and not msg.content:
        return False
    return True


def main():
    # 初始化知识库
    count = load_seed_data()
    app = build_graph()

    print("=" * 60)
    print("  🌍 AI 旅行规划助手")
    print(f"  知识库已加载 {count} 条旅行攻略")
    print("  输入你的旅行想法，我来帮你规划！")
    print("  输入 'quit' 退出")
    print("=" * 60)
    print()

    state = {
        "messages": [], "plan_generated": False, "plan_data": None,
        "html_path": "", "language": "zh", "currency": "CAD",
        "user_context": None, "check_result": "continue",
        "enrich_places": None, "enrich_links": None,
        "enrich_embeds": None, "enrich_deals": None,
    }

    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\n再见！祝你旅途愉快！ 👋")
            break

        state["messages"].append(HumanMessage(content=user_input))

        result = app.invoke(state)

        # 提取新增的面向用户的消息
        new_messages = result["messages"][len(state["messages"]):]
        user_facing = [m for m in new_messages if is_user_facing(m)]

        if result.get("plan_data") is not None:
            # 计划已生成 — enrich_plan 的消息包含文件路径
            for msg in user_facing:
                print(f"\n助手: {msg.content}")
            print()
        else:
            # 普通对话
            if user_facing:
                print(f"\n助手: {user_facing[0].content}\n")

        msgs = result["messages"]
        if len(msgs) > CHAT_WINDOW_SIZE:
            msgs = msgs[-CHAT_WINDOW_SIZE:]
            while msgs and isinstance(msgs[0], ToolMessage):
                msgs = msgs[1:]

        state = {
            "messages": msgs,
            "plan_generated": result.get("plan_generated", False),
            "plan_data": result.get("plan_data"),
            "html_path": result.get("html_path", ""),
            "language": "zh",
            "currency": "CAD",
            "user_context": result.get("user_context"),
            "check_result": "continue",
            "enrich_places": result.get("enrich_places"),
            "enrich_links": result.get("enrich_links"),
            "enrich_embeds": result.get("enrich_embeds"),
            "enrich_deals": result.get("enrich_deals"),
        }


if __name__ == "__main__":
    main()
