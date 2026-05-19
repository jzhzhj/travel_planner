"""Tests for app.py — utility functions."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import _is_user_facing, _sse


class TestIsUserFacing:
    def test_human_message(self):
        assert _is_user_facing(HumanMessage(content="hello")) is True

    def test_tool_message(self):
        assert _is_user_facing(ToolMessage(content="result", tool_call_id="1")) is False

    def test_ai_message_with_content(self):
        assert _is_user_facing(AIMessage(content="Sure!")) is True

    def test_ai_message_empty_content(self):
        assert _is_user_facing(AIMessage(content="")) is False

    def test_ai_message_with_tool_calls(self):
        msg = AIMessage(content="", tool_calls=[{"name": "foo", "args": {}, "id": "1"}])
        assert _is_user_facing(msg) is False

    def test_ai_message_with_content_and_tool_calls(self):
        msg = AIMessage(content="thinking...", tool_calls=[{"name": "foo", "args": {}, "id": "1"}])
        assert _is_user_facing(msg) is False


class TestSuggestViaPlaces:
    def test_returns_lat_lng(self):
        from unittest.mock import patch, MagicMock
        from tools.places import PlacePOI

        mock_pois = [
            PlacePOI(
                name="id1", display_name="Test Spot", rating=4.5,
                user_ratings_total=100, lat=47.6, lng=-122.3,
                editorial_summary="Great place",
            ),
        ]
        with patch("tools.places.search_places", return_value=mock_pois):
            from app import _suggest_via_places
            results = _suggest_via_places("Seattle", "activities", set(), "en")
            assert len(results) == 1
            assert results[0]["lat"] == 47.6
            assert results[0]["lng"] == -122.3
            assert results[0]["name"] == "Test Spot"


class TestSse:
    def test_format(self):
        result = _sse("done", {"session_id": "abc", "reply": "hi"})
        assert result.startswith("event: done\n")
        assert "data: " in result
        assert result.endswith("\n\n")

        data_line = result.split("data: ")[1].strip()
        parsed = json.loads(data_line)
        assert parsed["session_id"] == "abc"
        assert parsed["reply"] == "hi"

    def test_chinese_content(self):
        result = _sse("progress", {"status": "正在生成..."})
        assert "正在生成..." in result
        data_line = result.split("data: ")[1].strip()
        parsed = json.loads(data_line)
        assert parsed["status"] == "正在生成..."

    def test_event_types(self):
        for event_name in ["done", "progress", "plan_partial", "show_survey"]:
            result = _sse(event_name, {})
            assert f"event: {event_name}\n" in result
