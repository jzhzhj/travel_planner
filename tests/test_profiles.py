"""Tests for profiles.py — strategy prompt generation."""

import pytest

from profiles import (
    BUDGET_ALLOCATION,
    PACING_RULES,
    TravelProfile,
    build_strategy_prompt,
)


class TestTravelProfile:
    def test_defaults(self):
        p = TravelProfile()
        assert p.travel_style == "friends"
        assert p.budget_tier == "comfort"
        assert p.pace == "mixed"
        assert p.interests == ["food", "culture"]


class TestBuildStrategyPrompt:
    def test_chinese_output(self):
        profile = TravelProfile(travel_style="solo", budget_tier="budget", pace="intensive")
        result = build_strategy_prompt(profile, language="zh")
        assert "独自旅行" in result
        assert "穷游" in result
        assert "暴走打卡型" in result

    def test_english_output(self):
        profile = TravelProfile(travel_style="couple", budget_tier="luxury", pace="relaxed")
        result = build_strategy_prompt(profile, language="en")
        assert "Couple / Honeymoon" in result
        assert "Luxury" in result
        assert "Relaxed" in result

    def test_family_rules_included(self):
        profile = TravelProfile(travel_style="family")
        result_zh = build_strategy_prompt(profile, language="zh")
        assert "亲子特殊规则" in result_zh

        result_en = build_strategy_prompt(profile, language="en")
        assert "Family Rules" in result_en

    def test_no_family_rules_for_solo(self):
        profile = TravelProfile(travel_style="solo")
        result = build_strategy_prompt(profile, language="en")
        assert "Family Rules" not in result

    def test_pacing_max_activities(self):
        for pace_key, rules in PACING_RULES.items():
            profile = TravelProfile(pace=pace_key)
            result = build_strategy_prompt(profile, language="en")
            assert f"Max {rules['max_activities']}" in result

    def test_interest_rules_injected(self):
        profile = TravelProfile(interests=["food", "nature"])
        result = build_strategy_prompt(profile, language="zh")
        assert "美食" in result
        assert "自然" in result

    def test_anti_patterns_present(self):
        profile = TravelProfile(travel_style="family", budget_tier="budget", pace="relaxed")
        result = build_strategy_prompt(profile, language="zh")
        assert "避免以下问题" in result

    def test_budget_allocation_region(self):
        profile = TravelProfile(budget_tier="budget", region="asia")
        result = build_strategy_prompt(profile, language="en")
        assert "Budget Allocation" in result

    def test_accommodation_preference(self):
        profile = TravelProfile(travel_style="couple", budget_tier="luxury")
        result = build_strategy_prompt(profile, language="en")
        assert "Accommodation Preference" in result
        assert "5-star" in result or "luxury" in result.lower()
