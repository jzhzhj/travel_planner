"""Tests for models.py — Pydantic data model validation."""

import pytest
from pydantic import ValidationError

from models import (
    Activity,
    DayPlan,
    ExtractedContextPatch,
    ExtractedIntakeProfile,
    FlightRecommendation,
    HotelRecommendation,
    TravelPlan,
    UserContext,
)


class TestUserContext:
    def test_defaults(self):
        ctx = UserContext()
        assert ctx.destination is None
        assert ctx.duration_days is None
        assert ctx.interests == []
        assert ctx.wants_modification is False
        assert ctx.source == {}

    def test_partial_fill(self):
        ctx = UserContext(destination="Tokyo", duration_days=5)
        assert ctx.destination == "Tokyo"
        assert ctx.duration_days == 5
        assert ctx.travel_style is None

    def test_full_fill(self, sample_user_context):
        ctx = sample_user_context
        assert ctx.destination == "Seattle"
        assert ctx.region == "americas"
        assert "food" in ctx.interests

    def test_wants_modification_flag(self):
        ctx = UserContext(wants_modification=True)
        assert ctx.wants_modification is True


class TestActivity:
    def test_minimal(self):
        act = Activity(
            time_slot="10:00",
            place_name="Test Place",
            description="A test.",
        )
        assert act.duration_minutes == 60
        assert act.is_restaurant is False
        assert act.food_recommendation == ""

    def test_restaurant(self, sample_restaurant):
        assert sample_restaurant.is_restaurant is True
        assert sample_restaurant.food_recommendation != ""

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            Activity(time_slot="10:00")


class TestHotelRecommendation:
    def test_star_range(self):
        h = HotelRecommendation(
            name="X", area="Y", price_per_night="$100", stars=5, tier="premium"
        )
        assert h.stars == 5

    def test_star_below_range(self):
        with pytest.raises(ValidationError):
            HotelRecommendation(
                name="X", area="Y", price_per_night="$100", stars=0, tier="budget"
            )

    def test_star_above_range(self):
        with pytest.raises(ValidationError):
            HotelRecommendation(
                name="X", area="Y", price_per_night="$100", stars=6, tier="luxury"
            )


class TestDayPlan:
    def test_no_hotel(self, sample_activity):
        dp = DayPlan(day=1, theme="Day One", activities=[sample_activity])
        assert dp.hotel is None

    def test_with_hotel(self, sample_day_plan):
        assert sample_day_plan.hotel is not None
        assert sample_day_plan.hotel.name == "Hotel Max"


class TestTravelPlan:
    def test_full_plan(self, sample_travel_plan):
        plan = sample_travel_plan
        assert plan.destination == "Seattle"
        assert len(plan.daily_plans) == 1
        assert plan.daily_plans[0].day == 1
        assert len(plan.tips) == 1

    def test_empty_optional_fields(self):
        plan = TravelPlan(
            title="Minimal",
            destination="Nowhere",
            duration="0 days",
            overview="Nothing.",
            daily_plans=[],
        )
        assert plan.flight_recommendations == []
        assert plan.tips == []
        assert plan.departure_iata == ""


class TestExtractedContextPatch:
    def test_all_none_except_modification(self):
        patch = ExtractedContextPatch()
        assert patch.destination is None
        assert patch.wants_modification is False

    def test_partial_patch(self):
        patch = ExtractedContextPatch(
            destination="Paris",
            duration_days=3,
            wants_modification=True,
        )
        assert patch.destination == "Paris"
        assert patch.wants_modification is True
        assert patch.travel_style is None


class TestFlightRecommendation:
    def test_valid(self):
        f = FlightRecommendation(
            airline="Air Canada",
            route="YVR → SEA",
            price_estimate="$200 CAD",
            tier="budget",
        )
        assert f.note == ""
        assert f.airline == "Air Canada"
