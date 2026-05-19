"""Shared fixtures for tests."""

import pytest

from models import Activity, DayPlan, HotelRecommendation, TravelPlan, UserContext
from wiki import PlaceInfo


@pytest.fixture
def sample_activity():
    return Activity(
        time_slot="09:00",
        place_name="Space Needle",
        duration_minutes=90,
        description="Visit Seattle's iconic observation tower.",
        is_restaurant=False,
    )


@pytest.fixture
def sample_restaurant():
    return Activity(
        time_slot="12:00",
        place_name="Pike Place Chowder",
        duration_minutes=60,
        description="Famous clam chowder at Pike Place Market.",
        food_recommendation="New England Clam Chowder",
        is_restaurant=True,
    )


@pytest.fixture
def sample_hotel():
    return HotelRecommendation(
        name="Hotel Max",
        area="Downtown Seattle",
        price_per_night="$150 CAD",
        stars=3,
        highlight="Walking distance to Pike Place",
        tier="mid-range",
    )


@pytest.fixture
def sample_day_plan(sample_activity, sample_restaurant, sample_hotel):
    return DayPlan(
        day=1,
        theme="Seattle Highlights",
        activities=[sample_activity, sample_restaurant],
        hotel=sample_hotel,
    )


@pytest.fixture
def sample_travel_plan(sample_day_plan):
    return TravelPlan(
        title="Seattle Adventure",
        destination="Seattle",
        departure_iata="YVR",
        destination_iata="SEA",
        start_date="2026-07-01",
        end_date="2026-07-01",
        duration="1 day",
        overview="A day trip to Seattle from Vancouver.",
        daily_plans=[sample_day_plan],
        tips=["Bring an umbrella"],
        budget_summary="| Item | Cost |\n|---|---|\n| Total | $300 |",
    )


@pytest.fixture
def sample_place_info():
    return PlaceInfo(
        name="Space Needle",
        summary="An observation tower in Seattle.",
        image_url="https://example.com/space_needle.jpg",
        lat=47.6205,
        lon=-122.3493,
    )


@pytest.fixture
def sample_user_context():
    return UserContext(
        destination="Seattle",
        duration_days=1,
        departure_city="Vancouver",
        travel_style="solo",
        budget_tier="comfort",
        pace="mixed",
        interests=["food", "culture"],
        region="americas",
    )
