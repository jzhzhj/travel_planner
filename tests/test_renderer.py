"""Tests for renderer.py — HTML output verification."""

import pytest

from models import Activity, DayPlan, HotelRecommendation, TravelPlan
from wiki import PlaceInfo
from renderer import render_plan


@pytest.fixture
def minimal_plan():
    return TravelPlan(
        title="Test Trip",
        destination="Seattle",
        departure_iata="YVR",
        destination_iata="SEA",
        start_date="2026-07-01",
        end_date="2026-07-01",
        duration="1 day",
        overview="A test trip.",
        daily_plans=[
            DayPlan(
                day=1,
                theme="Test Day",
                activities=[
                    Activity(
                        time_slot="09:00",
                        place_name="Space Needle",
                        duration_minutes=90,
                        description="Visit the tower.",
                    ),
                    Activity(
                        time_slot="12:00",
                        place_name="Pike Place Market",
                        duration_minutes=60,
                        description="Explore the market.",
                        is_restaurant=True,
                        food_recommendation="Clam chowder",
                    ),
                ],
            )
        ],
    )


@pytest.fixture
def places_with_coords():
    return {
        "Space Needle": PlaceInfo(
            name="Space Needle",
            summary="Observation tower.",
            image_url="https://example.com/needle.jpg",
            lat=47.6205,
            lon=-122.3493,
        ),
        "Pike Place Market": PlaceInfo(
            name="Pike Place Market",
            summary="Historic market.",
            image_url="https://example.com/pike.jpg",
            lat=47.6097,
            lon=-122.3422,
        ),
    }


class TestRenderPlan:
    def test_html_structure(self, minimal_plan):
        html = render_plan(minimal_plan, {})
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "</html>" in html
        assert "Space Needle" in html
        assert "Pike Place Market" in html

    def test_data_lat_lng_attributes(self, minimal_plan, places_with_coords):
        html = render_plan(minimal_plan, places_with_coords)
        assert 'data-lat="47.6205"' in html
        assert 'data-lng="-122.3493"' in html
        assert 'data-lat="47.6097"' in html

    def test_data_place_name_attribute(self, minimal_plan, places_with_coords):
        html = render_plan(minimal_plan, places_with_coords)
        assert 'data-place-name="Space Needle"' in html
        assert 'data-place-name="Pike Place Market"' in html

    def test_recalc_transit_js_present(self, minimal_plan):
        html = render_plan(minimal_plan, {})
        assert "recalcTransit" in html
        assert "window._recalcTransit" in html
        assert "estimateTransit" in html
        assert "buildTransitRow" in html

    def test_recalc_transit_in_drop_handlers(self, minimal_plan):
        html = render_plan(minimal_plan, {})
        assert "window._recalcTransit" in html
        count = html.count("_recalcTransit")
        # At least: global assignment + multiple drop handlers
        assert count >= 5

    def test_create_activity_card_has_lat_lng_params(self, minimal_plan):
        """createActivityCard should accept lat/lng params and set data attributes."""
        html = render_plan(minimal_plan, {})
        assert "function createActivityCard(name, desc, lat, lng)" in html
        assert "data-lat" in html
        assert "data-lng" in html

    def test_create_held_card_has_lat_lng_params(self, minimal_plan):
        """createHeldCard should accept and store lat/lng."""
        html = render_plan(minimal_plan, {})
        assert "function createHeldCard(name, activityEl, desc, lat, lng)" in html
        assert "hc._lat" in html
        assert "hc._lng" in html

    def test_per_zone_transit_recalc(self, minimal_plan):
        """recalcTransit should iterate per zone, not across all zones."""
        html = render_plan(minimal_plan, {})
        assert "var zones = dayCard.querySelectorAll('.activities-zone')" in html
        assert "zones.forEach" in html

    def test_restaurant_badge(self, minimal_plan):
        html = render_plan(minimal_plan, {}, language="en")
        assert "Restaurant" in html
        assert 'data-restaurant="true"' in html
        assert 'data-restaurant="false"' in html

    def test_no_old_palette_colors(self, minimal_plan, places_with_coords):
        html = render_plan(minimal_plan, places_with_coords)
        # Old aubergine/orange palette colors should not appear
        assert "#E8C9BE" not in html
        assert "rgba(200, 90, 60" not in html
        assert "rgba(200,90,60" not in html
        assert "rgba(31, 20, 25" not in html
        assert "rgba(31,20,25" not in html

    def test_language_en(self, minimal_plan):
        html = render_plan(minimal_plan, {}, language="en")
        assert "Drag" in html

    def test_language_zh(self, minimal_plan):
        html = render_plan(minimal_plan, {}, language="zh")
        assert "拖动排序" in html

    def test_empty_coords_handled(self, minimal_plan):
        """Activities without place info should have empty data-lat/lng."""
        html = render_plan(minimal_plan, {})
        assert 'data-lat=""' in html
        assert 'data-lng=""' in html

    def test_draggable_attribute(self, minimal_plan):
        html = render_plan(minimal_plan, {})
        assert 'draggable="true"' in html

    def test_haversine_js_function(self, minimal_plan):
        html = render_plan(minimal_plan, {})
        assert "haversine" in html.lower() or "Haversine" in html
