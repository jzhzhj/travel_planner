"""Tests for tools/directions.py — transit time calculations."""

import math
from unittest.mock import patch

import pytest

from tools.directions import (
    TransitInfo,
    _estimate_transit,
    _format_distance,
    _format_duration,
    _haversine_meters,
    _parse_route,
    get_transit_between,
)


class TestFormatDuration:
    def test_minutes_only(self):
        assert _format_duration(300) == "5 mins"
        assert _format_duration(0) == "0 mins"
        assert _format_duration(59 * 60) == "59 mins"

    def test_hours_and_minutes(self):
        assert _format_duration(3900) == "1 hour 5 mins"  # 65 mins
        assert _format_duration(7200) == "2 hours"         # 120 mins

    def test_multiple_hours(self):
        assert _format_duration(10800) == "3 hours"  # 180 mins


class TestFormatDistance:
    def test_meters(self):
        assert _format_distance(500) == "500 m"
        assert _format_distance(999) == "999 m"

    def test_kilometers(self):
        assert _format_distance(1000) == "1.0 km"
        assert _format_distance(3200) == "3.2 km"
        assert _format_distance(15750) == "15.8 km"


class TestHaversine:
    def test_same_point(self):
        dist = _haversine_meters((35.6762, 139.6503), (35.6762, 139.6503))
        assert dist == pytest.approx(0, abs=1)

    def test_known_distance(self):
        # Tokyo to Osaka: ~400 km
        dist = _haversine_meters((35.6762, 139.6503), (34.6937, 135.5023))
        assert 390_000 < dist < 410_000

    def test_short_distance(self):
        # ~1.5 km apart in central Tokyo
        dist = _haversine_meters((35.6812, 139.7671), (35.6896, 139.6917))
        assert 5_000 < dist < 8_000


class TestParseRoute:
    def test_normal(self):
        route = {"duration": "300s", "distanceMeters": 950}
        secs, meters = _parse_route(route)
        assert secs == 300
        assert meters == 950

    def test_missing_fields(self):
        secs, meters = _parse_route({})
        assert secs == 0
        assert meters == 0


class TestEstimateTransit:
    def test_walking_short_distance(self):
        # Two points ~500m apart
        info = _estimate_transit(
            "A", "B",
            (35.6812, 139.7671), (35.6850, 139.7671),
        )
        assert info.mode == "walking"
        assert info.duration_mins < 25

    def test_driving_long_distance(self):
        # Two points ~10km apart
        info = _estimate_transit(
            "A", "B",
            (35.6762, 139.6503), (35.7100, 139.8107),
        )
        assert info.mode == "driving"
        assert info.duration_mins > 0

    def test_output_fields(self):
        info = _estimate_transit(
            "Place A", "Place B",
            (35.6762, 139.6503), (35.6800, 139.6550),
        )
        assert info.origin == "Place A"
        assert info.destination == "Place B"
        assert isinstance(info.duration_text, str)
        assert isinstance(info.distance_text, str)


class TestGetTransitBetween:
    @patch.dict("os.environ", {"GOOGLE_MAPS_API_KEY": ""})
    def test_no_api_key_uses_estimate(self):
        info = get_transit_between(
            "A", "B",
            (35.6812, 139.7671), (35.6850, 139.7671),
        )
        assert info is not None
        assert info.mode in ("walking", "driving")

    def test_zero_coords_returns_none(self):
        info = get_transit_between("A", "B", (0.0, 0.0), (35.0, 139.0))
        assert info is None
