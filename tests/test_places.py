"""Tests for tools/places.py — Google Places API client."""

from unittest.mock import patch, MagicMock

import pytest

from tools.places import PlacePOI, _parse_place, format_pois_for_prompt, search_places


SAMPLE_PLACE_RESPONSE = {
    "id": "ChIJ12345",
    "displayName": {"text": "Tokyo Tower"},
    "rating": 4.5,
    "userRatingCount": 12000,
    "location": {"latitude": 35.6586, "longitude": 139.7454},
    "formattedAddress": "4-2-8 Shibakoen, Minato City",
    "types": ["tourist_attraction", "point_of_interest"],
    "priceLevel": "PRICE_LEVEL_MODERATE",
    "photos": [{"name": "places/ChIJ12345/photos/abc123"}],
    "editorialSummary": {"text": "Iconic Tokyo landmark."},
}


class TestParsePlace:
    def test_full_data(self):
        poi = _parse_place(SAMPLE_PLACE_RESPONSE)
        assert poi.display_name == "Tokyo Tower"
        assert poi.rating == 4.5
        assert poi.user_ratings_total == 12000
        assert poi.lat == 35.6586
        assert poi.lng == 139.7454
        assert poi.address == "4-2-8 Shibakoen, Minato City"
        assert "tourist_attraction" in poi.types
        assert poi.photo_name == "places/ChIJ12345/photos/abc123"
        assert poi.editorial_summary == "Iconic Tokyo landmark."

    def test_minimal_data(self):
        poi = _parse_place({"displayName": {"text": "Unknown"}, "id": "x"})
        assert poi.display_name == "Unknown"
        assert poi.rating == 0.0
        assert poi.lat == 0.0
        assert poi.photo_name == ""
        assert poi.editorial_summary == ""

    def test_no_photos(self):
        data = {**SAMPLE_PLACE_RESPONSE, "photos": []}
        poi = _parse_place(data)
        assert poi.photo_name == ""


class TestFormatPoisForPrompt:
    def test_empty(self):
        result = format_pois_for_prompt({"attractions": [], "restaurants": []})
        assert result == ""

    def test_attractions_only_en(self):
        pois = {
            "attractions": [
                PlacePOI(name="id1", display_name="Tower", rating=4.5,
                         user_ratings_total=100, editorial_summary="Great view"),
            ],
            "restaurants": [],
        }
        result = format_pois_for_prompt(pois, language="en")
        assert "Top Attractions" in result
        assert "Tower" in result
        assert "4.5/5" in result
        assert "Great view" in result

    def test_restaurants_zh(self):
        pois = {
            "attractions": [],
            "restaurants": [
                PlacePOI(name="id2", display_name="Sushi Dai", rating=4.8,
                         user_ratings_total=500, price_level="MODERATE"),
            ],
        }
        result = format_pois_for_prompt(pois, language="zh")
        assert "热门餐厅" in result
        assert "Sushi Dai" in result

    def test_both_categories(self):
        pois = {
            "attractions": [PlacePOI(name="a", display_name="A")],
            "restaurants": [PlacePOI(name="b", display_name="B")],
        }
        result = format_pois_for_prompt(pois, language="en")
        assert "Attractions" in result
        assert "Restaurants" in result


class TestSearchDestinationPois:
    @patch.dict("os.environ", {"GOOGLE_MAPS_API_KEY": ""})
    def test_no_api_key(self):
        from tools.places import search_destination_pois
        result = search_destination_pois("Tokyo")
        assert result == {"attractions": [], "restaurants": []}

    @patch("tools.places.search_places")
    @patch.dict("os.environ", {"GOOGLE_MAPS_API_KEY": "fake-key"})
    def test_restaurant_deduplication(self, mock_search):
        from tools.places import search_destination_pois
        # Simulate overlapping results from multiple queries
        poi_a = PlacePOI(name="a", display_name="Sushi Place", rating=4.5, user_ratings_total=100)
        poi_b = PlacePOI(name="b", display_name="Ramen Shop", rating=4.8, user_ratings_total=200)
        poi_a_dup = PlacePOI(name="a2", display_name="Sushi Place", rating=4.5, user_ratings_total=100)

        def side_effect(query, **kwargs):
            if "attraction" in query.lower():
                return [PlacePOI(name="t", display_name="Tower", rating=4.0)]
            elif "breakfast" in query.lower():
                return [poi_a]
            elif "dinner" in query.lower():
                return [poi_a_dup, poi_b]
            else:
                return [poi_a, poi_b]

        mock_search.side_effect = side_effect
        result = search_destination_pois("Tokyo", language="en")
        rest_names = [r.display_name for r in result["restaurants"]]
        # Should deduplicate — "Sushi Place" only once
        assert rest_names.count("Sushi Place") == 1
        assert "Ramen Shop" in rest_names
        # Should be sorted by rating (Ramen 4.8 first)
        assert result["restaurants"][0].display_name == "Ramen Shop"


class TestSearchPlaces:
    @patch.dict("os.environ", {"GOOGLE_MAPS_API_KEY": ""})
    def test_no_api_key_returns_empty(self):
        result = search_places("Tokyo attractions")
        assert result == []

    @patch("tools.places.httpx.post")
    @patch.dict("os.environ", {"GOOGLE_MAPS_API_KEY": "fake-key"})
    def test_successful_search(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"places": [SAMPLE_PLACE_RESPONSE]}
        mock_post.return_value = mock_resp

        results = search_places("Tokyo attractions", max_results=5)
        assert len(results) == 1
        assert results[0].display_name == "Tokyo Tower"
        mock_post.assert_called_once()

    @patch("tools.places.httpx.post")
    @patch.dict("os.environ", {"GOOGLE_MAPS_API_KEY": "fake-key"})
    def test_api_error_returns_empty(self, mock_post):
        mock_post.side_effect = Exception("Connection error")
        results = search_places("test query")
        assert results == []

    @patch("tools.places.httpx.post")
    @patch.dict("os.environ", {"GOOGLE_MAPS_API_KEY": "fake-key"})
    def test_max_results_capped_at_20(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"places": []}
        mock_post.return_value = mock_resp

        search_places("test", max_results=50)
        call_body = mock_post.call_args[1]["json"]
        assert call_body["maxResultCount"] == 20
