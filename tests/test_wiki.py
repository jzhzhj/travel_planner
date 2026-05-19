"""Tests for wiki.py — place info fetching with mocked HTTP."""

from unittest.mock import MagicMock, patch

import pytest

from wiki import PlaceInfo, _geocode_nominatim, _try_fetch, fetch_place_info


class TestTryFetch:
    @patch("wiki.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "extract": "A famous temple.",
            "thumbnail": {"source": "https://example.com/thumb.jpg"},
            "coordinates": {"lat": 35.0, "lon": 139.0},
        }
        mock_get.return_value = mock_resp

        info = _try_fetch("https://en.wikipedia.org/api/rest_v1", "Test Temple")
        assert info is not None
        assert info.summary == "A famous temple."
        assert info.image_url == "https://example.com/thumb.jpg"
        assert info.lat == 35.0
        assert info.lon == 139.0

    @patch("wiki.requests.get")
    def test_404(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        info = _try_fetch("https://en.wikipedia.org/api/rest_v1", "Nonexistent")
        assert info is None

    @patch("wiki.requests.get")
    def test_network_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("timeout")
        info = _try_fetch("https://en.wikipedia.org/api/rest_v1", "Test")
        assert info is None

    @patch("wiki.requests.get")
    def test_no_thumbnail_uses_originalimage(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "extract": "Summary",
            "originalimage": {"source": "https://example.com/original.jpg"},
        }
        mock_get.return_value = mock_resp

        info = _try_fetch("https://en.wikipedia.org/api/rest_v1", "Test")
        assert info.image_url == "https://example.com/original.jpg"


class TestGeocodeNominatim:
    @patch("wiki.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"lat": "47.6", "lon": "-122.3"}]
        mock_get.return_value = mock_resp

        result = _geocode_nominatim("Space Needle")
        assert result is not None
        assert result[0] == pytest.approx(47.6)
        assert result[1] == pytest.approx(-122.3)

    @patch("wiki.requests.get")
    def test_no_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        result = _geocode_nominatim("xyznonexistent")
        assert result is None

    @patch("wiki.requests.get")
    def test_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("fail")
        result = _geocode_nominatim("Test")
        assert result is None


class TestFetchPlaceInfo:
    @patch("wiki._search_google_places_photo", return_value=(None, None, None))
    @patch("wiki._search_unsplash", return_value=None)
    @patch("wiki._geocode_nominatim", return_value=None)
    @patch("wiki._try_fetch")
    def test_all_sources_fail(self, mock_fetch, mock_geo, mock_unsplash, mock_gp):
        mock_fetch.return_value = None
        info = fetch_place_info("Nonexistent Place")
        assert info.name == "Nonexistent Place"
        assert info.summary == ""
        assert info.image_url is None

    @patch("wiki._search_google_places_photo", return_value=("https://gp.photo", 35.0, 139.0))
    @patch("wiki._try_fetch")
    def test_google_places_photo_priority(self, mock_fetch, mock_gp):
        mock_fetch.return_value = PlaceInfo(
            name="Temple", summary="A temple.", image_url="https://wiki.img",
            lat=35.0, lon=139.0,
        )
        info = fetch_place_info("Temple")
        assert info.image_url == "https://gp.photo"
        assert info.lat == 35.0

    @patch("wiki._search_google_places_photo", return_value=(None, None, None))
    @patch("wiki._search_unsplash", return_value="https://unsplash.img")
    @patch("wiki._try_fetch")
    def test_unsplash_fallback(self, mock_fetch, mock_unsplash, mock_gp):
        mock_fetch.return_value = PlaceInfo(
            name="Temple", summary="A temple.", image_url=None,
            lat=35.0, lon=139.0,
        )
        info = fetch_place_info("Temple")
        assert info.image_url == "https://unsplash.img"

    @patch("wiki._search_google_places_photo", return_value=(None, None, None))
    @patch("wiki._search_unsplash", return_value=None)
    @patch("wiki._geocode_nominatim", return_value=(47.6, -122.3))
    @patch("wiki._try_fetch")
    def test_nominatim_geocode_fallback(self, mock_fetch, mock_geo, mock_unsplash, mock_gp):
        mock_fetch.return_value = PlaceInfo(
            name="Place", summary="Desc", image_url=None,
        )
        info = fetch_place_info("Place")
        assert info.lat == 47.6
        assert info.lon == -122.3

    def test_language_api_order(self):
        """Verify that language='en' tries English Wikipedia first."""
        with patch("wiki._try_fetch") as mock_fetch, \
             patch("wiki._search_google_places_photo", return_value=(None, None, None)), \
             patch("wiki._search_unsplash", return_value=None), \
             patch("wiki._geocode_nominatim", return_value=None):
            mock_fetch.return_value = PlaceInfo(name="X", summary="ok", image_url=None)
            fetch_place_info("X", language="en")
            first_call_url = mock_fetch.call_args_list[0][0][0]
            assert "en.wikipedia" in first_call_url
