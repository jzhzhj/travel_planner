"""Google Places API (New) — search real attractions & restaurants for a destination."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("travel_planner")

# Google Places API (New) endpoints
_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
_NEARBY_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"
_PHOTO_BASE = "https://places.googleapis.com/v1"


def _api_key() -> str:
    return os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()


@dataclass
class PlacePOI:
    """A real point-of-interest from Google Places."""
    name: str
    display_name: str
    rating: float = 0.0
    user_ratings_total: int = 0
    lat: float = 0.0
    lng: float = 0.0
    address: str = ""
    types: list[str] = field(default_factory=list)
    price_level: str = ""
    photo_name: str = ""  # resource name for Places photo API
    editorial_summary: str = ""


def _parse_place(place: dict) -> PlacePOI:
    """Parse a place object from the Places API (New) response."""
    loc = place.get("location", {})
    display = place.get("displayName", {}).get("text", "")
    photos = place.get("photos", [])
    photo_name = photos[0].get("name", "") if photos else ""
    summary = place.get("editorialSummary", {}).get("text", "")

    return PlacePOI(
        name=place.get("id", ""),
        display_name=display,
        rating=place.get("rating", 0.0),
        user_ratings_total=place.get("userRatingCount", 0),
        lat=loc.get("latitude", 0.0),
        lng=loc.get("longitude", 0.0),
        address=place.get("formattedAddress", ""),
        types=place.get("types", []),
        price_level=place.get("priceLevel", ""),
        photo_name=photo_name,
        editorial_summary=summary,
    )


def search_places(
    query: str,
    *,
    max_results: int = 10,
    language: str = "en",
) -> list[PlacePOI]:
    """Text Search for places using Google Places API (New).

    Args:
        query: e.g. "top attractions in Tokyo", "best restaurants in Paris"
        max_results: max number of results (up to 20)
        language: language code for results
    """
    key = _api_key()
    if not key:
        log.warning("[places] no GOOGLE_MAPS_API_KEY, skipping Places search")
        return []

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.rating,places.userRatingCount,"
            "places.location,places.formattedAddress,places.types,"
            "places.priceLevel,places.photos,places.editorialSummary"
        ),
    }
    body = {
        "textQuery": query,
        "maxResultCount": min(max_results, 20),
        "languageCode": language,
    }

    try:
        resp = httpx.post(_TEXT_SEARCH_URL, json=body, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        places = data.get("places", [])
        results = [_parse_place(p) for p in places]
        log.info(f"[places] query={query!r} → {len(results)} results")
        return results
    except Exception as e:
        log.warning(f"[places] search error: {type(e).__name__}: {e}")
        return []


def get_photo_url(photo_name: str, max_width: int = 800) -> str | None:
    """Get a photo URL from Google Places photo resource name.

    Args:
        photo_name: e.g. "places/xxx/photos/yyy"
        max_width: max width in pixels
    """
    key = _api_key()
    if not key or not photo_name:
        return None

    url = f"{_PHOTO_BASE}/{photo_name}/media?maxWidthPx={max_width}&key={key}"
    try:
        resp = httpx.get(url, timeout=8, follow_redirects=True)
        if resp.status_code == 200:
            # The API redirects to the actual image URL
            return str(resp.url)
    except Exception as e:
        log.warning(f"[places] photo error: {e}")
    return None


def search_destination_pois(
    destination: str,
    *,
    language: str = "en",
    num_attractions: int = 15,
    num_restaurants: int = 10,
) -> dict[str, list[PlacePOI]]:
    """Search for top attractions and restaurants in a destination.

    Restaurants use multiple targeted queries (breakfast, local cuisine, dinner)
    to get diverse, high-quality results instead of one generic query.

    Returns:
        {"attractions": [...], "restaurants": [...]}
    """
    key = _api_key()
    if not key:
        return {"attractions": [], "restaurants": []}

    lang_code = "zh" if language == "zh" else "en"

    if lang_code == "zh":
        attr_query = f"{destination} 必去景点 热门旅游景点"
        rest_queries = [
            (f"{destination} 当地特色美食 必吃餐厅", 5),
            (f"{destination} 早餐 早午餐 人气店", 3),
            (f"{destination} 晚餐 高评分餐厅 推荐", 4),
        ]
    else:
        attr_query = f"top tourist attractions in {destination}"
        rest_queries = [
            (f"best local cuisine restaurants in {destination}", 5),
            (f"popular breakfast brunch spots in {destination}", 3),
            (f"top rated dinner restaurants in {destination}", 4),
        ]

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        attr_future = pool.submit(
            search_places, attr_query,
            max_results=num_attractions, language=lang_code,
        )
        rest_futures = [
            pool.submit(search_places, q, max_results=n, language=lang_code)
            for q, n in rest_queries
        ]

        attractions = attr_future.result(timeout=15)

        # Merge restaurant results, deduplicate by display_name
        seen_names: set[str] = set()
        restaurants: list[PlacePOI] = []
        for fut in rest_futures:
            try:
                for poi in fut.result(timeout=15):
                    if poi.display_name.lower() not in seen_names:
                        seen_names.add(poi.display_name.lower())
                        restaurants.append(poi)
            except Exception:
                pass
        # Sort by rating (best first), cap at num_restaurants
        restaurants.sort(key=lambda p: (-p.rating, -p.user_ratings_total))
        restaurants = restaurants[:num_restaurants]

    log.info(
        f"[places] destination={destination}: "
        f"{len(attractions)} attractions, {len(restaurants)} restaurants"
    )
    return {"attractions": attractions, "restaurants": restaurants}


def format_pois_for_prompt(pois: dict[str, list[PlacePOI]], language: str = "en") -> str:
    """Format POI data into a text block for LLM prompt injection.

    Produces a concise reference list that the LLM should pick from.
    """
    lines = []
    zh = language == "zh"

    attractions = pois.get("attractions", [])
    restaurants = pois.get("restaurants", [])

    if not attractions and not restaurants:
        return ""

    if attractions:
        header = "Google Maps 热门景点（请优先从以下列表选择）" if zh else \
                 "Google Maps Top Attractions (prefer picking from this list)"
        lines.append(f"[{header}]")
        for p in attractions:
            rating_str = f" ({p.rating}/5, {p.user_ratings_total} reviews)" if p.rating else ""
            summary = f" — {p.editorial_summary}" if p.editorial_summary else ""
            lines.append(f"  - {p.display_name}{rating_str}{summary}")
        lines.append("")

    if restaurants:
        header = "Google Maps 热门餐厅（请优先从以下列表选择）" if zh else \
                 "Google Maps Top Restaurants (prefer picking from this list)"
        lines.append(f"[{header}]")
        for p in restaurants:
            rating_str = f" ({p.rating}/5, {p.user_ratings_total} reviews)" if p.rating else ""
            price = f" [{p.price_level}]" if p.price_level else ""
            summary = f" — {p.editorial_summary}" if p.editorial_summary else ""
            lines.append(f"  - {p.display_name}{rating_str}{price}{summary}")
        lines.append("")

    return "\n".join(lines)
