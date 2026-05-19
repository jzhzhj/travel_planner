# AI Travel Planner

An AI-powered travel planning assistant built with LangGraph and FastAPI. It collects your travel preferences through natural conversation, generates a personalized structured itinerary, and renders it as an interactive HTML page with maps, videos, flight deals, and more.

## Features

- **Conversational Planning** — Chat naturally to describe your trip; the AI incrementally extracts destinations, dates, budget, travel style, and interests
- **User Profiling** — Automatically classifies travelers (solo / couple / family / friends) and adjusts recommendations by budget tier, pace, and interests
- **Structured Itineraries** — Generates day-by-day plans with attractions, restaurants, transport, and time estimates via GPT-4
- **Interactive HTML Output** — Renders a self-contained HTML travel plan with:
  - Apple MapKit JS / Google Maps with route directions
  - YouTube & Bilibili video embeds for each destination
  - TikTok / Instagram / Xiaohongshu short-video links
  - Real-time flight deal widgets (Travelpayouts)
  - Weather forecasts and local holiday info
  - Wikipedia summaries and photos for attractions
- **RAG Knowledge Base** — ChromaDB vector store with curated travel guides for context-aware recommendations
- **Plan Modification** — Ask the AI to adjust your plan after generation — swap activities, change days, add stops
- **Bilingual** — Supports Chinese (zh) and English (en) prompts and output
- **Dual Interface** — Web UI (FastAPI + SSE streaming) and CLI

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o-mini (chat) + GPT-4.1 (plan generation) |
| Agent Framework | LangGraph StateGraph + ToolNode |
| Knowledge Retrieval | ChromaDB vector store + cosine similarity |
| Web Framework | FastAPI + Server-Sent Events (SSE) |
| Frontend | Vanilla HTML / CSS / JS single-page app |
| Maps | Apple MapKit JS / Google Maps (dual provider) |
| Flights | Travelpayouts affiliate widgets |
| Video | YouTube Data API / Bilibili API |
| Short Video | TikTok oEmbed / Instagram oEmbed / Xiaohongshu |

## Project Structure

```
ai_travel_planner/
├── app.py              # FastAPI web server (SSE streaming)
├── main.py             # CLI entry point
├── graph.py            # LangGraph state graph (core orchestration)
├── models.py           # Pydantic data models
├── profiles.py         # User profiling & strategy rules
├── renderer.py         # HTML travel plan renderer
├── wiki.py             # Wikipedia + Unsplash integration
├── prompts/            # Versioned prompt templates (zh/en)
├── rag/                # ChromaDB vector store & seed data
├── tools/              # LangGraph tool implementations
│   ├── currency.py     #   Currency conversion
│   ├── directions.py   #   Route & directions
│   ├── embeds.py       #   Video embed search
│   ├── holidays.py     #   Local holiday lookup
│   ├── links.py        #   Video link search
│   ├── place_search.py #   Place & attraction search
│   ├── travel_api.py   #   Flight deal queries
│   └── weather.py      #   Weather forecasts
└── static/             # Frontend assets
```

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key

### Installation

```bash
git clone https://github.com/jzhzhj/travel_planner.git
cd travel_planner

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
```

Required:
- `OPENAI_API_KEY` — OpenAI API key

Optional (enables additional features):
- `YOUTUBE_API_KEY` — YouTube video search
- `APPLE_MAPKIT_TOKEN` — Apple Maps (preferred)
- `GOOGLE_MAPS_API_KEY` — Google Maps (fallback)
- `INSTAGRAM_TOKEN` — Instagram embed thumbnails
- `CALENDARIFIC_API_KEY` — Holiday data for more countries

### Run

**Web UI:**
```bash
uvicorn app:app --reload
```
Then open http://localhost:8000 in your browser.

**CLI:**
```bash
python main.py
```

## Usage

1. Start a conversation — tell the AI where you want to go, when, and your preferences
2. The AI will ask follow-up questions to understand your travel style, budget, and interests
3. Once it has enough info, it automatically generates a full day-by-day itinerary
4. The plan is rendered as an interactive HTML page with maps, videos, and flight deals
5. You can ask the AI to modify the plan — swap activities, adjust days, add destinations

## License

MIT
