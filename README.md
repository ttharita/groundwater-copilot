# Groundwater Intelligence Copilot (GraphRAG + GeoAI)

Executive-friendly groundwater analysis tool with interactive map, GraphRAG-powered retrieval, and Gemini LLM copilot.

## Features

- Interactive map with wells, hydraulic head grid, and flow direction arrows
- In-memory knowledge graph (GraphRAG) built from local GIS data
- Copilot chat with deterministic retrieval + optional Gemini LLM answers (Thai)
- Evidence cards, top insights, and 1-page printable report generation

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Gemini API key (optional but recommended)

**Option A** – Streamlit secrets (preferred):

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "YOUR_KEY"
```

**Option B** – Environment variable:

```bash
# Linux / macOS
export GEMINI_API_KEY="YOUR_KEY"

# Windows (PowerShell)
$env:GEMINI_API_KEY = "YOUR_KEY"
```

If no key is provided, the app still works using template-based fallback answers.

### 3. Place data files

Ensure the following files exist in `./data/`:

- `wells_wgs84.geojson`
- `head_grid_wgs84.geojson`
- `flow_dir_arrows_wgs84.geojson`
- `intervals.csv`
- `surface_points.csv`
- `orientations.csv`

### 4. Run

```bash
streamlit run app.py
```

## Architecture

```
app.py                  # Main Streamlit UI
utils/
  data_loader.py        # Load GeoJSON / CSV files
  geospatial.py         # Haversine, point-in-polygon, nearest cell
  graph.py              # In-memory knowledge graph (nodes + edges)
  retrieval.py          # Keyword intent → graph traversal → evidence
  gemini_client.py      # Gemini API client (key from secrets/env)
  templates.py          # Fallback answers + report HTML builder
data/                   # Local GIS data files (not committed)
```

## GraphRAG Schema

**Nodes:** Well, Interval, Formation, HeadCell

**Edges:**
- Well → HAS_INTERVAL → Interval
- Interval → PART_OF_FORMATION → Formation
- Well → IN_CELL → HeadCell (point-in-polygon)
- Well → NEAR → Well (≤ 2 km haversine)
- HeadCell → FLOWS_TO → HeadCell (from flow arrows)
