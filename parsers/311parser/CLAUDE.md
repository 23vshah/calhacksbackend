# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based SF311 data parser that scrapes and parses civic issue reports from the San Francisco 311 mobile API (mobile311.sfgov.org). The parser extracts structured data about offenses/tickets including type, location, coordinates, and descriptions from HTML responses.

## Development Commands

```bash
# Set up virtual environment (first time only)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the main scraper (fetches page 22 by default)
python demo_exact_request.py

# Use the client library programmatically
python sf311_client.py
```

## Architecture

### Core Components

**demo_exact_request.py** (Main executable script)
- Primary entry point for scraping SF311 data
- Orchestrates the full workflow: fetch → parse → enrich → output → save
- Fetches HTML from `https://mobile311.sfgov.org/tickets` with pagination
- Parses offense data from HTML using BeautifulSoup
- Enriches incomplete records by fetching individual ticket detail pages
- Outputs formatted offense data to console and saves to JSON
- Default behavior: fetches page 22, saves to `offenses_page22.json`

**sf311_client.py** (Reusable client class)
- Incomplete/partial implementation of SF311Client class
- Provides session management and basic fetch functionality
- Not currently used by demo_exact_request.py (which uses standalone functions instead)

### Data Extraction Logic

**HTML Parsing Strategy** (`parse_html` and `parse_offense` functions)
The parser uses a multi-strategy approach to extract data from inconsistent HTML:

1. **Container Detection**: Finds ticket containers via:
   - Class-based search for 'ticket', 'report', or 'card' classes
   - Link-based search for `/tickets/{id}` URLs, then finds parent containers

2. **Field Extraction** (parse_offense function):
   - **Offense ID**: Extracts from `#12345` patterns in text, `/tickets/12345` in links, or `data-linkable-node-url-value` attributes
   - **Coordinates**: Multi-source extraction from:
     - Inline text patterns `(lat, lon)`
     - Data attributes (`data-lat`, `data-lng`, `data-latitude`, `data-longitude`)
     - Query parameters in map links (`lat=`, `lng=`, `lon=`)
   - **Address**: Regex-based pattern matching for street addresses and intersections
   - **Offense Type**: Extracted from header tags (h1-h4, strong, b) with length validation
   - **Description**: Prefers elements with `text-gray-700` class, falls back to `<p>` tags with deduplication logic

3. **Data Enrichment** (`fetch_ticket_detail` function):
   - Triggered when coordinates or descriptions are missing from list view
   - Fetches individual ticket detail pages at `/tickets/{id}`
   - Extracts richer data from detail page:
     - Longer descriptions from main content areas
     - Coordinates from data attributes, map iframes, embedded JSON in `<script>` tags
   - Prevents duplicate work by only enriching when fields are genuinely missing

### Data Flow

1. `fetch_sf311_page(page)` → Fetches HTML from API with chronological descending order
2. `parse_html(html)` → Extracts list of offense dictionaries with 5 fields
3. Enrichment loop → Fetches detail pages for incomplete records
4. Console output → Prints formatted offense data
5. JSON export → Saves to `offenses_page{page}.json`

### Output Format

Each offense dictionary contains:
```python
{
    'offense_type': str | None,      # e.g., "Graffiti", "Blocked driveway & illegal parking"
    'address': str | None,           # e.g., "28 2 Nd St", "416 Sloat Blvd"
    'coordinates': str | None,       # Format: "(lat, lng)" - note string, not tuple
    'description': str | None,       # Short description or status (e.g., "Removed - Case Resolved.")
    'offense_id': str | None         # Format: "#101002856039"
}
```

## Technology Stack

- **Python 3.x** (developed on 3.13, likely compatible with 3.8+)
- **requests** (2.31.0+) - HTTP client for API calls
- **BeautifulSoup4** (4.12.0+) - HTML parsing
- **lxml** (4.9.0+) - Fast XML/HTML parser backend for BeautifulSoup
- **html5lib** (1.1+) - Optional alternative HTML parser

## Important Implementation Notes

### HTML Parsing Challenges
- The SF311 mobile site uses inconsistent HTML structure across pages
- Container elements vary (article, div, li tags with different class names)
- Multiple fallback strategies are necessary for robust extraction
- Some fields are only available in detail pages, requiring secondary requests

### Coordinate Handling
- Coordinates are stored as strings in "(lat, lng)" format, not as numeric tuples
- Lat/lng values may be found in 7+ different locations (attributes, links, JSON, iframes)
- Always validate coordinate presence before attempting geocoding operations

### Description Deduplication
- List view often duplicates the address as description
- Logic filters out descriptions that exactly match the address
- Descriptions shorter than 5 characters are treated as missing
- Enrichment from detail page provides more meaningful descriptions

### Network Resilience
- Uses session with User-Agent header to avoid blocks
- 30-second timeout on list view requests, 15-second on detail requests
- Try/except around detail fetching prevents one failure from breaking entire batch
- No rate limiting currently implemented - add delays if scraping multiple pages

### Session Management
- sf311_client.py sets up a requests.Session with persistent headers
- demo_exact_request.py does not use sessions (creates new connection per request)
- Consider refactoring to use SF311Client for connection pooling if scaling up

## File Outputs

- `offenses_page22.json`: Default output file containing parsed offense data
- File is saved in same directory as the script using `Path(__file__).resolve().parent`
- JSON is pretty-printed with 2-space indentation

## Integration Notes

This parser is designed to feed data into the parent React/deck.gl visualization application. The backend API (assumed to run on localhost:3000) likely:
- Imports this parser module to fetch fresh SF311 data
- Transforms offense dictionaries to the format expected by frontend (add topic classification, normalize coordinates to numeric lat/lon)
- Serves data via `/api/flattened` and `/api/flattened/{topic}` endpoints

When extending this parser:
- Ensure coordinate strings are converted to numeric values before sending to frontend
- Add topic classification logic to group offenses (graffiti, parking, flooding, etc.)
- Consider implementing pagination loops to fetch multiple pages
- Add data persistence (database) rather than just JSON file export
