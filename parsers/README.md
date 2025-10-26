# Parsers

This directory contains data parsers for gathering city-related information from external sources.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Subreddit Parser

The subreddit parser allows you to find and scrape Reddit posts related to specific cities and topics.

### Functions

#### `find_subreddits(city, keywords=None, limit=10)`
Find subreddits related to a specific city.

**Parameters:**
- `city` (str): The name of the city to search for
- `keywords` (list, optional): Additional keywords to refine the search
- `limit` (int, optional): Maximum number of subreddits to return (default: 10)

**Returns:** List of subreddit names

**Example:**
```python
from redditParser.find_subreddits import find_subreddits

subreddits = find_subreddits("San Francisco", keywords=["housing", "transportation"])
print(subreddits)
```

#### `get_subreddit_posts(subreddit, sort='hot', limit=25, time_filter='all')`
Get posts from a specific subreddit for a given topic.

**Parameters:**
- `subreddit` (str): The name of the subreddit to scrape
- `sort` (str): Sorting method - 'hot', 'top', 'new', 'rising' (default: 'hot')
- `limit` (int): Number of posts to retrieve (default: 25)
- `time_filter` (str): Time range for sorting - 'all', 'day', 'week', 'month', 'year' (default: 'all')

**Returns:** List of post objects

**Example:**
```python
from redditParser.scrape_subreddits import SubredditScraper

scraper = SubredditScraper()
posts = scraper.get_subreddit_posts("sftransportation", sort='top', limit=50)
```

## 311 Parser

The 311 parser fetches 311 service request data from San Francisco's public API.

### Functions

#### `get_sf311_offenses(page=22)`
Get 311 service requests from San Francisco. Each page contains 10 requests, and page 1 is the most recent.

**Parameters:**
- `page` (int): Page number to fetch (default: 22). You can go up to 1000 pages.

**Returns:** List of offense/service request objects

**Note:** Page 1 contains the most recent requests. Each page has 10 requests. The API supports up to 1000 pages.

**Example:**
```python
from 311parser.demo_exact_request import get_sf311_offenses

# Get the first page (most recent)
recent_requests = get_sf311_offenses(page=1)

# Get page 22
older_requests = get_sf311_offenses(page=22)

for request in recent_requests:
    print(f"{request['offense_id']}: {request['offense_type']} at {request['address']}")
```

**Request Object Structure:**
- `offense_type`: Type of service request
- `address`: Location of the request
- `coordinates`: GPS coordinates
- `description`: Description of the request
- `offense_id`: Unique identifier for the request


