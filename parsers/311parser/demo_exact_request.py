#!/usr/bin/env python3
"""
SF311 API - Exact Implementation as Requested (FIXED VERSION)

This script does exactly what was requested:
1. Calls GET https://mobile311.sfgov.org/tickets?order[by]=chronological&order[direction]=descending&page=22
2. Parses the HTML content
3. Prints offense data in the specified format
4. Returns a list of offense objects
"""

import requests
from bs4 import BeautifulSoup
import re
import json
from typing import List, Dict, Optional


BASE_URL = "https://mobile311.sfgov.org"


def fetch_sf311_page(page: int = 22) -> str:
    """Fetch HTML from SF311 API."""
    url = f"{BASE_URL}/tickets"
    params = {
        'order[by]': 'chronological',
        'order[direction]': 'descending',
        'page': page
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print(f"Calling API: GET {url}")
    print(f"Parameters: {params}")
    print()
    
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    
    print(f"✓ Successfully fetched {len(response.text):,} characters of HTML")
    print()
    
    return response.text


def parse_offense(element) -> Dict[str, Optional[str]]:
    """Extract offense data from HTML element."""
    text = element.get_text(separator=' ', strip=True)
    
    offense = {
        'offense_type': None,
        'address': None,
        'coordinates': None,
        'description': None,
        'offense_id': None
    }
    
    # Extract ID
    # Try several ways to find a ticket id
    id_match = re.search(r'#(\d+)', text)
    if id_match:
        offense['offense_id'] = f"#{id_match.group(1)}"
    else:
        # Look for links like /tickets/12345
        a = element.find('a', href=re.compile(r'/tickets/\d+'))
        if a and a.get('href'):
            m = re.search(r'/tickets/(\d+)', a['href'])
            if m:
                offense['offense_id'] = f"#{m.group(1)}"
    
    # Extract coordinates from several possible places: inline, data attributes, or links
    coord_match = re.search(r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', text)
    if coord_match:
        offense['coordinates'] = f"({coord_match.group(1)}, {coord_match.group(2)})"
    else:
        # Check for data-lat / data-lng in the element or children
        lat = None
        lng = None
        if element.attrs:
            lat = element.attrs.get('data-lat') or element.attrs.get('data-latitude')
            lng = element.attrs.get('data-lng') or element.attrs.get('data-longitude')
        if not lat or not lng:
            # Search children for attributes
            for child in element.find_all(True):
                if not lat:
                    lat = child.attrs.get('data-lat') or child.attrs.get('data-latitude')
                if not lng:
                    lng = child.attrs.get('data-lng') or child.attrs.get('data-longitude')
                if lat and lng:
                    break
        # If still not found, try query parameters in links
        if not (lat and lng):
            link = element.find('a', href=re.compile(r'lat=|-?\d+\.\d+'))
            if link and link.get('href'):
                q = link['href']
                m1 = re.search(r'lat=\s*(-?\d+\.?\d*)', q)
                m2 = re.search(r'lng=\s*(-?\d+\.?\d*)|lon=\s*(-?\d+\.?\d*)', q)
                if m1 and m2:
                    lat = m1.group(1)
                    lng = m2.group(1) or m2.group(2)
        if lat and lng:
            offense['coordinates'] = f"({lat}, {lng})"
    
    # Extract address
    address_patterns = [
        r'\d+\s+[A-Za-z0-9\s]+\s+(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|Way|Dr|Drive|Ct|Court|Pl|Place|Ln|Lane)',
        r'\d+\s+\d+(?:st|nd|rd|th)\s+(?:St|Street|Ave|Avenue)',
        r'Intersection\s+[A-Za-z0-9\s,]+',
    ]
    
    for pattern in address_patterns:
        addr_match = re.search(pattern, text, re.IGNORECASE)
        if addr_match:
            offense['address'] = addr_match.group(0).strip()
            break
    
    # Extract offense type
    for tag in ['h1', 'h2', 'h3', 'h4', 'strong', 'b']:
        elem = element.find(tag)
        if elem:
            offense_type = elem.get_text(strip=True)
            if 5 < len(offense_type) < 100 and not offense_type.startswith('#'):
                offense['offense_type'] = offense_type
                break
    
    # Extract description: preferred by class, otherwise pick sensible <p> or <div>
    # Prefer specific visual description classes used in the listing UI
    # e.g., <p class="text-gray-700 block ..."> contains the short description
    desc_elem = element.find('p', class_=re.compile(r'text-gray-700|description|detail|body|lead', re.I))
    if desc_elem and desc_elem.get_text(strip=True):
        offense['description'] = desc_elem.get_text(strip=True)
    else:
        # Try any <p> that is not the address or header text; prefer later <p> nodes (listing description often follows address)
        candidates = element.find_all('p')
        for p in candidates[::-1]:
            pt = p.get_text(strip=True)
            if not pt:
                continue
            # skip if it looks like an address or contains the offense type or only the id
            if offense.get('address') and pt.strip() == offense['address']:
                continue
            if offense.get('offense_type') and pt.strip().startswith(offense['offense_type'][:10]):
                continue
            # otherwise take this as description
            offense['description'] = pt
            break

    # If description is identical to address or too short, treat as missing so we can enrich from detail page
    if offense.get('description') and offense.get('address') and offense['description'].strip() == offense['address'].strip():
        offense['description'] = None
    if offense.get('description') and len(offense['description'].strip()) < 5:
        offense['description'] = None

    # Capture a direct ticket path from data attributes if present (helps detail fetch)
    if not offense.get('offense_id'):
        # some containers include data-linkable-node-url-value like '/tickets/12345'
        dl = element.attrs.get('data-linkable-node-url-value') or element.attrs.get('data-linkable-node-url')
        if dl:
            m = re.search(r'/tickets/(\d+)', dl)
            if m:
                offense['offense_id'] = f"#{m.group(1)}"
    
    return offense


def parse_html(html_content: str) -> List[Dict[str, Optional[str]]]:
    """Parse HTML to extract all offenses."""
    soup = BeautifulSoup(html_content, 'html.parser')
    offenses = []

    # Try to find ticket containers
    containers = []

    # Strategy 1: Look for containers with 'flex items-center' classes (SF311 ticket list structure)
    # Filter to only divs that contain ticket links
    candidates = soup.find_all('div', class_=lambda x: x and 'flex' in x and 'items-center' in x)
    seen_tickets = set()
    for div in candidates:
        ticket_link = div.find('a', href=re.compile(r'/tickets/\d+'))
        if ticket_link:
            # Extract ticket ID to deduplicate
            match = re.search(r'/tickets/(\d+)', ticket_link['href'])
            if match:
                ticket_id = match.group(1)
                if ticket_id not in seen_tickets:
                    containers.append(div)
                    seen_tickets.add(ticket_id)

    # Strategy 2: Find by class patterns
    if not containers:
        for tag in ['article', 'div', 'li']:
            found = soup.find_all(tag, class_=re.compile(r'ticket|report|card', re.I))
            if found:
                containers = found
                break

    # Strategy 3: Find by links and go up to outer container
    if not containers:
        links = soup.find_all('a', href=re.compile(r'/tickets/\d+'))
        seen = set()
        for link in links:
            # Go up multiple levels to find the outer container with description
            current = link
            for _ in range(6):  # Go up max 6 levels
                parent = current.find_parent()
                if not parent or parent.name in ['body', 'html']:
                    break
                # Check if this parent has the description element
                if parent.find('p', class_=re.compile(r'text-gray-700')):
                    if id(parent) not in seen:
                        containers.append(parent)
                        seen.add(id(parent))
                    break
                current = parent

    # Parse each container
    for container in containers:
        offense = parse_offense(container)
        if offense and any(offense.values()):
            offenses.append(offense)

    return offenses


def fetch_ticket_detail(ticket_id_or_path: str) -> Dict[str, Optional[str]]:
    """Fetch a ticket detail page and try to extract description and coordinates.

    Accepts either an id like '101002855968' or a path like '/tickets/101002855968'.
    Returns a dict with optional 'description' and 'coordinates'.
    """
    result = {'description': None, 'coordinates': None}

    # Normalize to path
    if ticket_id_or_path.startswith('/tickets/'):
        path = ticket_id_or_path
    elif ticket_id_or_path.startswith('#'):
        path = f"/tickets/{ticket_id_or_path.lstrip('#')}"
    elif ticket_id_or_path.isdigit():
        path = f"/tickets/{ticket_id_or_path}"
    else:
        # fallback assume it's a path
        path = ticket_id_or_path

    url = f"{BASE_URL}{path}"

    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return result

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Try to find a long description area first
    # Common layouts use a prominent <p> or a div with classes indicating content
    desc_candidates = []
    # look for a main article or section
    main = soup.find('main') or soup.find('article') or soup
    for sel in [
        ("p", None),
        ("div", re.compile(r'desc|description|detail|body|content', re.I)),
        ("section", re.compile(r'desc|description|detail|body|content', re.I)),
    ]:
        tag, cls = sel
        if cls:
            found = main.find_all(tag, class_=cls)
        else:
            found = main.find_all(tag)
        for f in found:
            text = f.get_text(separator=' ', strip=True)
            if text and len(text) > 10:
                desc_candidates.append(text)
        if desc_candidates:
            break

    if desc_candidates:
        result['description'] = desc_candidates[0]

    # Try to find coordinates: data attributes, map iframe, or inline JSON
    # 1) data-lat / data-lng attributes
    lat = None
    lng = None
    el = soup.find(attrs={'data-lat': True}) or soup.find(attrs={'data-latitude': True})
    if el:
        lat = el.attrs.get('data-lat') or el.attrs.get('data-latitude')
        lng = el.attrs.get('data-lng') or el.attrs.get('data-longitude')

    # 2) look for map iframe or links with @lat,lng or query params
    if not (lat and lng):
        # look for iframe or a[href*="@"] pattern
        iframe = soup.find('iframe', src=True)
        if iframe and iframe['src']:
            m = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', iframe['src'])
            if m:
                lat, lng = m.group(1), m.group(2)
        if not (lat and lng):
            # look in any links
            for a in soup.find_all('a', href=True):
                href = a['href']
                # Apple Maps links use ll=lat%2Clng (URL-encoded comma)
                m = re.search(r'll=(-?\d+\.?\d*)%2C(-?\d+\.?\d*)', href)
                if m:
                    lat, lng = m.group(1), m.group(2)
                    break
                m = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', href)
                if m:
                    lat, lng = m.group(1), m.group(2)
                    break
                m1 = re.search(r'lat=\s*(-?\d+\.?\d*)', href)
                m2 = re.search(r'lng=\s*(-?\d+\.?\d*)|lon=\s*(-?\d+\.?\d*)', href)
                if m1 and m2:
                    lat = m1.group(1)
                    lng = m2.group(1) or m2.group(2)
                    break

    # 3) search JSON in scripts for latitude/longitude
    if not (lat and lng):
        for script in soup.find_all('script'):
            if not script.string:
                continue
            s = script.string
            m_lat = re.search(r'"latitude"\s*:\s*([+-]?\d+\.?\d*)', s)
            m_lng = re.search(r'"longitude"\s*:\s*([+-]?\d+\.?\d*)', s)
            if m_lat and m_lng:
                lat, lng = m_lat.group(1), m_lng.group(1)
                break
            m_lat = re.search(r'"lat"\s*:\s*([+-]?\d+\.?\d*)', s)
            m_lng = re.search(r'"lng"\s*:\s*([+-]?\d+\.?\d*)', s)
            if m_lat and m_lng:
                lat, lng = m_lat.group(1), m_lng.group(1)
                break

    if lat and lng:
        result['coordinates'] = f"({lat}, {lng})"

    return result


def get_sf311_offenses(page: int = 22) -> List[Dict[str, Optional[str]]]:
    """
    Fetch and parse SF311 offenses.
    
    Returns:
        list: List of offense dictionaries
    """
    print("=" * 80)
    print(f"SF311 OFFENSE FETCHER - PAGE {page}")
    print("=" * 80)
    print()
    
    # Step 1: Fetch from API
    html_content = fetch_sf311_page(page)
    
    # Step 2: Parse HTML
    print("Parsing offense data...")
    offenses = parse_html(html_content)
    print(f"✓ Found {len(offenses)} offenses")
    print()

    # Enrich offenses with detail page when fields are missing
    print("Enriching offenses from detail pages where needed...")
    for offense in offenses:
        needs_coords = not offense.get('coordinates')
        # treat description equal-to-address as missing
        needs_desc = (not offense.get('description')) or (offense.get('address') and offense.get('description') == offense.get('address'))
        if (needs_coords or needs_desc):
            # determine ticket id/path to fetch: prefer data attribute / tickets id
            tid = None
            if offense.get('offense_id'):
                tid = offense['offense_id'].lstrip('#')
            # try to find a data-linkable-node-url-value in the offense container if present
            if not tid and offense.get('offense_id'):
                tid = offense['offense_id'].lstrip('#')
            if not tid:
                # try to find any href-like value in the offense dict (we didn't store it); skip if missing
                pass
            if tid:
                detail = fetch_ticket_detail(tid)
                if needs_desc and detail.get('description'):
                    offense['description'] = detail['description']
                if needs_coords and detail.get('coordinates'):
                    offense['coordinates'] = detail['coordinates']
    print("✓ Enrichment complete")
    print()
    
    # Step 3: Print in specified format
    print("=" * 80)
    print(f"OFFENSE DATA (Page {page})")
    print("=" * 80)
    print()
    
    for offense in offenses:
        if offense['offense_type']:
            print(offense['offense_type'])
        if offense['address']:
            print(offense['address'])
        if offense['coordinates']:
            print(offense['coordinates'])
        if offense['description']:
            print(offense['description'])
        if offense['offense_id']:
            print(offense['offense_id'])
        print()
    
    print("=" * 80)
    print(f"Total: {len(offenses)} offenses")
    print("=" * 80)
    
    # Step 4: Return list
    return offenses


if __name__ == "__main__":
    try:
        # Call the function
        offense_list = get_sf311_offenses(page=22)
        
        # Save to JSON
        # Save next to this script so the path works regardless of cwd
        from pathlib import Path

        outpath = Path(__file__).resolve().parent / 'offenses_page22.json'
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, 'w') as f:
            json.dump(offense_list, f, indent=2)

        print(f"\n✓ Saved {len(offense_list)} offenses to {outpath}")
        
        # Show some examples
        if offense_list:
            print("\n" + "=" * 80)
            print("EXAMPLE USAGE")
            print("=" * 80)
            
            print(f"\nFirst offense:")
            first = offense_list[0]
            print(f"  Type: {first['offense_type']}")
            print(f"  Address: {first['address']}")
            print(f"  ID: {first['offense_id']}")
            
            # Count offense types
            types = {}
            for o in offense_list:
                if o['offense_type']:
                    types[o['offense_type']] = types.get(o['offense_type'], 0) + 1
            
            print(f"\nOffense type distribution:")
            for offense_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {offense_type}: {count}")
        
    except requests.RequestException as e:
        print(f"\n✗ Network Error: {e}")
        print("\nMake sure you have internet access and the SF311 API is available.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()