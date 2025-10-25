#!/usr/bin/env python3
"""
SF311 API Client - Complete Solution
Fetch and parse SF311 tickets from the API or HTML file
"""

import requests
from bs4 import BeautifulSoup
import re
import json
from typing import List, Dict, Optional


class SF311Client:
    """Client for fetching and parsing SF311 data."""
    
    def __init__(self, base_url: str = "https://mobile311.sfgov.org"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_tickets(self, page: int = 1, order_by: str = "chronological", 
                     order_direction: str = "descending") -> str:
        """
        Fetch tickets from SF311 API.
        
        Args:
            page: Page number
            order_by: Sort order
            order_direction: 'ascending' or 'descending'
            
        Returns:
            HTML content
        """
        url = f"{self.base_url}/tickets"
        params = {
            'order[by]': order_by,
            'order[direction]': order_direction,
            'page': page
        }
        