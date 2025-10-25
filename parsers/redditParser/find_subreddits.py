import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import sys


class SubredditFinder:
    """Finds and ranks subreddits based on search criteria"""
    
    def __init__(self):
        self.base_url = "https://www.reddit.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_delay = 1.0  # 1 second between requests
    
    def _make_request(self, url: str, params: dict = None) -> Optional[dict]:
        """Make a request with rate limiting and error handling"""
        # Rate limiting: ensure minimum delay between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            # Update rate limiting tracking
            self.last_request_time = time.time()
            self.request_count += 1
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None
    
    def search_subreddits(self, query: str, limit: int = 50) -> List[Dict]:
        """Search for subreddits matching a query"""
        print(f"Searching Reddit for: '{query}'...")
        
        url = f"{self.base_url}/subreddits/search.json"
        params = {
            'q': query,
            'limit': min(limit, 100),
            'include_over_18': 'off'
        }
        
        data = self._make_request(url, params)
        if not data:
            return []
        
        subreddits = []
        for child in data.get('data', {}).get('children', []):
            sub_data = child['data']
            subreddits.append({
                'name': sub_data.get('display_name'),
                'title': sub_data.get('title'),
                'description': sub_data.get('public_description', ''),
                'subscribers': sub_data.get('subscribers') or 0,
                'url': f"https://www.reddit.com/r/{sub_data.get('display_name')}",
                'created_utc': datetime.fromtimestamp(sub_data.get('created_utc', 0)).isoformat(),
                'active_users': sub_data.get('active_user_count') or 0,
                'over18': sub_data.get('over18', False)
            })
        
        print(f"   Found {len(subreddits)} subreddits")
        time.sleep(1)  # Rate limiting
        
        return subreddits
    
    def calculate_relevance_score(self, sub: Dict, city: str, keywords: List[str]) -> float:
        """
        Calculate how relevant a subreddit is to the city
        Higher score = more relevant
        """
        score = 0.0
        name = sub['name'].lower()
        title = sub['title'].lower()
        desc = sub['description'].lower()
        
        city_lower = city.lower()
        city_words = city_lower.split()
        
        # Strong positive signals
        # Subreddit name contains city name
        if city_lower.replace(' ', '') in name:
            score += 50
        elif any(word in name for word in city_words if len(word) > 2):
            score += 30
        
        # Title contains city name
        if city_lower in title:
            score += 20
        
        # Description contains city name
        if city_lower in desc:
            score += 15
        
        # Keywords in description
        for keyword in keywords:
            if keyword.lower() in desc:
                score += 10
        
        # Negative signals - things that suggest it's not about the city
        negative_keywords = [
            'fantasy', 'football', 'baseball', 'basketball', 'sports', 'team',
            'game', 'gaming', 'nsfw', 'dating', 'hookup', 'r4r', 'personals',
            'meme', 'circle', 'jerk', 'porn', 'xxx', 'buy', 'sell', 'market'
        ]
        
        for neg_word in negative_keywords:
            if neg_word in name or neg_word in title:
                score -= 30
        
        # Boost for common city/location subreddit patterns
        location_indicators = ['ask', 'housing', 'jobs', 'list', 'events', 'food']
        for indicator in location_indicators:
            if indicator in name:
                score += 15
        
        return score
    
    def find_subreddits(self, city: str, keywords: List[str] = None, 
                       min_subscribers: int = 1000, max_results: int = 10,
                       min_relevance_score: float = 20.0) -> Dict:
        """
        Main function: Find and rank subreddits
        
        Args:
            city: City name to search for
            keywords: Additional search terms (defaults to just city name)
            min_subscribers: Minimum subscriber count
            max_results: Maximum number of subreddits to return
            min_relevance_score: Minimum relevance score (0-100)
            
        Returns:
            Dictionary with subreddit results and metadata
        """
        if keywords is None:
            keywords = [city]
        
        print("\n" + "="*80)
        print(f" SUBREDDIT DISCOVERY: {city.upper()}")
        print("="*80)
        print(f"Keywords: {', '.join(keywords)}")
        print(f"Min subscribers: {min_subscribers:,}")
        print(f"Min relevance score: {min_relevance_score}")
        print(f"Max results: {max_results}")
        print("="*80 + "\n")
        
        # Search using all keywords
        all_subreddits = []
        for keyword in keywords:
            results = self.search_subreddits(keyword, limit=50)
            all_subreddits.extend(results)
        
        # Remove duplicates (by name)
        unique_subs = {sub['name']: sub for sub in all_subreddits}.values()
        unique_subs = list(unique_subs)
        
        print(f"\n Processing Results:")
        print(f"   Total found: {len(unique_subs)}")
        
        # Calculate relevance scores
        for sub in unique_subs:
            sub['relevance_score'] = self.calculate_relevance_score(sub, city, keywords)
        
        # Filter by minimum subscribers AND relevance
        filtered = [
            sub for sub in unique_subs 
            if sub['subscribers'] >= min_subscribers
            and sub['relevance_score'] >= min_relevance_score
        ]
        print(f"   After filtering (>{min_subscribers:,} subs, relevance >{min_relevance_score}): {len(filtered)}")
        
        # Sort by relevance score first, then subscribers
        sorted_subs = sorted(filtered, 
                            key=lambda x: (x['relevance_score'], x['subscribers']), 
                            reverse=True)
        
        # Select top N
        selected = sorted_subs[:max_results]
        
        print(f"\nTOP {len(selected)} SUBREDDITS (by relevance):\n")
        for i, sub in enumerate(selected, 1):
            print(f"{i:2}. r/{sub['name']:.<30} {sub['subscribers']:>8,} subs | Score: {sub['relevance_score']:.0f}")
            if sub['description']:
                desc = sub['description'][:60] + "..." if len(sub['description']) > 60 else sub['description']
                print(f"    {desc}")
        
        # Prepare output
        result = {
            'query': {
                'city': city,
                'keywords': keywords,
                'min_subscribers': min_subscribers,
                'min_relevance_score': min_relevance_score,
                'max_results': max_results
            },
            'stats': {
                'total_found': len(unique_subs),
                'after_filtering': len(filtered),
                'selected': len(selected)
            },
            'subreddits': selected,
            'subreddit_names': [sub['name'] for sub in selected],
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def save_results(self, results: Dict, filename: str = 'subreddits_found.json'):
        """Save results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n Results saved to: {filename}")


def find_subreddits(city: str, keywords: List[str] = None, 
                   min_subscribers: int = 1000, max_results: int = 10,
                   min_relevance_score: float = 20.0) -> Dict:
    """
    Convenience function for use as module
    
    Example:
        from find_subreddits import find_subreddits
        results = find_subreddits("san francisco", ["sf", "bay area"])
        subreddit_names = results['subreddit_names']
    """
    finder = SubredditFinder()
    return finder.find_subreddits(city, keywords, min_subscribers, max_results, min_relevance_score)


def main():
    """Run as standalone script"""
    
    # ========================================================================
    # CONFIGURATION - Edit these for your city
    # ========================================================================
    CITY = "san francisco"
    KEYWORDS = ["san francisco", "bay area", "sf"]
    MIN_SUBSCRIBERS = 1000
    MIN_RELEVANCE_SCORE = 20.0  # Minimum relevance score (0-100)
    MAX_RESULTS = 10
    OUTPUT_FILE = "subreddits_found.json"
    
    # Run discovery
    finder = SubredditFinder()
    results = finder.find_subreddits(
        city=CITY,
        keywords=KEYWORDS,
        min_subscribers=MIN_SUBSCRIBERS,
        max_results=MAX_RESULTS,
        min_relevance_score=MIN_RELEVANCE_SCORE
    )
    
    # Save results
    finder.save_results(results, OUTPUT_FILE)
    
    print("\n" + "="*80)
    print("DISCOVERY COMPLETE")
    print("="*80)
    print(f"\n Found {len(results['subreddits'])} subreddits")
    print(f" Saved to: {OUTPUT_FILE}")
    print(f"\n Top 5 subreddits to use for scraping:")
    for i, name in enumerate(results['subreddit_names'][:5], 1):
        print(f"   {i}. r/{name}")
    
    print("\n Next step:")
    print(f"   python scrape_subreddits.py")
    print("   (This will automatically use the top 5 subreddits)")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
