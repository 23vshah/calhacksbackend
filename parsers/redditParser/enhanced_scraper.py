#!/usr/bin/env python3
"""
Enhanced Reddit San Francisco Issues Scraper - NO AUTHENTICATION REQUIRED
Uses Reddit's public JSON API - works immediately without any setup
"""

import requests
import json
import csv
import time
import sys
from datetime import datetime
from typing import List, Dict, Optional
from collections import Counter


class EnhancedRedditScraper:
    """Enhanced scraper for Reddit using JSON API (no authentication required)"""
    
    def __init__(self):
        self.base_url = "https://www.reddit.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.request_count = 0
        self.start_time = time.time()
        
    def _rate_limit(self):
        """Simple rate limiting to be respectful"""
        self.request_count += 1
        if self.request_count % 10 == 0:
            time.sleep(2)  # Pause every 10 requests
    
    def _make_request(self, url: str, params: dict = None) -> Optional[dict]:
        """Make a request with error handling"""
        self._rate_limit()
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f" Error making request: {e}")
            return None
    
    def search_subreddits(self, query: str, limit: int = 50) -> List[Dict]:
        """
        Search for subreddits matching a query
        
        Args:
            query: Search term (e.g., "san francisco")
            limit: Number of results (max 100)
        """
        print(f"Searching for subreddits: '{query}'...")
        
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
                'description': sub_data.get('public_description'),
                'subscribers': sub_data.get('subscribers') or 0,  # Handle None
                'url': f"https://www.reddit.com/r/{sub_data.get('display_name')}",
                'created_utc': datetime.fromtimestamp(sub_data.get('created_utc', 0)).isoformat(),
                'over18': sub_data.get('over18', False)
            })
        
        print(f" Found {len(subreddits)} subreddits")
        return subreddits
    
    def get_subreddit_posts(self, subreddit: str, sort: str = 'hot', 
                           limit: int = 100, time_filter: str = 'all',
                           after: str = None) -> List[Dict]:
        """
        Get posts from a specific subreddit
        
        Args:
            subreddit: Name of the subreddit
            sort: 'hot', 'new', 'top', 'rising', 'controversial'
            limit: Number of posts to retrieve (max 100 per request)
            time_filter: For 'top'/'controversial': 'hour', 'day', 'week', 'month', 'year', 'all'
            after: Pagination token (for getting more than 100 posts)
        """
        print(f" Getting {sort} posts from r/{subreddit}...")
        
        url = f"{self.base_url}/r/{subreddit}/{sort}.json"
        params = {
            'limit': min(limit, 100),
            't': time_filter
        }
        
        if after:
            params['after'] = after
        
        data = self._make_request(url, params)
        if not data:
            return []
        
        posts = []
        for child in data.get('data', {}).get('children', []):
            post_data = child['data']
            posts.append({
                'id': post_data.get('id'),
                'title': post_data.get('title'),
                'author': post_data.get('author'),
                'score': post_data.get('score', 0),
                'upvote_ratio': post_data.get('upvote_ratio', 0),
                'num_comments': post_data.get('num_comments', 0),
                'created_utc': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat(),
                'url': f"https://www.reddit.com{post_data.get('permalink')}",
                'selftext': post_data.get('selftext', '')[:1000],
                'link_flair_text': post_data.get('link_flair_text'),
                'subreddit': post_data.get('subreddit'),
                'domain': post_data.get('domain'),
                'is_self': post_data.get('is_self', False),
                'thumbnail': post_data.get('thumbnail', ''),
                'gilded': post_data.get('gilded', 0),
                'after': data.get('data', {}).get('after')  # For pagination
            })
        
        print(f" Retrieved {len(posts)} posts")
        return posts
    
    def search_posts(self, subreddit: str, query: str, sort: str = 'relevance',
                    limit: int = 100, time_filter: str = 'all') -> List[Dict]:
        """
        Search for posts in a subreddit
        
        Args:
            subreddit: Subreddit to search in (use 'all' for site-wide)
            query: Search query
            sort: 'relevance', 'hot', 'top', 'new', 'comments'
            limit: Number of results
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
        """
        print(f" Searching r/{subreddit} for '{query}'...")
        
        url = f"{self.base_url}/r/{subreddit}/search.json"
        params = {
            'q': query,
            'restrict_sr': 'on' if subreddit != 'all' else 'off',
            'sort': sort,
            'limit': min(limit, 100),
            't': time_filter,
            'include_over_18': 'off'
        }
        
        data = self._make_request(url, params)
        if not data:
            return []
        
        posts = []
        for child in data.get('data', {}).get('children', []):
            post_data = child['data']
            posts.append({
                'id': post_data.get('id'),
                'title': post_data.get('title'),
                'author': post_data.get('author'),
                'score': post_data.get('score', 0),
                'num_comments': post_data.get('num_comments', 0),
                'created_utc': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat(),
                'url': f"https://www.reddit.com{post_data.get('permalink')}",
                'selftext': post_data.get('selftext', '')[:1000],
                'link_flair_text': post_data.get('link_flair_text'),
                'subreddit': post_data.get('subreddit')
            })
        
        print(f" Found {len(posts)} matching posts")
        return posts
    
    def get_multiple_pages(self, subreddit: str, sort: str = 'hot', 
                          total_posts: int = 250, time_filter: str = 'all') -> List[Dict]:
        """
        Get more than 100 posts by using pagination
        
        Args:
            subreddit: Subreddit name
            sort: Sort method
            total_posts: Total number of posts to retrieve
            time_filter: Time filter
        """
        print(f" Getting up to {total_posts} posts from r/{subreddit}...")
        
        all_posts = []
        after = None
        
        while len(all_posts) < total_posts:
            posts = self.get_subreddit_posts(subreddit, sort, 100, time_filter, after)
            
            if not posts:
                break
            
            all_posts.extend(posts)
            
            # Get pagination token
            after = posts[-1].get('after') if posts else None
            if not after:
                break
            
            print(f"   Progress: {len(all_posts)} posts retrieved...")
            time.sleep(2)  # Be nice to Reddit's servers
        
        print(f" Total retrieved: {len(all_posts)} posts")
        return all_posts[:total_posts]
    
    def bulk_search_issues(self, subreddits: List[str], keywords: List[str], 
                          posts_per_keyword: int = 25) -> List[Dict]:
        """
        Search multiple subreddits for multiple keywords
        
        Args:
            subreddits: List of subreddit names
            keywords: List of search terms
            posts_per_keyword: Posts to get per keyword per subreddit
        """
        print(f"\nBulk searching {len(subreddits)} subreddits for {len(keywords)} keywords...")
        
        all_posts = []
        total_searches = len(subreddits) * len(keywords)
        current = 0
        
        for subreddit in subreddits:
            for keyword in keywords:
                current += 1
                print(f"\n[{current}/{total_searches}] r/{subreddit} → '{keyword}'")
                
                posts = self.search_posts(
                    subreddit=subreddit,
                    query=keyword,
                    sort='top',
                    time_filter='month',
                    limit=posts_per_keyword
                )
                
                all_posts.extend(posts)
                time.sleep(2)  # Rate limiting
        
        print(f"\n Bulk search complete! Total posts: {len(all_posts)}")
        return all_posts
    
    def get_top_keywords(self, posts: List[Dict], top_n: int = 20) -> Dict[str, int]:
        """Extract most common keywords from post titles"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'what', 'how', 'why', 'when', 'where', 'who', 'this', 'that', 'my',
            'your', 'can', 'just', 'about', 'any', 'some', 'all'
        }
        
        word_counts = Counter()
        
        for post in posts:
            words = post['title'].lower().split()
            for word in words:
                word = ''.join(c for c in word if c.isalnum())
                if word and len(word) > 2 and word not in stop_words:
                    word_counts[word] += 1
        
        return dict(word_counts.most_common(top_n))
    
    def save_to_csv(self, data: List[Dict], filename: str):
        """Save data to CSV file"""
        if not data:
            print("  No data to save")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        print(f" Saved {len(data)} items to {filename}")
    
    def save_to_json(self, data, filename: str):
        """Save data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f" Saved data to {filename}")
    
    def print_stats(self):
        """Print scraping statistics"""
        elapsed = time.time() - self.start_time
        print(f"\n Stats: {self.request_count} requests in {elapsed:.1f} seconds")


def main():
    """Main function - Enhanced SF Issues scraping"""
    scraper = EnhancedRedditScraper()
    
    print("\n" + "="*80)
    print(" CITY REDDIT ISSUES SCRAPER - NO AUTH REQUIRED")
    print("="*80)
    
    # ========================================================================
    # CONFIGURATION - Change these for different cities!
    # ========================================================================
    CITY_NAME = "san francisco"  # Change this to any city!
    CITY_KEYWORDS = ["san francisco", "bay area", "sf"]  # Related search terms
    MIN_SUBSCRIBERS = 1000  # Minimum subscribers to consider
    MAX_SUBREDDITS = 10  # Max number of subreddits to scrape from
    
    print(f"\nSearching for: {CITY_NAME}")
    print(f" Will select top {MAX_SUBREDDITS} subreddits with >{MIN_SUBSCRIBERS:,} subscribers")
    
    # Common SF issues to search for
    issue_keywords = [
        'homelessness',
        'housing crisis',
        'crime',
        'public safety',
        'BART',
        'traffic',
        'cost of living',
        'rent',
        'affordable housing',
        'gentrification'
    ]
    
    # ========================================================================
    # PART 1: Find and Select City-Related Subreddits
    # ========================================================================
    print("\n" + "="*80)
    print(f"PART 1: Finding Subreddits Related to {CITY_NAME.title()}")
    print("="*80)
    
    # Search for subreddits using all keywords
    all_subreddits = []
    for keyword in CITY_KEYWORDS:
        results = scraper.search_subreddits(keyword, limit=50)
        all_subreddits.extend(results)
    
    # Remove duplicates (by subreddit name)
    unique_subreddits = {sub['name']: sub for sub in all_subreddits}.values()
    unique_subreddits = list(unique_subreddits)
    
    # Filter by minimum subscribers and sort by subscriber count
    filtered_subs = [
        sub for sub in unique_subreddits 
        if (sub['subscribers'] or 0) >= MIN_SUBSCRIBERS
    ]
    sorted_subs = sorted(filtered_subs, key=lambda x: x['subscribers'] or 0, reverse=True)
    
    # Select top N subreddits to scrape
    selected_subreddits = sorted_subs[:MAX_SUBREDDITS]
    sf_subreddits = [sub['name'] for sub in selected_subreddits]
    
    if selected_subreddits:
        print(f"\n Found {len(unique_subreddits)} total subreddits")
        print(f" Selected top {len(selected_subreddits)} subreddits for scraping:")
        print(f"\n Will scrape these subreddits:")
        for i, sub in enumerate(selected_subreddits, 1):
            subs = sub['subscribers'] or 0
            print(f"{i:2}. r/{sub['name']:.<30} {subs:>8,} subscribers")
        
        scraper.save_to_json(unique_subreddits, 'sf_subreddits_all.json')
        scraper.save_to_json(selected_subreddits, 'sf_subreddits_selected.json')
    else:
        print(f" No subreddits found for {CITY_NAME}")
        return
    
    # ========================================================================
    # PART 2: Get recent posts from largest subreddit
    # ========================================================================
    print("\n" + "="*80)
    print(f"PART 2: Getting Recent Posts from r/{sf_subreddits[0]}")
    print("="*80)
    
    # Get more posts using pagination from the largest subreddit
    recent_posts = scraper.get_multiple_pages(sf_subreddits[0], sort='hot', total_posts=200)
    
    if recent_posts:
        print(f"\n Top 10 Hot Posts:")
        for i, post in enumerate(recent_posts[:10], 1):
            print(f"{i:2}. [{post['score']:>4}] {post['title'][:65]}")
        
        scraper.save_to_csv(recent_posts, 'sf_recent_posts.csv')
        scraper.save_to_json(recent_posts, 'sf_recent_posts.json')
        
        # Analyze keywords
        print(f"\n Top Keywords in Recent Posts:")
        keywords = scraper.get_top_keywords(recent_posts, top_n=15)
        for i, (word, count) in enumerate(keywords.items(), 1):
            print(f"{i:2}. {word:.<25} {count:>3} mentions")
    
    # ========================================================================
    # PART 3: Search for specific city issues
    # ========================================================================
    print("\n" + "="*80)
    print(f"PART 3: Searching for Specific Issues in {CITY_NAME.title()}")
    print("="*80)
    
    # Use bulk search for efficiency (limit to top 5 subreddits to avoid too many requests)
    issue_posts = scraper.bulk_search_issues(
        subreddits=sf_subreddits[:5],  # Use top 5 dynamically selected subreddits
        keywords=issue_keywords[:6],   # First 6 keywords
        posts_per_keyword=15
    )
    
    if issue_posts:
        # Remove duplicates by ID
        unique_posts = {post['id']: post for post in issue_posts}.values()
        unique_posts = list(unique_posts)
        
        print(f"\n Issue Search Results:")
        print(f"   Total unique posts found: {len(unique_posts)}")
        
        scraper.save_to_csv(unique_posts, 'sf_issues_posts.csv')
        scraper.save_to_json(unique_posts, 'sf_issues_posts.json')
        
        # Top issues found
        print(f"\n Top Posts About SF Issues:")
        sorted_posts = sorted(unique_posts, key=lambda x: x['score'], reverse=True)
        for i, post in enumerate(sorted_posts[:10], 1):
            print(f"{i:2}. [{post['score']:>4}] {post['title'][:65]}")
            print(f"     r/{post['subreddit']} • {post['url']}")
    
    # ========================================================================
    # PART 4: Get top posts from the past week
    # ========================================================================
    print("\n" + "="*80)
    print(f"PART 4: Top Posts from the Past Week (r/{sf_subreddits[0]})")
    print("="*80)
    
    weekly_top = scraper.get_subreddit_posts(
        sf_subreddits[0],  # Use the largest subreddit
        sort='top', 
        limit=50, 
        time_filter='week'
    )
    
    if weekly_top:
        print(f"\n Top Weekly Posts:")
        for i, post in enumerate(weekly_top[:10], 1):
            print(f"{i:2}. [{post['score']:>4} {post['num_comments']:>3}] {post['title'][:60]}")
        
        scraper.save_to_csv(weekly_top, 'sf_weekly_top.csv')
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print(" SCRAPING COMPLETE!")
    print("="*80)
    
    print(f"\n Scraped {len(sf_subreddits)} subreddits for: {CITY_NAME.title()}")
    print(f"\n Files created:")
    print("   • sf_subreddits_all.json - All subreddits found")
    print("   • sf_subreddits_selected.json - Selected subreddits")
    print(f"   • sf_recent_posts.csv/json - Recent posts from r/{sf_subreddits[0]}")
    print("   • sf_issues_posts.csv/json - Posts about specific issues")
    print("   • sf_weekly_top.csv - Top posts from past week")
    
    scraper.print_stats()
    
    print("\n Next steps:")
    print("   • Open CSV files in Excel/Google Sheets")
    print("   • Run: python analyze_issues.py (for analysis)")
    print(f"   • To scrape a different city, change CITY_NAME at line ~289")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Scraping interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n Error: {e}")
        sys.exit(1)
