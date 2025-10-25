#!/usr/bin/env python3
"""
API 2: Scrape Subreddits
Scrapes content from a list of subreddits

Usage as script:
    python scrape_subreddits.py
    (Automatically loads subreddits from subreddits_found.json)

Usage as module:
    from scrape_subreddits import scrape_subreddits
    results = scrape_subreddits(subreddit_names=['sanfrancisco', 'bayarea'])
"""

import requests
import json
import csv
import time
from datetime import datetime
from typing import List, Dict, Optional
from collections import Counter
import sys
import os


class SubredditScraper:
    """Scrapes content from specified subreddits"""
    
    def __init__(self):
        self.base_url = "https://www.reddit.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_delay = 1.0  # 1 second between requests
    
    def _rate_limit(self):
        """Rate limiting to respect Reddit's API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1
        if self.request_count % 5 == 0:
            time.sleep(2)
    
    def _make_request(self, url: str, params: dict = None) -> Optional[dict]:
        """Make a request with error handling"""
        self._rate_limit()
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None
    
    def get_subreddit_posts(self, subreddit: str, sort: str = 'hot', 
                           limit: int = 100, time_filter: str = 'all') -> List[Dict]:
        """Get posts from a subreddit"""
        print(f" Getting {sort} posts from r/{subreddit}...")
        
        url = f"{self.base_url}/r/{subreddit}/{sort}.json"
        params = {
            'limit': min(limit, 100),
            't': time_filter
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
                'upvote_ratio': post_data.get('upvote_ratio', 0),
                'num_comments': post_data.get('num_comments', 0),
                'created_utc': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat(),
                'url': f"https://www.reddit.com{post_data.get('permalink')}",
                'selftext': post_data.get('selftext', '')[:1000],
                'link_flair_text': post_data.get('link_flair_text'),
                'subreddit': post_data.get('subreddit')
            })
        
        print(f"   Retrieved {len(posts)} posts")
        return posts
    
    def search_posts(self, subreddit: str, query: str, sort: str = 'top',
                    limit: int = 25, time_filter: str = 'month') -> List[Dict]:
        """Search for posts in a subreddit"""
        print(f" Searching r/{subreddit} for '{query}'...")
        
        url = f"{self.base_url}/r/{subreddit}/search.json"
        params = {
            'q': query,
            'restrict_sr': 'on',
            'sort': sort,
            'limit': min(limit, 100),
            't': time_filter
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
                'subreddit': post_data.get('subreddit'),
                'search_query': query
            })
        
        print(f"   Found {len(posts)} posts")
        return posts
    
    def bulk_search(self, subreddits: List[str], keywords: List[str],
                   posts_per_keyword: int = 15) -> List[Dict]:
        """Search multiple subreddits for multiple keywords"""
        print(f"\nBulk search: {len(subreddits)} subreddits × {len(keywords)} keywords")
        
        all_posts = []
        total = len(subreddits) * len(keywords)
        current = 0
        
        for subreddit in subreddits:
            for keyword in keywords:
                current += 1
                print(f"[{current}/{total}] ", end='')
                
                posts = self.search_posts(
                    subreddit=subreddit,
                    query=keyword,
                    sort='top',
                    time_filter='month',
                    limit=posts_per_keyword
                )
                
                all_posts.extend(posts)
                time.sleep(1)
        
        return all_posts
    
    def scrape_subreddits(self, subreddit_names: List[str], 
                         issue_keywords: List[str] = None,
                         get_recent: bool = True,
                         get_weekly_top: bool = True,
                         search_issues: bool = True) -> Dict:
        """
        Main scraping function
        
        Args:
            subreddit_names: List of subreddit names to scrape
            issue_keywords: Keywords to search for (default: common issues)
            get_recent: Whether to get recent posts from largest sub
            get_weekly_top: Whether to get top posts from past week
            search_issues: Whether to search for specific issues
            
        Returns:
            Dictionary with all scraped data
        """
        if not subreddit_names:
            raise ValueError("No subreddits provided")
        
        # Default issue keywords if not provided
        if issue_keywords is None:
            issue_keywords = [
                'homelessness',
                'housing crisis',
                'crime',
                'public safety',
                'traffic',
                'cost of living'
            ]
        
        print("\n" + "="*80)
        print(" SCRAPING SUBREDDITS")
        print("="*80)
        print(f"Subreddits: {', '.join(['r/'+s for s in subreddit_names])}")
        print(f"Keywords: {', '.join(issue_keywords)}")
        print("="*80 + "\n")
        
        results = {
            'config': {
                'subreddits': subreddit_names,
                'keywords': issue_keywords,
                'timestamp': datetime.now().isoformat()
            },
            'data': {}
        }
        
        # 1. Get recent posts from largest subreddit
        if get_recent and subreddit_names:
            print("\n" + "-"*80)
            print("STEP 1: Recent Posts from Main Subreddit")
            print("-"*80)
            
            recent_posts = self.get_subreddit_posts(
                subreddit=subreddit_names[0],
                sort='hot',
                limit=100
            )
            
            results['data']['recent_posts'] = recent_posts
            
            if recent_posts:
                print(f"\n Top 5 posts:")
                for i, post in enumerate(recent_posts[:5], 1):
                    print(f"{i}. [{post['score']:>4}] {post['title'][:65]}")
        
        # 2. Get weekly top posts
        if get_weekly_top and subreddit_names:
            print("\n" + "-"*80)
            print("STEP 2: Top Posts from Past Week")
            print("-"*80)
            
            weekly_top = self.get_subreddit_posts(
                subreddit=subreddit_names[0],
                sort='top',
                limit=50,
                time_filter='week'
            )
            
            results['data']['weekly_top'] = weekly_top
            
            if weekly_top:
                print(f"\n Top 3 this week:")
                for i, post in enumerate(weekly_top[:3], 1):
                    print(f"{i}. [{post['score']:>4}] {post['title'][:65]}")
        
        # 3. Search for specific issues
        if search_issues:
            print("\n" + "-"*80)
            print("STEP 3: Searching for Specific Issues")
            print("-"*80)
            
            issue_posts = self.bulk_search(
                subreddits=subreddit_names,
                keywords=issue_keywords,
                posts_per_keyword=15
            )
            
            # Remove duplicates
            unique_issues = {post['id']: post for post in issue_posts}.values()
            unique_issues = list(unique_issues)
            
            results['data']['issue_posts'] = unique_issues
            
            print(f"\nFound {len(unique_issues)} unique posts about issues")
            
            # Show top issues
            if unique_issues:
                sorted_issues = sorted(unique_issues, key=lambda x: x['score'], reverse=True)
                print(f"\n Top 3 issue discussions:")
                for i, post in enumerate(sorted_issues[:3], 1):
                    print(f"{i}. [{post['score']:>4}] {post['title'][:65]}")
                    print(f"   Search: '{post['search_query']}' in r/{post['subreddit']}")
        
        # Calculate stats
        results['stats'] = {
            'total_posts': sum(len(v) for k, v in results['data'].items() if isinstance(v, list)),
            'recent_posts_count': len(results['data'].get('recent_posts', [])),
            'weekly_top_count': len(results['data'].get('weekly_top', [])),
            'issue_posts_count': len(results['data'].get('issue_posts', []))
        }
        
        return results
    
    def save_results(self, results: Dict, prefix: str = 'scraped'):
        """Save results to CSV and JSON files"""
        print("\n" + "="*80)
        print(" SAVING RESULTS")
        print("="*80)
        
        # Save full results as JSON
        json_file = f"{prefix}_all_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON: {json_file}")
        
        # Save each dataset as CSV
        for key, data in results['data'].items():
            if isinstance(data, list) and data:
                csv_file = f"{prefix}_{key}.csv"
                
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                
                print(f"Saved CSV: {csv_file} ({len(data)} posts)")
        
        print("\n" + "="*80)


def scrape_subreddits(subreddit_names: List[str], issue_keywords: List[str] = None) -> Dict:
    """
    Convenience function for use as module
    
    Example:
        from scrape_subreddits import scrape_subreddits
        results = scrape_subreddits(['sanfrancisco', 'bayarea'])
    """
    scraper = SubredditScraper()
    return scraper.scrape_subreddits(subreddit_names, issue_keywords)


def main():
    """Run as standalone script - loads subreddits from find_subreddits.py output"""
    
    # Try to load subreddits from previous step
    input_file = 'subreddits_found.json'
    
    if os.path.exists(input_file):
        print(f" Loading subreddits from: {input_file}")
        with open(input_file, 'r') as f:
            discovery_results = json.load(f)
        
        # Get top 5 subreddits
        subreddit_names = discovery_results['subreddit_names'][:5]
        city = discovery_results['query']['city']
        
        print(f"Loaded {len(subreddit_names)} subreddits for: {city}")
    else:
        print(f"Warning: {input_file} not found!")
        print(f"   Run 'python find_subreddits.py' first, or edit this script.")
        print(f"\n   Using default subreddits...")
        
        # Fallback to default
        subreddit_names = ['sanfrancisco', 'bayarea', 'AskSF', 'oakland', 'SFBayHousing']
        city = "san francisco"
    
    # Issue keywords to search for
    issue_keywords = [
        'homelessness',
        'housing crisis',
        'crime',
        'public safety',
        'BART',
        'traffic',
        'cost of living',
        'rent'
    ]
    
    print(f"\n Will scrape top {len(subreddit_names)} subreddits:")
    for i, name in enumerate(subreddit_names, 1):
        print(f"   {i}. r/{name}")
    
    # Run scraping
    scraper = SubredditScraper()
    results = scraper.scrape_subreddits(
        subreddit_names=subreddit_names,
        issue_keywords=issue_keywords[:6]  # Use first 6 to avoid too many requests
    )
    
    # Save results
    scraper.save_results(results, prefix='scraped')
    
    # Summary
    print("\n" + "="*80)
    print("SCRAPING COMPLETE")
    print("="*80)
    print(f"\n Summary:")
    print(f"   Total posts collected: {results['stats']['total_posts']:,}")
    print(f"   Recent posts: {results['stats']['recent_posts_count']:,}")
    print(f"   Weekly top: {results['stats']['weekly_top_count']:,}")
    print(f"   Issue posts: {results['stats']['issue_posts_count']:,}")
    
    print(f"\n Files created:")
    print(f"   • scraped_all_data.json - Complete dataset")
    print(f"   • scraped_recent_posts.csv - Recent posts")
    print(f"   • scraped_weekly_top.csv - Top posts this week")
    print(f"   • scraped_issue_posts.csv - Posts about specific issues")
    
    print(f"\n Next step:")
    print(f"   python analyze_issues.py")
    print(f"   (Analyze the scraped data)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
