#!/usr/bin/env python3
"""
Analyze scraped Reddit data for San Francisco issues
Identifies trends, sentiment, and key topics
"""

import json
import csv
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict
import re


class RedditDataAnalyzer:
    """Analyze scraped Reddit data for patterns and insights"""
    
    def __init__(self):
        self.posts = []
        
    def load_csv(self, filename: str) -> List[Dict]:
        """Load data from CSV file"""
        data = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            print(f"Loaded {len(data)} items from {filename}")
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
        
        return data
    
    def load_json(self, filename: str):
        """Load data from JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded data from {filename}")
            return data
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return None
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return None
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from text"""
        # Common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Convert to lowercase and split
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [w for w in words if len(w) >= min_length and w not in stop_words]
        
        return keywords
    
    def identify_issue_categories(self, posts: List[Dict]) -> Dict[str, int]:
        """Categorize posts by SF issues"""
        issue_patterns = {
            'homelessness': ['homeless', 'unhoused', 'encampment', 'tent', 'vagrant'],
            'housing': ['housing', 'rent', 'apartment', 'lease', 'landlord', 'eviction', 'affordable'],
            'crime': ['crime', 'theft', 'robbery', 'stolen', 'burglary', 'break-in', 'mugged'],
            'transportation': ['bart', 'muni', 'traffic', 'transit', 'subway', 'bus', 'caltrain'],
            'cost_of_living': ['expensive', 'cost', 'price', 'afford', 'salary', 'wage'],
            'gentrification': ['gentrif', 'displacement', 'tech', 'startup', 'yuppie'],
            'public_safety': ['safety', 'police', 'cops', 'security', 'violence'],
            'infrastructure': ['street', 'road', 'sidewalk', 'pothole', 'construction', 'repair'],
            'business': ['restaurant', 'store', 'shop', 'business', 'closing', 'opening'],
            'politics': ['mayor', 'supervisor', 'council', 'election', 'vote', 'policy']
        }
        
        category_counts = defaultdict(int)
        
        for post in posts:
            title = post.get('title', '').lower()
            text = post.get('selftext', '').lower() if 'selftext' in post else ''
            combined = f"{title} {text}"
            
            for category, patterns in issue_patterns.items():
                if any(pattern in combined for pattern in patterns):
                    category_counts[category] += 1
        
        return dict(category_counts)
    
    def analyze_sentiment_keywords(self, posts: List[Dict]) -> Dict[str, int]:
        """Simple sentiment analysis based on keywords"""
        positive_words = [
            'love', 'great', 'amazing', 'beautiful', 'excellent', 'wonderful',
            'fantastic', 'best', 'good', 'nice', 'happy', 'enjoy', 'perfect'
        ]
        
        negative_words = [
            'hate', 'worst', 'terrible', 'awful', 'disgusting', 'horrible',
            'bad', 'worse', 'sad', 'angry', 'frustrated', 'disappointed', 'sucks'
        ]
        
        sentiment = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for post in posts:
            title = post.get('title', '').lower()
            
            pos_count = sum(1 for word in positive_words if word in title)
            neg_count = sum(1 for word in negative_words if word in title)
            
            if pos_count > neg_count:
                sentiment['positive'] += 1
            elif neg_count > pos_count:
                sentiment['negative'] += 1
            else:
                sentiment['neutral'] += 1
        
        return sentiment
    
    def get_top_posts(self, posts: List[Dict], n: int = 10) -> List[Dict]:
        """Get top posts by score"""
        try:
            sorted_posts = sorted(posts, key=lambda x: int(x.get('score', 0)), reverse=True)
            return sorted_posts[:n]
        except:
            return posts[:n]
    
    def analyze_posting_times(self, posts: List[Dict]) -> Dict:
        """Analyze when posts are created"""
        hours = defaultdict(int)
        days = defaultdict(int)
        
        for post in posts:
            created = post.get('created_utc', '')
            if created:
                try:
                    if isinstance(created, str):
                        dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    else:
                        dt = datetime.fromtimestamp(float(created))
                    
                    hours[dt.hour] += 1
                    days[dt.strftime('%A')] += 1
                except:
                    pass
        
        return {
            'by_hour': dict(hours),
            'by_day': dict(days)
        }
    
    def get_most_active_users(self, posts: List[Dict], n: int = 10) -> Dict[str, int]:
        """Find most active posters"""
        user_counts = Counter()
        
        for post in posts:
            author = post.get('author', '')
            if author and author != '[deleted]':
                user_counts[author] += 1
        
        return dict(user_counts.most_common(n))
    
    def generate_report(self, posts: List[Dict]) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'total_posts': len(posts),
            'date_generated': datetime.now().isoformat(),
            'issue_categories': self.identify_issue_categories(posts),
            'sentiment': self.analyze_sentiment_keywords(posts),
            'top_posts': self.get_top_posts(posts, n=10),
            'posting_times': self.analyze_posting_times(posts),
            'most_active_users': self.get_most_active_users(posts, n=10),
            'subreddit_distribution': self._get_subreddit_distribution(posts)
        }
        
        # Extract top keywords from titles
        all_keywords = []
        for post in posts:
            all_keywords.extend(self.extract_keywords(post.get('title', '')))
        
        keyword_counts = Counter(all_keywords)
        report['top_keywords'] = dict(keyword_counts.most_common(30))
        
        return report
    
    def _get_subreddit_distribution(self, posts: List[Dict]) -> Dict[str, int]:
        """Get distribution of posts across subreddits"""
        subreddit_counts = Counter()
        
        for post in posts:
            subreddit = post.get('subreddit', '')
            if subreddit:
                subreddit_counts[subreddit] += 1
        
        return dict(subreddit_counts)
    
    def print_report(self, report: Dict):
        """Print formatted report"""
        print("\n" + "="*80)
        print("SAN FRANCISCO REDDIT ISSUES ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nTotal Posts Analyzed: {report['total_posts']}")
        print(f"Report Generated: {report['date_generated']}")
        
        print("\n" + "-"*80)
        print("ISSUE CATEGORIES")
        print("-"*80)
        for category, count in sorted(report['issue_categories'].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"{category.replace('_', ' ').title():.<40} {count:>5}")
        
        print("\n" + "-"*80)
        print("SENTIMENT ANALYSIS")
        print("-"*80)
        sentiment = report['sentiment']
        total = sum(sentiment.values())
        if total > 0:
            for sent_type, count in sentiment.items():
                percentage = (count / total) * 100
                print(f"{sent_type.title():.<40} {count:>5} ({percentage:>5.1f}%)")
        
        print("\n" + "-"*80)
        print("TOP 10 MOST DISCUSSED KEYWORDS")
        print("-"*80)
        for i, (keyword, count) in enumerate(list(report['top_keywords'].items())[:10], 1):
            print(f"{i:>2}. {keyword:.<35} {count:>5} mentions")
        
        print("\n" + "-"*80)
        print("SUBREDDIT DISTRIBUTION")
        print("-"*80)
        for subreddit, count in sorted(report['subreddit_distribution'].items(),
                                      key=lambda x: x[1], reverse=True):
            print(f"r/{subreddit:.<40} {count:>5} posts")
        
        print("\n" + "-"*80)
        print("TOP 5 POSTS BY SCORE")
        print("-"*80)
        for i, post in enumerate(report['top_posts'][:5], 1):
            score = post.get('score', 0)
            title = post.get('title', '')[:60]
            print(f"{i}. [{score}] {title}")
            print(f"   {post.get('url', '')}")
        
        print("\n" + "-"*80)
        print("POSTING TIME ANALYSIS")
        print("-"*80)
        
        if 'by_day' in report['posting_times']:
            print("\nPosts by Day of Week:")
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                         'Friday', 'Saturday', 'Sunday']
            for day in days_order:
                count = report['posting_times']['by_day'].get(day, 0)
                print(f"  {day:.<30} {count:>5}")
        
        print("\n" + "-"*80)
        print("MOST ACTIVE POSTERS")
        print("-"*80)
        for i, (user, count) in enumerate(report['most_active_users'].items(), 1):
            print(f"{i:>2}. u/{user:.<35} {count:>5} posts")
        
        print("\n" + "="*80 + "\n")
    
    def save_report(self, report: Dict, filename: str):
        """Save report to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {filename}")


def main():
    """Analyze scraped Reddit data"""
    analyzer = RedditDataAnalyzer()
    
    print("San Francisco Reddit Data Analyzer")
    print("="*80)
    
    # Try to load data from multiple possible sources
    data_files = [
        '/mnt/user-data/outputs/sf_issues_posts.csv',
        '/mnt/user-data/outputs/sf_recent_posts.csv',
        '/mnt/user-data/outputs/sf_issues_posts_praw.csv',
        '/mnt/user-data/outputs/sf_recent_posts_praw.csv'
    ]
    
    all_posts = []
    for filename in data_files:
        posts = analyzer.load_csv(filename)
        if posts:
            all_posts.extend(posts)
    
    if not all_posts:
        print("\nNo data files found. Please run one of the scrapers first:")
        print("  - python reddit_sf_scraper.py")
        print("  - python praw_scraper.py")
        return
    
    print(f"\nTotal posts loaded: {len(all_posts)}")
    
    # Generate report
    print("\nGenerating analysis report...")
    report = analyzer.generate_report(all_posts)
    
    # Print report
    analyzer.print_report(report)
    
    # Save report
    output_file = '/mnt/user-data/outputs/sf_analysis_report.json'
    analyzer.save_report(report, output_file)
    
    print(f"\nFull report saved to: {output_file}")


if __name__ == "__main__":
    main()
