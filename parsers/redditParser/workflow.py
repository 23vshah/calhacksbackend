import json
from find_subreddits import SubredditFinder
from scrape_subreddits import SubredditScraper


def run_workflow(city: str, 
                 keywords: list = None,
                 min_subscribers: int = 1000,
                 min_relevance_score: float = 20.0,
                 max_subreddits_to_find: int = 10,
                 subreddits_to_scrape: int = 5,
                 issue_keywords: list = None):
    """
    Run the complete agentic workflow
    
    Args:
        city: City name to search for
        keywords: Search keywords (defaults to city name)
        min_subscribers: Minimum subscriber count for subreddits
        min_relevance_score: Minimum relevance score (0-100)
        max_subreddits_to_find: How many subreddits to discover
        subreddits_to_scrape: How many (top N) to actually scrape
        issue_keywords: Topics to search for in posts
    
    Returns:
        Dictionary with complete results from both steps
    """
    
    if keywords is None:
        keywords = [city]
    
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
    print("🤖 AGENTIC WORKFLOW: REDDIT CITY SCRAPER")
    print("="*80)
    print(f"City: {city}")
    print(f"Pipeline: Find {max_subreddits_to_find} subreddits → Scrape top {subreddits_to_scrape}")
    print("="*80 + "\n")
    
    # ========================================================================
    # AGENT 1: Find Subreddits
    # ========================================================================
    print("\n" + ""*40)
    print("AGENT 1: SUBREDDIT DISCOVERY")
    print(""*40 + "\n")
    
    finder = SubredditFinder()
    discovery_results = finder.find_subreddits(
        city=city,
        keywords=keywords,
        min_subscribers=min_subscribers,
        max_results=max_subreddits_to_find,
        min_relevance_score=min_relevance_score
    )
    
    # Save intermediate results
    finder.save_results(discovery_results, 'subreddits_found.json')
    
    # Extract subreddit names for next agent
    discovered_subreddits = discovery_results['subreddit_names']
    
    if not discovered_subreddits:
        print(" No subreddits found! Cannot proceed to scraping.")
        return {
            'success': False,
            'discovery': discovery_results,
            'scraping': None
        }
    
    print(f"\n Agent 1 complete: Found {len(discovered_subreddits)} subreddits")
    print(f" Passing to Agent 2: Top {subreddits_to_scrape} subreddits")
    
    # Select top N for scraping
    selected_for_scraping = discovered_subreddits[:subreddits_to_scrape]
    
    # ========================================================================
    # AGENT 2: Scrape Content
    # ========================================================================
    print("\n" + "🟢"*40)
    print("AGENT 2: CONTENT SCRAPING")
    print("🟢"*40 + "\n")
    
    print(f" Received {len(selected_for_scraping)} subreddits from Agent 1:")
    for i, name in enumerate(selected_for_scraping, 1):
        print(f"   {i}. r/{name}")
    
    scraper = SubredditScraper()
    scraping_results = scraper.scrape_subreddits(
        subreddit_names=selected_for_scraping,
        issue_keywords=issue_keywords
    )
    
    # Save scraping results
    scraper.save_results(scraping_results, 'scraped')
    
    print(f"\n Agent 2 complete: Scraped {scraping_results['stats']['total_posts']} posts")
    
    # ========================================================================
    # Workflow Complete
    # ========================================================================
    print("\n" + "="*80)
    print(" WORKFLOW COMPLETE")
    print("="*80)
    
    # Combine results
    workflow_results = {
        'success': True,
        'city': city,
        'discovery': {
            'subreddits_found': len(discovered_subreddits),
            'subreddits': discovery_results['subreddits']
        },
        'scraping': {
            'subreddits_scraped': selected_for_scraping,
            'stats': scraping_results['stats'],
            'data_files': [
                'scraped_all_data.json',
                'scraped_recent_posts.csv',
                'scraped_weekly_top.csv',
                'scraped_issue_posts.csv'
            ]
        },
        'pipeline': {
            'step1': 'Find subreddits',
            'step2': 'Scrape content',
            'subreddits_discovered': len(discovered_subreddits),
            'subreddits_scraped': len(selected_for_scraping),
            'total_posts': scraping_results['stats']['total_posts']
        }
    }
    
    # Save workflow summary
    with open('workflow_results.json', 'w') as f:
        json.dump(workflow_results, f, indent=2)
    
    print("\n Pipeline Summary:")
    print(f"   Agent 1: Found {len(discovered_subreddits)} subreddits")
    print(f"   Agent 2: Scraped {len(selected_for_scraping)} subreddits")
    print(f"   Total posts: {scraping_results['stats']['total_posts']:,}")
    
    print("\n Output Files:")
    print("   Discovery:")
    print("      • subreddits_found.json")
    print("    Scraping:")
    print("      • scraped_all_data.json")
    print("      • scraped_recent_posts.csv")
    print("      • scraped_weekly_top.csv")  
    print("      • scraped_issue_posts.csv")
    print("    Summary:")
    print("      • workflow_results.json")
    
    print("\n" + "="*80 + "\n")
    
    return workflow_results


def main():
    """Run the workflow with configuration"""
    
    # ========================================================================
    # CONFIGURATION - Edit these for your city
    # ========================================================================
    CITY = "san francisco"
    KEYWORDS = ["san francisco", "bay area", "sf"]
    MIN_SUBSCRIBERS = 1000
    MIN_RELEVANCE_SCORE = 20.0   # Filter out irrelevant subreddits (0-100)
    MAX_SUBREDDITS_TO_FIND = 10  # Agent 1 will find this many
    SUBREDDITS_TO_SCRAPE = 5     # Agent 2 will scrape top N of those
    
    ISSUE_KEYWORDS = [
        'homelessness',
        'housing crisis',
        'crime',
        'public safety',
        'BART',
        'traffic',
        'cost of living',
        'rent'
    ]
    
    # Run the workflow
    results = run_workflow(
        city=CITY,
        keywords=KEYWORDS,
        min_subscribers=MIN_SUBSCRIBERS,
        min_relevance_score=MIN_RELEVANCE_SCORE,
        max_subreddits_to_find=MAX_SUBREDDITS_TO_FIND,
        subreddits_to_scrape=SUBREDDITS_TO_SCRAPE,
        issue_keywords=ISSUE_KEYWORDS
    )
    
    if results['success']:
        print(" Workflow completed successfully!")
        print("\n Next steps:")
        print("   • Open CSV files in Excel")
        print("   • Run: python analyze_issues.py")
        print("   • Review workflow_results.json for summary")
    else:
        print(" Workflow failed!")


if __name__ == "__main__":
    main()
