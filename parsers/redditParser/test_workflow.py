import sys
import os

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import requests
        print("    requests")
    except ImportError:
        print("    requests - Run: pip install requests")
        return False
    
    try:
        from find_subreddits import SubredditFinder, find_subreddits
        print("    find_subreddits.py")
    except Exception as e:
        print(f"    find_subreddits.py - Error: {e}")
        return False
    
    try:
        from scrape_subreddits import SubredditScraper, scrape_subreddits
        print("    scrape_subreddits.py")
    except Exception as e:
        print(f"    scrape_subreddits.py - Error: {e}")
        return False
    
    try:
        from workflow import run_workflow
        print("    workflow.py")
    except Exception as e:
        print(f"    workflow.py - Error: {e}")
        return False
    
    return True


def test_reddit_access():
    """Test if Reddit is accessible"""
    print("\n🧪 Testing Reddit access...")
    
    import requests
    
    try:
        response = requests.get(
            "https://www.reddit.com/r/test.json",
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10
        )
        
        if response.status_code == 200:
            print("    Reddit is accessible")
            return True
        else:
            print(f"     Reddit returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"    Cannot connect to Reddit")
        print(f"      Error: {e}")
        print(f"      Try: VPN, different network, or check if reddit.com loads in browser")
        return False
    except Exception as e:
        print(f"    Error accessing Reddit: {e}")
        return False


def test_agent1():
    """Test Agent 1: Subreddit Discovery"""
    print("\n🧪 Testing Agent 1 (Subreddit Discovery)...")
    
    try:
        from find_subreddits import SubredditFinder
        
        finder = SubredditFinder()
        
        # Try a simple search
        results = finder.search_subreddits("test", limit=5)
        
        if results and len(results) > 0:
            print(f"    Agent 1 works! Found {len(results)} subreddits")
            print(f"      Example: r/{results[0]['name']}")
            return True
        else:
            print("     Agent 1 ran but found no results")
            return False
            
    except Exception as e:
        print(f"    Agent 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent2():
    """Test Agent 2: Content Scraping"""
    print("\n🧪 Testing Agent 2 (Content Scraping)...")
    
    try:
        from scrape_subreddits import SubredditScraper
        
        scraper = SubredditScraper()
        
        # Try to get a few posts from a test subreddit
        posts = scraper.get_subreddit_posts('test', sort='hot', limit=5)
        
        if posts and len(posts) > 0:
            print(f"    Agent 2 works! Retrieved {len(posts)} posts")
            print(f"      Example: {posts[0]['title'][:50]}...")
            return True
        else:
            print("     Agent 2 ran but got no posts")
            return False
            
    except Exception as e:
        print(f"    Agent 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow():
    """Test complete workflow with minimal config"""
    print("\n🧪 Testing Complete Workflow (minimal)...")
    
    try:
        from workflow import run_workflow
        
        # Run with minimal settings
        results = run_workflow(
            city="test",
            keywords=["test"],
            min_subscribers=1000,
            min_relevance_score=0.0,  # Accept any relevance for test
            max_subreddits_to_find=3,
            subreddits_to_scrape=1,
            issue_keywords=["test"]
        )
        
        if results and results.get('success'):
            print("    Workflow works!")
            print(f"      Found: {results['pipeline']['subreddits_discovered']} subreddits")
            print(f"      Scraped: {results['pipeline']['total_posts']} posts")
            
            # Clean up test files
            import os
            test_files = [
                'subreddits_found.json',
                'scraped_all_data.json',
                'scraped_recent_posts.csv',
                'scraped_weekly_top.csv',
                'scraped_issue_posts.csv',
                'workflow_results.json'
            ]
            for f in test_files:
                if os.path.exists(f):
                    os.remove(f)
            print("      (Test files cleaned up)")
            
            return True
        else:
            print("    Workflow failed")
            return False
            
    except Exception as e:
        print(f"    Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("🧪 AGENTIC WORKFLOW TEST SUITE")
    print("="*80)
    print("This will verify everything works before running the full workflow\n")
    
    results = {
        'imports': False,
        'reddit': False,
        'agent1': False,
        'agent2': False,
        'workflow': False
    }
    
    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print("\n Import test failed. Fix errors above before continuing.")
        sys.exit(1)
    
    # Test 2: Reddit Access
    results['reddit'] = test_reddit_access()
    if not results['reddit']:
        print("\n  Reddit access test failed.")
        print("   The workflow may not work if Reddit is blocked.")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Test 3: Agent 1
    results['agent1'] = test_agent1()
    
    # Test 4: Agent 2
    results['agent2'] = test_agent2()
    
    # Test 5: Complete Workflow
    if results['agent1'] and results['agent2']:
        results['workflow'] = test_workflow()
    else:
        print("\n  Skipping workflow test (agents failed)")
    
    # Summary
    print("\n" + "="*80)
    print(" TEST RESULTS SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = " PASS" if passed else " FAIL"
        print(f"{test_name.upper():.<20} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    
    if all_passed:
        print(" ALL TESTS PASSED!")
        print("\nYou're ready to run the full workflow:")
        print("   python workflow.py")
    else:
        print(" SOME TESTS FAILED")
        print("\nCheck the errors above and:")
        print("   1. Make sure Reddit is accessible (try in browser)")
        print("   2. Install dependencies: pip install requests")
        print("   3. Check for typos in the code")
        print("\nNeed help? Review the error messages above.")
    
    print("="*80 + "\n")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
