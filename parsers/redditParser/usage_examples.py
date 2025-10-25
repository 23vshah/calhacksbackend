#!/usr/bin/env python3
"""
USAGE EXAMPLES - San Francisco Reddit Scraper
Practical examples showing how to use the scraper for different purposes
"""

from enhanced_scraper import EnhancedRedditScraper

# Initialize the scraper
scraper = EnhancedRedditScraper()

# ============================================================================
# EXAMPLE 1: Quick Start - Get Recent SF Posts
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 1: Get Recent Posts from r/sanfrancisco")
print("="*80)

posts = scraper.get_subreddit_posts('sanfrancisco', sort='hot', limit=50)
scraper.save_to_csv(posts, 'example1_recent_posts.csv')

print(f" Saved {len(posts)} posts to example1_recent_posts.csv")


# ============================================================================
# EXAMPLE 2: Search for Specific Topic
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 2: Search for Homelessness Discussions")
print("="*80)

homelessness_posts = scraper.search_posts(
    subreddit='sanfrancisco',
    query='homelessness',
    sort='top',
    time_filter='month',
    limit=50
)

scraper.save_to_csv(homelessness_posts, 'example2_homelessness.csv')

print(f" Found {len(homelessness_posts)} posts about homelessness")
print("\nTop 5 posts:")
for i, post in enumerate(homelessness_posts[:5], 1):
    print(f"{i}. [{post['score']}] {post['title']}")


# ============================================================================
# EXAMPLE 3: Get Lots of Data (Using Pagination)
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 3: Get 250 Posts Using Pagination")
print("="*80)

many_posts = scraper.get_multiple_pages(
    subreddit='sanfrancisco',
    sort='hot',
    total_posts=250
)

scraper.save_to_csv(many_posts, 'example3_many_posts.csv')

print(f" Retrieved {len(many_posts)} posts total")


# ============================================================================
# EXAMPLE 4: Search Multiple Subreddits for Housing Issues
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 4: Housing Crisis Across Multiple Subreddits")
print("="*80)

housing_posts = scraper.bulk_search_issues(
    subreddits=['sanfrancisco', 'bayarea', 'SFBayHousing'],
    keywords=['rent', 'housing crisis', 'eviction', 'affordable housing'],
    posts_per_keyword=20
)

# Remove duplicates
unique_housing = {post['id']: post for post in housing_posts}.values()
unique_housing = list(unique_housing)

scraper.save_to_csv(unique_housing, 'example4_housing_crisis.csv')

print(f" Found {len(unique_housing)} unique posts about housing")


# ============================================================================
# EXAMPLE 5: Track Crime Reports
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 5: Recent Crime Discussions")
print("="*80)

crime_posts = scraper.search_posts(
    subreddit='sanfrancisco',
    query='crime OR theft OR robbery OR stolen',
    sort='new',  # Most recent first
    limit=50
)

scraper.save_to_csv(crime_posts, 'example5_crime_reports.csv')

print(f" Found {len(crime_posts)} recent crime-related posts")


# ============================================================================
# EXAMPLE 6: Find Best Restaurants/Places (Top Rated)
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 6: Best Restaurants (Top Posts This Year)")
print("="*80)

restaurant_posts = scraper.search_posts(
    subreddit='sanfrancisco',
    query='best restaurant OR food recommendation',
    sort='top',
    time_filter='year',
    limit=30
)

scraper.save_to_csv(restaurant_posts, 'example6_restaurants.csv')

print(f" Found {len(restaurant_posts)} restaurant recommendation posts")
print("\nTop 3 posts:")
for i, post in enumerate(restaurant_posts[:3], 1):
    print(f"{i}. [{post['score']}] {post['title']}")


# ============================================================================
# EXAMPLE 7: Track BART/Transportation Issues
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 7: BART and Transportation Discussions")
print("="*80)

transit_posts = scraper.bulk_search_issues(
    subreddits=['sanfrancisco', 'bayarea'],
    keywords=['BART', 'Muni', 'Caltrain', 'public transit'],
    posts_per_keyword=15
)

scraper.save_to_csv(transit_posts, 'example7_transit.csv')

print(f" Found {len(transit_posts)} transportation-related posts")


# ============================================================================
# EXAMPLE 8: Analyze Keywords from Posts
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 8: Find Trending Keywords")
print("="*80)

recent = scraper.get_subreddit_posts('sanfrancisco', limit=100)
keywords = scraper.get_top_keywords(recent, top_n=20)

print("\nTop 20 Keywords:")
for i, (word, count) in enumerate(keywords.items(), 1):
    print(f"{i:2}. {word:.<25} {count:>3} mentions")

scraper.save_to_json(keywords, 'example8_keywords.json')


# ============================================================================
# EXAMPLE 9: Weekly Top Posts
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 9: Top Posts from the Past Week")
print("="*80)

weekly_top = scraper.get_subreddit_posts(
    subreddit='sanfrancisco',
    sort='top',
    time_filter='week',
    limit=25
)

scraper.save_to_csv(weekly_top, 'example9_weekly_top.csv')

print("\nTop 5 posts this week:")
for i, post in enumerate(weekly_top[:5], 1):
    print(f"{i}. [{post['score']:>4} {post['num_comments']:>3}] {post['title'][:60]}")


# ============================================================================
# EXAMPLE 10: Monitor Specific Neighborhoods
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 10: Search Specific SF Neighborhoods")
print("="*80)

neighborhoods = ['Mission', 'Tenderloin', 'Castro', 'Sunset', 'Richmond']
neighborhood_posts = []

for hood in neighborhoods:
    posts = scraper.search_posts('sanfrancisco', hood, limit=10)
    neighborhood_posts.extend(posts)
    print(f"   {hood}: {len(posts)} posts")

scraper.save_to_csv(neighborhood_posts, 'example10_neighborhoods.csv')

print(f"\n Total: {len(neighborhood_posts)} posts about neighborhoods")


# ============================================================================
# EXAMPLE 11: Compare Multiple Cities
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 11: Compare SF with Other Bay Area Cities")
print("="*80)

bay_cities = {
    'San Francisco': 'sanfrancisco',
    'Oakland': 'oakland',
    'San Jose': 'sanjose',
    'Berkeley': 'berkeley'
}

for city_name, subreddit in bay_cities.items():
    posts = scraper.get_subreddit_posts(subreddit, sort='hot', limit=25)
    scraper.save_to_csv(posts, f'example11_{subreddit}.csv')
    print(f"   {city_name}: {len(posts)} posts saved")

print(" Compare the CSV files to see differences between cities")


# ============================================================================
# EXAMPLE 12: Track Specific User Topics (e.g., Moving to SF)
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 12: Moving to SF Questions")
print("="*80)

moving_posts = scraper.search_posts(
    subreddit='sanfrancisco',
    query='moving to sf OR relocating OR new to san francisco',
    sort='new',
    limit=50
)

scraper.save_to_csv(moving_posts, 'example12_moving_to_sf.csv')

print(f" Found {len(moving_posts)} posts from people moving to SF")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print(" ALL EXAMPLES COMPLETE!")
print("="*80)

print("\n Files created:")
print("   • example1_recent_posts.csv - Recent posts")
print("   • example2_homelessness.csv - Homelessness discussions")
print("   • example3_many_posts.csv - 250 posts via pagination")
print("   • example4_housing_crisis.csv - Housing issues")
print("   • example5_crime_reports.csv - Crime discussions")
print("   • example6_restaurants.csv - Restaurant recommendations")
print("   • example7_transit.csv - Transportation issues")
print("   • example8_keywords.json - Trending keywords")
print("   • example9_weekly_top.csv - Top posts this week")
print("   • example10_neighborhoods.csv - Neighborhood discussions")
print("   • example11_*.csv - Multiple Bay Area cities")
print("   • example12_moving_to_sf.csv - Moving questions")

print("\n Tips:")
print("   • Open CSV files in Excel or Google Sheets")
print("   • Modify these examples for your needs")
print("   • Combine multiple searches for comprehensive data")
print("   • Run analyze_issues.py on any CSV file")

scraper.print_stats()

print("\n" + "="*80 + "\n")
