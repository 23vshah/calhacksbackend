#!/usr/bin/env python3
"""
Test optimized Reddit agent with rate limiting
"""

import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_optimized_reddit():
    """Test the rate-limit optimized Reddit agent"""
    
    from app.services.agents.reddit_agent import RedditAgent, RedditTask
    
    logger.info("Testing optimized Reddit agent with rate limiting...")
    
    # Test with conservative parameters
    reddit_agent = RedditAgent()
    reddit_task = RedditTask(
        task_id="test_optimized_001",
        agent_id="reddit_agent",
        data={},
        city="San Francisco",
        keywords=["homeless", "housing", "safety"],  # Only 3 keywords
        max_subreddits=2,  # Limit to 2 subreddits
        max_posts_per_subreddit=5  # Limit to 5 posts per subreddit
    )
    
    try:
        logger.info("Starting optimized Reddit agent...")
        reddit_result = await reddit_agent.execute(reddit_task)
        
        logger.info(f"Reddit Agent Result: {reddit_result.status}")
        logger.info(f"Subreddits found: {reddit_result.subreddits_found}")
        logger.info(f"Posts analyzed: {reddit_result.posts_analyzed}")
        logger.info(f"Issues identified: {len(reddit_result.issues_identified)}")
        
        if reddit_result.issues_identified:
            logger.info("Sample issues found:")
            for i, issue in enumerate(reddit_result.issues_identified[:3], 1):
                logger.info(f"  {i}. {issue.get('title', 'No title')} (Severity: {issue.get('severity', 'Unknown')})")
        
        logger.info("Optimized Reddit agent test completed successfully!")
        
    except Exception as e:
        logger.error(f"Optimized Reddit agent failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_optimized_reddit())
