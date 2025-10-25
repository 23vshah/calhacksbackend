#!/usr/bin/env python3
"""
Test SF311 agent with real data
"""

import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_sf311():
    """Test SF311 agent with real data"""
    
    from app.services.agents.sf311_agent import SF311Agent, SF311Task
    
    logger.info("Testing SF311 agent with REAL data...")
    
    # Test with real SF311 data
    sf311_agent = SF311Agent()
    sf311_task = SF311Task(
        task_id="test_real_sf311_001",
        agent_id="sf311_agent",
        data={},
        pages=20,  # Test with 20 pages for more diverse data
        filter_types=["Graffiti", "Street Cleaning", "Blocked Driveway", "Pothole", "Broken Streetlight"],
        min_severity="low"
    )
    
    try:
        logger.info("Starting SF311 agent with real data...")
        sf311_result = await sf311_agent.execute(sf311_task)
        
        logger.info(f"SF311 Agent Result: {sf311_result.status}")
        logger.info(f"Total requests fetched: {sf311_result.total_requests}")
        logger.info(f"Issues found: {len(sf311_result.data.get('issues', []))}")
        
        if sf311_result.data.get('issues'):
            logger.info("Real SF311 issues found:")
            for i, issue in enumerate(sf311_result.data['issues'], 1):
                logger.info(f"  {i}. {issue.get('title', 'No title')} (Source: {issue.get('source', 'Unknown')})")
                logger.info(f"     Description: {issue.get('description', 'No description')[:100]}...")
                logger.info(f"     Location: {issue.get('location', {}).get('neighborhood', 'Unknown')}")
                logger.info("")
        
        logger.info("Real SF311 agent test completed successfully!")
        
    except Exception as e:
        logger.error(f"Real SF311 agent failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_real_sf311())
