#!/usr/bin/env python3
"""
Test LLM-powered agents with smart sampling and relationship detection
"""

import asyncio
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_llm_agents():
    """Test the LLM-powered agents"""
    
    # Import agents
    from app.services.agents.reddit_agent import RedditAgent, RedditTask
    from app.services.agents.sf311_agent import SF311Agent, SF311Task
    from app.services.agents.knowledge_graph_agent import KnowledgeGraphAgent, KnowledgeGraphTask
    
    logger.info("Testing LLM-powered agents...")
    
    # Test Reddit Agent
    logger.info("Testing Reddit Agent...")
    reddit_agent = RedditAgent()
    reddit_task = RedditTask(
        task_id="test_reddit_001",
        agent_id="reddit_agent",
        data={},
        city="San Francisco",
        keywords=["homeless", "housing", "safety"],
        max_subreddits=5,
        max_posts_per_subreddit=10
    )
    
    reddit_result = None
    try:
        reddit_result = await reddit_agent.execute(reddit_task)
        logger.info(f"Reddit Agent Result: {reddit_result.status}")
        logger.info(f"Issues found: {len(reddit_result.data.get('issues', []))}")
        
        if reddit_result.data.get('issues'):
            for issue in reddit_result.data['issues'][:20]:  # Show first 3
                logger.info(f"  - {issue.get('title', 'No title')} (Source: {issue.get('source', 'Unknown')})")
    except Exception as e:
        logger.error(f"Reddit Agent failed: {str(e)}")
        reddit_result = None
    
    # Test SF311 Agent
    logger.info("\nTesting SF311 Agent...")
    sf311_agent = SF311Agent()
    sf311_task = SF311Task(
        task_id="test_sf311_001",
        agent_id="sf311_agent",
        data={},
        pages=20,  # Use 20 pages of real SF311 data
        filter_types=None,  # No filters - get all issues
        min_severity="low"
    )
    
    try:
        sf311_result = await sf311_agent.execute(sf311_task)
        logger.info(f"SF311 Agent Result: {sf311_result.status}")
        logger.info(f"Issues found: {len(sf311_result.data.get('issues', []))}")
        
        if sf311_result.data.get('issues'):
            for issue in sf311_result.data['issues'][:30]:  # Show first 3
                logger.info(f"  - {issue.get('title', 'No title')} (Source: {issue.get('source', 'Unknown')})")
    except Exception as e:
        logger.error(f"SF311 Agent failed: {str(e)}")
    
    # Test Knowledge Graph Agent with REAL data from agents
    logger.info("\nTesting Knowledge Graph Agent with REAL data...")
    kg_agent = KnowledgeGraphAgent()
    
    # Collect real issues from both agents
    real_issues = []
    
    # Add SF311 issues if available
    if sf311_result.data.get('issues'):
        for issue in sf311_result.data['issues']:
            real_issues.append(issue)
        logger.info(f"Added {len(sf311_result.data['issues'])} real SF311 issues to knowledge graph")
    
    # Add Reddit issues if available (if Reddit agent worked)
    if reddit_result and reddit_result.data.get('issues'):
        for issue in reddit_result.data['issues']:
            real_issues.append(issue)
        logger.info(f"Added {len(reddit_result.data['issues'])} real Reddit issues to knowledge graph")
    
    # If no real issues, create a minimal test set
    if not real_issues:
        logger.warning("No real issues found, creating minimal test set")
        real_issues = [
            {
                "title": "Test Community Issue",
                "description": "Test issue for knowledge graph",
                "source": "test",
                "severity": "medium",
                "location": {"neighborhood": "Test", "coordinates": [37.7749, -122.4194]},
                "created_at": datetime.now(timezone.utc)
            }
        ]
    
    kg_task = KnowledgeGraphTask(
        task_id="test_kg_001",
        agent_id="knowledge_graph_agent",
        data={},
        new_issues=real_issues,
        similarity_threshold=0.6,
        max_clusters=10
    )
    
    try:
        kg_result = await kg_agent.execute(kg_task)
        logger.info(f"Knowledge Graph Agent Result: {kg_result.status}")
        logger.info(f"Relationships found: {len(kg_result.data.get('relationships', []))}")
        logger.info(f"Clusters identified: {len(kg_result.data.get('clusters', []))}")
        
        if kg_result.data.get('relationships'):
            logger.info("Real issue relationships found:")
            for rel in kg_result.data['relationships'][:5]:  # Show first 5
                issue1_title = rel.get('issue_1', {}).get('title', 'Unknown')[:50]
                issue2_title = rel.get('issue_2', {}).get('title', 'Unknown')[:50]
                logger.info(f"  - {rel.get('relationship_type', 'Unknown')} (Strength: {rel.get('strength', 0):.2f})")
                logger.info(f"    '{issue1_title}' ↔ '{issue2_title}'")
        
        if kg_result.data.get('clusters'):
            logger.info("Real issue clusters identified:")
            for cluster in kg_result.data['clusters']:
                cluster_name = cluster.get('name', 'Unknown')
                issue_count = len(cluster.get('issues', []))
                logger.info(f"  - {cluster_name}: {issue_count} related issues")
                
                # Show sample issues in cluster
                for issue in cluster.get('issues', [])[:2]:
                    logger.info(f"    • {issue.get('title', 'No title')[:60]}...")
        
        if kg_result.data.get('insights'):
            logger.info("Knowledge graph insights:")
            for insight in kg_result.data['insights'][:5]:  # Show first 5 insights
                logger.info(f"  - {insight}")
                
    except Exception as e:
        logger.error(f"Knowledge Graph Agent failed: {str(e)}")
    
    logger.info("\nLLM Agent testing completed!")

if __name__ == "__main__":
    asyncio.run(test_llm_agents())
