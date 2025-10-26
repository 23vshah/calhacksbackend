#!/usr/bin/env python3
"""
Test script to run agents and populate the knowledge graph
"""

import asyncio
import json
from datetime import datetime
from app.services.agent_framework import AgentOrchestrator, AgentTask
from app.services.agents.reddit_agent import RedditAgent, RedditTask
from app.services.agents.sf311_agent import SF311Agent, SF311Task
from app.services.agents.knowledge_graph_agent import KnowledgeGraphAgent, KnowledgeGraphTask

async def test_agents():
    """Test the agent system"""
    
    print("=" * 80)
    print("TESTING AGENT SYSTEM")
    print("=" * 80)
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    
    # Register agents
    reddit_agent = RedditAgent("reddit_agent")
    sf311_agent = SF311Agent("sf311_agent")
    kg_agent = KnowledgeGraphAgent("knowledge_graph_agent")
    
    orchestrator.register_agent(reddit_agent)
    orchestrator.register_agent(sf311_agent)
    orchestrator.register_agent(kg_agent)
    
    print(f"Registered {len(orchestrator.agents)} agents")
    
    # Create tasks
    reddit_task = RedditTask(
        task_id="reddit_001",
        agent_id="reddit_agent",
        data={},
        city="San Francisco",
        keywords=["transportation", "housing", "safety"],
        max_subreddits=3,
        max_posts_per_subreddit=5
    )
    
    sf311_task = SF311Task(
        task_id="sf311_001",
        agent_id="sf311_agent",
        data={},
        pages=2,
        filter_types=["Graffiti", "Street Cleaning"]
    )
    
    # Run agents
    print("\nRunning Reddit agent...")
    reddit_result = await orchestrator.run_agent("reddit_agent", reddit_task)
    print(f"Reddit agent result: {reddit_result.status}")
    if reddit_result.status.value == "success":
        print(f"Found {len(reddit_result.data.get('issues', []))} issues")
    
    print("\nRunning SF311 agent...")
    sf311_result = await orchestrator.run_agent("sf311_agent", sf311_task)
    print(f"SF311 agent result: {sf311_result.status}")
    if sf311_result.status.value == "success":
        print(f"Found {len(sf311_result.data.get('issues', []))} issues")
    
    # Run knowledge graph agent if we have data
    if (reddit_result.status.value == "success" and 
        sf311_result.status.value == "success"):
        
        print("\nRunning Knowledge Graph agent...")
        
        # Combine issues from both agents
        all_issues = []
        if reddit_result.data.get('issues'):
            all_issues.extend(reddit_result.data['issues'])
        if sf311_result.data.get('issues'):
            all_issues.extend(sf311_result.data['issues'])
        
        kg_task = KnowledgeGraphTask(
            task_id="kg_001",
            agent_id="knowledge_graph_agent",
            data={},
            new_issues=all_issues
        )
        
        kg_result = await orchestrator.run_agent("knowledge_graph_agent", kg_task)
        print(f"Knowledge Graph agent result: {kg_result.status}")
        if kg_result.status.value == "success":
            print(f"Created {kg_result.data.get('relationships_created', 0)} relationships")
            print(f"Created {kg_result.data.get('clusters_created', 0)} clusters")
    
    # Check database
    print("\nChecking database...")
    from app.database import AsyncSessionLocal
    from sqlalchemy import text
    
    async with AsyncSessionLocal() as db:
        # Check issues
        result = await db.execute(text("SELECT COUNT(*) FROM community_issues"))
        issues_count = result.scalar()
        print(f"Total issues in database: {issues_count}")
        
        # Check relationships
        result = await db.execute(text("SELECT COUNT(*) FROM issue_relationships"))
        relationships_count = result.scalar()
        print(f"Total relationships in database: {relationships_count}")
        
        # Check clusters
        result = await db.execute(text("SELECT COUNT(*) FROM issue_clusters"))
        clusters_count = result.scalar()
        print(f"Total clusters in database: {clusters_count}")
        
        # Check locations
        result = await db.execute(text("SELECT COUNT(*) FROM locations"))
        locations_count = result.scalar()
        print(f"Total locations in database: {locations_count}")

if __name__ == "__main__":
    asyncio.run(test_agents())

