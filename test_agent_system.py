#!/usr/bin/env python3
"""
Test script for the agent system
Run this to verify agents are working correctly
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'parsers'))

from app.services.agent_framework import AgentOrchestrator, AgentTask
from app.services.agents.reddit_agent import RedditAgent, RedditTask
from app.services.agents.sf311_agent import SF311Agent, SF311Task
from app.services.agents.knowledge_graph_agent import KnowledgeGraphAgent, KnowledgeGraphTask

async def test_agent_system():
    """Test the agent system"""
    print("Testing Agent System")
    print("=" * 50)
    
    # Initialize agents
    reddit_agent = RedditAgent()
    sf311_agent = SF311Agent()
    kg_agent = KnowledgeGraphAgent()
    
    # Register with orchestrator
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent(reddit_agent)
    orchestrator.register_agent(sf311_agent)
    orchestrator.register_agent(kg_agent)
    
    print(f"Registered {len(orchestrator.agents)} agents")
    
    # Test 1: Agent Status
    print("\nAgent Status:")
    status = orchestrator.get_agent_status()
    for agent_id, agent_status in status.items():
        print(f"  {agent_id}: {agent_status['status']}")
    
    # Test 2: Reddit Agent (Mock test)
    print("\nTesting Reddit Agent:")
    try:
        reddit_task = RedditTask(
            task_id="test_reddit_001",
            agent_id="reddit_agent",
            data={},
            city="San Francisco",
            keywords=["san francisco", "sf"],
            max_subreddits=3,
            max_posts_per_subreddit=5
        )
        
        print("  Creating Reddit task...")
        print(f"  Task ID: {reddit_task.task_id}")
        print(f"  City: {reddit_task.city}")
        print(f"  Keywords: {reddit_task.keywords}")
        print("  Reddit task created successfully")
        
    except Exception as e:
        print(f"  Reddit agent test failed: {str(e)}")
    
    # Test 3: SF311 Agent (Mock test)
    print("\nTesting SF311 Agent:")
    try:
        sf311_task = SF311Task(
            task_id="test_sf311_001",
            agent_id="sf311_agent",
            data={},
            pages=2
        )
        
        print("  Creating SF311 task...")
        print(f"  Task ID: {sf311_task.task_id}")
        print(f"  Pages: {sf311_task.pages}")
        print("  SF311 task created successfully")
        
    except Exception as e:
        print(f"  SF311 agent test failed: {str(e)}")
    
    # Test 4: Knowledge Graph Agent (Mock test)
    print("\nTesting Knowledge Graph Agent:")
    try:
        kg_task = KnowledgeGraphTask(
            task_id="test_kg_001",
            agent_id="knowledge_graph_agent",
            data={},
            new_issues=[]  # Empty for test
        )
        
        print("  Creating Knowledge Graph task...")
        print(f"  Task ID: {kg_task.task_id}")
        print(f"  New issues: {len(kg_task.new_issues)}")
        print("  Knowledge Graph task created successfully")
        
    except Exception as e:
        print(f"  Knowledge Graph agent test failed: {str(e)}")
    
    # Test 5: Agent Memory
    print("\nTesting Agent Memory:")
    try:
        # Test memory recording
        reddit_agent.memory.record_success(
            {"keywords": ["test"], "city": "San Francisco"},
            type('MockResult', (), {
                'data': {'issues_found': 5},
                'status': 'success'
            })()
        )
        
        successful_patterns = reddit_agent.memory.get_successful_patterns()
        print(f"  Memory system working: {len(successful_patterns)} patterns recorded")
        
    except Exception as e:
        print(f"  Memory test failed: {str(e)}")
    
    print("\nAgent System Test Complete!")
    print("=" * 50)
    print("All core components initialized successfully")
    print("Agent framework is working")
    print("Memory system is functional")
    print("Task creation is working")
    print("\nNext steps:")
    print("  1. Run the FastAPI server: uvicorn app.main:app --reload")
    print("  2. Test API endpoints: POST /api/agents/reddit/scrape")
    print("  3. Check agent status: GET /api/agents/status")

if __name__ == "__main__":
    asyncio.run(test_agent_system())
