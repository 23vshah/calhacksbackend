#!/usr/bin/env python3
"""
Demo showing API endpoint responses
"""

def show_api_responses():
    print("=" * 60)
    print("API ENDPOINT RESPONSES")
    print("=" * 60)
    
    print("\n1. POST /api/agents/reddit/scrape")
    print("   Response:")
    print("   {")
    print('     "task_id": "reddit_20241201_143022",')
    print('     "agent_id": "reddit_agent",')
    print('     "status": "success",')
    print('     "execution_time": 2.3,')
    print('     "insights": [')
    print('       "Found 3 relevant subreddits for San Francisco",')
    print('       "Identified 2 community issues",')
    print('       "Most active subreddit: sftransportation"')
    print('     ]')
    print("   }")
    
    print("\n2. POST /api/agents/311/scrape")
    print("   Response:")
    print("   {")
    print('     "task_id": "sf311_20241201_143025",')
    print('     "agent_id": "sf311_agent",')
    print('     "status": "success",')
    print('     "execution_time": 1.8,')
    print('     "insights": [')
    print('       "Processed 50 311 requests",')
    print('       "Identified 35 community issues",')
    print('       "Top issue type: Graffiti"')
    print('     ]')
    print("   }")
    
    print("\n3. POST /api/agents/run-pipeline")
    print("   Response:")
    print("   {")
    print('     "pipeline_status": "completed",')
    print('     "agents_run": 3,')
    print('     "total_issues_found": 37,')
    print('     "results": [')
    print('       {')
    print('         "agent_id": "reddit_agent",')
    print('         "status": "success",')
    print('         "execution_time": 2.3')
    print('       },')
    print('       {')
    print('         "agent_id": "sf311_agent",')
    print('         "status": "success",')
    print('         "execution_time": 1.8')
    print('       },')
    print('       {')
    print('         "agent_id": "knowledge_graph_agent",')
    print('         "status": "success",')
    print('         "execution_time": 3.1')
    print('       }')
    print('     ]')
    print("   }")
    
    print("\n4. GET /api/agents/status")
    print("   Response:")
    print("   {")
    print('     "agents": {')
    print('       "reddit_agent": {')
    print('         "agent_id": "reddit_agent",')
    print('         "status": "idle",')
    print('         "last_run": "2024-12-01T14:30:25Z",')
    print('         "successful_patterns": 5,')
    print('         "failed_patterns": 1')
    print('       },')
    print('       "sf311_agent": {')
    print('         "agent_id": "sf311_agent",')
    print('         "status": "idle",')
    print('         "last_run": "2024-12-01T14:30:25Z",')
    print('         "successful_patterns": 8,')
    print('         "failed_patterns": 0')
    print('       }')
    print('     },')
    print('     "total_agents": 3')
    print("   }")
    
    print("\n5. POST /api/map/data")
    print("   Response:")
    print("   {")
    print('     "issues": [')
    print('       {')
    print('         "id": "transportation_cluster_001",')
    print('         "type": "Transportation",')
    print('         "severity": "high",')
    print('         "location": [37.7749, -122.4194],')
    print('         "neighborhood": "Mission District",')
    print('         "count": 3,')
    print('         "trend": "increasing",')
    print('         "related_reddit_posts": 2,')
    print('         "311_reports": 1')
    print('       }')
    print('     ],')
    print('     "hotspots": [')
    print('       {')
    print('         "neighborhood": "Mission District",')
    print('         "issue_types": ["transportation", "graffiti"],')
    print('         "severity_score": 0.7,')
    print('         "trend": "worsening",')
    print('         "issue_count": 8')
    print('       }')
    print('     ],')
    print('     "insights": [')
    print('       "Transportation issues concentrated in Mission District",')
    print('       "Graffiti complaints increase 40% after major events"')
    print('     ],')
    print('     "total_issues": 20')
    print("   }")
    
    print("\n6. GET /api/insights/trends")
    print("   Response:")
    print("   {")
    print('     "timeframe_days": 7,')
    print('     "trends": [')
    print('       {')
    print('         "issue_type": "Graffiti",')
    print('         "trend": "increasing",')
    print('         "percentage_change": 15.2,')
    print('         "confidence": 0.85')
    print('       }')
    print('     ],')
    print('     "insights": [')
    print('       "Graffiti complaints show seasonal patterns",')
    print('       "Homelessness issues remain stable"')
    print('     ]')
    print("   }")
    
    print("\n" + "=" * 60)
    print("HOW TO USE THE SYSTEM")
    print("=" * 60)
    print("""
    1. START THE SERVER:
       uvicorn app.main:app --reload
    
    2. TRIGGER AGENTS:
       POST /api/agents/reddit/scrape
       POST /api/agents/311/scrape
       POST /api/agents/run-pipeline
    
    3. GET REAL-TIME DATA:
       POST /api/map/data
       GET /api/insights/trends
       GET /api/agents/status
    
    4. MONITOR SYSTEM:
       GET /api/agents/status
       GET /api/agents/results/recent
       GET /api/agents/health
    """)

if __name__ == "__main__":
    show_api_responses()

