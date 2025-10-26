"""
Agent API Routes
Manual triggers and real-time data access
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.services.agent_framework import AgentOrchestrator, AgentTask
from app.services.agents.reddit_agent import RedditAgent, RedditTask
from app.services.agents.sf311_agent import SF311Agent, SF311Task
from app.services.agents.knowledge_graph_agent import KnowledgeGraphAgent, KnowledgeGraphTask

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize agents
reddit_agent = RedditAgent()
sf311_agent = SF311Agent()
kg_agent = KnowledgeGraphAgent()

# Register agents with orchestrator
orchestrator = AgentOrchestrator()
orchestrator.register_agent(reddit_agent)
orchestrator.register_agent(sf311_agent)
orchestrator.register_agent(kg_agent)

# Request/Response Models
class AgentTriggerRequest(BaseModel):
    city: str = "San Francisco"
    keywords: Optional[List[str]] = None
    pages: int = 5
    max_subreddits: int = 10
    max_posts_per_subreddit: int = 25
    issue_keywords: Optional[List[str]] = None

class AgentStatusResponse(BaseModel):
    agent_id: str
    status: str
    last_run: Optional[datetime]
    successful_patterns: int
    failed_patterns: int

class AgentResultResponse(BaseModel):
    task_id: str
    agent_id: str
    status: str
    execution_time: float
    insights: List[str]
    data: Dict[str, Any]

class MapDataRequest(BaseModel):
    bounds: Optional[Dict[str, float]] = None  # {lat1, lng1, lat2, lng2}
    neighborhood: Optional[str] = None
    issue_types: Optional[List[str]] = None
    severity: Optional[str] = None

class MapDataResponse(BaseModel):
    issues: List[Dict[str, Any]]
    hotspots: List[Dict[str, Any]]
    insights: List[str]
    total_issues: int

# Agent Trigger Endpoints
@router.post("/agents/reddit/scrape")
async def trigger_reddit_agent(request: AgentTriggerRequest):
    """Trigger Reddit agent to scrape city discussions"""
    try:
        task = RedditTask(
            task_id=f"reddit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            agent_id="reddit_agent",
            data={},
            city=request.city,
            keywords=request.keywords or ["san francisco", "sf", "bay area"],
            max_subreddits=request.max_subreddits,
            max_posts_per_subreddit=request.max_posts_per_subreddit,
            issue_keywords=request.issue_keywords
        )
        
        result = await orchestrator.run_agent("reddit_agent", task)
        
        return AgentResultResponse(
            task_id=result.task_id,
            agent_id=result.agent_id,
            status=result.status.value,
            execution_time=result.execution_time,
            insights=result.insights or [],
            data=result.data
        )
        
    except Exception as e:
        logger.error(f"Reddit agent trigger failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reddit agent failed: {str(e)}")

@router.post("/agents/311/scrape")
async def trigger_sf311_agent(request: AgentTriggerRequest):
    """Trigger SF311 agent to scrape official city data"""
    try:
        task = SF311Task(
            task_id=f"sf311_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            agent_id="sf311_agent",
            data={},
            pages=request.pages
        )
        
        result = await orchestrator.run_agent("sf311_agent", task)
        
        return AgentResultResponse(
            task_id=result.task_id,
            agent_id=result.agent_id,
            status=result.status.value,
            execution_time=result.execution_time,
            insights=result.insights or [],
            data=result.data
        )
        
    except Exception as e:
        logger.error(f"SF311 agent trigger failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SF311 agent failed: {str(e)}")

@router.post("/agents/analyze")
async def trigger_knowledge_graph_agent(request: AgentTriggerRequest):
    """Trigger knowledge graph agent to analyze and connect issues"""
    try:
        # This would typically get issues from database
        # For now, we'll use empty list as placeholder
        new_issues = []  # This should be populated from database
        
        task = KnowledgeGraphTask(
            task_id=f"kg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            agent_id="knowledge_graph_agent",
            data={},
            new_issues=new_issues
        )
        
        result = await orchestrator.run_agent("knowledge_graph_agent", task)
        
        return AgentResultResponse(
            task_id=result.task_id,
            agent_id=result.agent_id,
            status=result.status.value,
            execution_time=result.execution_time,
            insights=result.insights or [],
            data=result.data
        )
        
    except Exception as e:
        logger.error(f"Knowledge graph agent trigger failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Knowledge graph agent failed: {str(e)}")

@router.post("/agents/run-pipeline")
async def run_agent_pipeline(request: AgentTriggerRequest):
    """Run complete agent pipeline: Reddit + SF311 + Knowledge Graph"""
    try:
        # Create tasks for all agents
        tasks = [
            RedditTask(
                task_id=f"reddit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                agent_id="reddit_agent",
                data={},
                city=request.city,
                keywords=request.keywords or ["san francisco", "sf", "bay area"],
                max_subreddits=request.max_subreddits,
                max_posts_per_subreddit=request.max_posts_per_subreddit,
                issue_keywords=request.issue_keywords
            ),
            SF311Task(
                task_id=f"sf311_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                agent_id="sf311_agent",
                data={},
                pages=request.pages
            )
        ]
        
        # Run agents in parallel
        results = await orchestrator.run_parallel(tasks)
        
        # Collect all issues for knowledge graph analysis
        all_issues = []
        for result in results:
            if result.status.value == "success" and "issues" in result.data:
                all_issues.extend(result.data["issues"])
        
        # Run knowledge graph agent if we have issues
        if all_issues:
            kg_task = KnowledgeGraphTask(
                task_id=f"kg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                agent_id="knowledge_graph_agent",
                data={},
                new_issues=all_issues
            )
            kg_result = await orchestrator.run_agent("knowledge_graph_agent", kg_task)
            results.append(kg_result)
        
        # Return combined results
        return {
            "pipeline_status": "completed",
            "agents_run": len(results),
            "total_issues_found": len(all_issues),
            "results": [
                {
                    "agent_id": result.agent_id,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "insights": result.insights or []
                }
                for result in results
            ]
        }
        
    except Exception as e:
        logger.error(f"Agent pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {str(e)}")

# Status and Monitoring Endpoints
@router.get("/agents/status")
async def get_agent_status():
    """Get status of all agents"""
    try:
        status = orchestrator.get_agent_status()
        return {
            "agents": status,
            "total_agents": len(status),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get agent status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")

@router.get("/agents/results/recent")
async def get_recent_results(limit: int = 10):
    """Get recent agent results"""
    try:
        results = orchestrator.get_recent_results(limit)
        return {
            "results": [
                {
                    "task_id": result.task_id,
                    "agent_id": result.agent_id,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "created_at": result.created_at.isoformat(),
                    "insights": result.insights or []
                }
                for result in results
            ],
            "total_results": len(results)
        }
    except Exception as e:
        logger.error(f"Failed to get recent results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent results: {str(e)}")

# Real-time Data Access Endpoints
@router.post("/map/data")
async def get_map_data(request: MapDataRequest):
    """Get geospatial data for map visualization"""
    try:
        # This would query the database for issues within bounds
        # For now, return mock data structure
        mock_data = {
            "issues": [
                {
                    "id": "issue_001",
                    "type": "Graffiti",
                    "severity": "medium",
                    "location": [37.7749, -122.4194],
                    "neighborhood": "Mission District",
                    "count": 5,
                    "trend": "increasing",
                    "related_reddit_posts": 2,
                    "311_reports": 5,
                    "pattern": "Commercial building facades"
                }
            ],
            "hotspots": [
                {
                    "neighborhood": "Tenderloin",
                    "issue_types": ["homelessness", "graffiti", "litter"],
                    "severity_score": 0.8,
                    "trend": "worsening"
                }
            ],
            "insights": [
                "Graffiti complaints increase 40% after major events",
                "Homelessness issues concentrated in Tenderloin",
                "Traffic complaints peak during rush hours"
            ],
            "total_issues": 1
        }
        
        return MapDataResponse(
            issues=mock_data["issues"],
            hotspots=mock_data["hotspots"],
            insights=mock_data["insights"],
            total_issues=mock_data["total_issues"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get map data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get map data: {str(e)}")

@router.get("/map/hotspots")
async def get_hotspots(neighborhood: Optional[str] = None):
    """Get geographic hotspots"""
    try:
        # This would query the database for hotspots
        # For now, return mock data
        mock_hotspots = [
            {
                "neighborhood": "Tenderloin",
                "issue_types": ["homelessness", "graffiti", "litter"],
                "severity_score": 0.8,
                "trend": "worsening",
                "issue_count": 15
            },
            {
                "neighborhood": "Mission District",
                "issue_types": ["graffiti", "traffic", "parking"],
                "severity_score": 0.6,
                "trend": "stable",
                "issue_count": 12
            }
        ]
        
        if neighborhood:
            filtered = [h for h in mock_hotspots if h["neighborhood"].lower() == neighborhood.lower()]
            return {"hotspots": filtered}
        
        return {"hotspots": mock_hotspots}
        
    except Exception as e:
        logger.error(f"Failed to get hotspots: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get hotspots: {str(e)}")

@router.get("/insights/trends")
async def get_trends(timeframe: int = 7):
    """Get trend insights"""
    try:
        # This would analyze temporal patterns
        # For now, return mock data
        mock_trends = {
            "timeframe_days": timeframe,
            "trends": [
                {
                    "issue_type": "Graffiti",
                    "trend": "increasing",
                    "percentage_change": 15.2,
                    "confidence": 0.85
                },
                {
                    "issue_type": "Homelessness",
                    "trend": "stable",
                    "percentage_change": 2.1,
                    "confidence": 0.72
                }
            ],
            "insights": [
                "Graffiti complaints show seasonal patterns",
                "Homelessness issues remain stable despite policy changes"
            ]
        }
        
        return mock_trends
        
    except Exception as e:
        logger.error(f"Failed to get trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")

@router.get("/insights/correlations")
async def get_correlations(issue_type: Optional[str] = None):
    """Get issue correlations"""
    try:
        # This would analyze relationships between issues
        # For now, return mock data
        mock_correlations = {
            "issue_type": issue_type or "all",
            "correlations": [
                {
                    "issue_1": "Graffiti",
                    "issue_2": "Litter",
                    "correlation_strength": 0.73,
                    "relationship_type": "geographic",
                    "confidence": 0.89
                },
                {
                    "issue_1": "Homelessness",
                    "issue_2": "Litter",
                    "correlation_strength": 0.65,
                    "relationship_type": "temporal",
                    "confidence": 0.76
                }
            ],
            "insights": [
                "Graffiti and litter often occur in same locations",
                "Homelessness and litter show temporal correlation"
            ]
        }
        
        return mock_correlations
        
    except Exception as e:
        logger.error(f"Failed to get correlations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get correlations: {str(e)}")

# Health Check
@router.get("/agents/health")
async def agent_health_check():
    """Health check for agent system"""
    try:
        status = orchestrator.get_agent_status()
        healthy_agents = sum(1 for agent_status in status.values() if agent_status["status"] != "failed")
        
        return {
            "status": "healthy" if healthy_agents == len(status) else "degraded",
            "total_agents": len(status),
            "healthy_agents": healthy_agents,
            "agents": status
        }
    except Exception as e:
        logger.error(f"Agent health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

