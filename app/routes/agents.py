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

# Cache storage
sf311_cache = {
    "issues": [],
    "raw_requests": [],
    "last_updated": None,
    "total_requests": 0,
    "filtered_requests": 0
}

neighborhood_insights_cache = {
    "insights": [],
    "summary": None,
    "last_updated": None
}

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

@router.post("/agents/sf311/launch")
async def launch_sf311_agent(
    pages: int = 20,
    filter_types: List[str] = None,
    min_severity: str = "low",
    clear_cache: bool = False
):
    """Launch SF311 agent to fetch real-time 311 data for map visualization"""
    try:
        logger.info(f"Launching SF311 agent with {pages} pages, clear_cache={clear_cache}")
        
        # Clear cache if requested
        if clear_cache:
            sf311_cache["issues"] = []
            sf311_cache["raw_requests"] = []
            sf311_cache["total_requests"] = 0
            sf311_cache["filtered_requests"] = 0
            logger.info("SF311 cache cleared")
        
        # Create SF311 agent and task
        sf311_agent = SF311Agent()
        sf311_task = SF311Task(
            task_id=f"sf311_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            agent_id="sf311_agent",
            data={},
            pages=pages,
            filter_types=filter_types,
            min_severity=min_severity
        )
        
        # Execute the agent
        result = await sf311_agent.execute(sf311_task)
        
        # Extract issues for frontend
        issues = []
        if result.data.get('issues'):
            for issue in result.data['issues']:
                # Extract coordinates if available - check multiple sources
                coordinates = None
                
                # First try direct coordinates
                if issue.get('coordinates'):
                    coordinates = issue['coordinates']
                # Then try metadata.geographic_analysis.coordinates
                elif issue.get('metadata', {}).get('geographic_analysis', {}).get('coordinates'):
                    coords_str = issue['metadata']['geographic_analysis']['coordinates']
                    if coords_str:
                        # Parse coordinates like "(37.7749, -122.4194)"
                        import re
                        coords_match = re.search(r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', coords_str)
                        if coords_match:
                            lat = float(coords_match.group(1))
                            lng = float(coords_match.group(2))
                            coordinates = [lng, lat]  # [longitude, latitude] for frontend
                
                # Extract neighborhood from multiple sources
                neighborhood = (issue.get('neighborhood') or 
                              issue.get('metadata', {}).get('geographic_analysis', {}).get('neighborhood'))
                
                issues.append({
                    "id": issue.get('id', f"issue_{len(issues)}"),
                    "title": issue.get('title', 'Unknown Issue'),
                    "description": issue.get('description', ''),
                    "severity": issue.get('severity', 'low'),
                    "source": issue.get('source', '311'),
                    "coordinates": coordinates,
                    "neighborhood": neighborhood,
                    "metadata": issue.get('metadata', {})
                })
        
        # Also include raw 311 requests for detailed mapping
        raw_requests = []
        if result.data.get('filtered_data'):
            for req in result.data['filtered_data']:
                # Extract coordinates from raw request
                coordinates = None
                if req.get('coordinates'):
                    coords_str = req['coordinates']
                    import re
                    coords_match = re.search(r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', coords_str)
                    if coords_match:
                        lat = float(coords_match.group(1))
                        lng = float(coords_match.group(2))
                        coordinates = [lng, lat]  # [longitude, latitude] for frontend
                
                raw_requests.append({
                    "id": req.get('offense_id', f"request_{len(raw_requests)}"),
                    "offense_type": req.get('offense_type', ''),
                    "description": req.get('description', ''),
                    "address": req.get('address', ''),
                    "coordinates": coordinates,
                    "neighborhood": req.get('neighborhood'),
                    "severity": req.get('severity', 'low'),
                    "offense_id": req.get('offense_id', '')
                })
        
        # Merge with cache
        sf311_cache["issues"].extend(issues)
        sf311_cache["raw_requests"].extend(raw_requests)
        sf311_cache["total_requests"] += result.total_requests
        sf311_cache["filtered_requests"] += result.filtered_requests
        sf311_cache["last_updated"] = datetime.utcnow().isoformat()
        
        return {
            "success": True,
            "status": result.status.value,
            "issues": sf311_cache["issues"],  # Return all cached issues
            "raw_requests": sf311_cache["raw_requests"],  # Return all cached requests
            "summary": {
                "total_requests": sf311_cache["total_requests"],
                "filtered_requests": sf311_cache["filtered_requests"],
                "issues_identified": len(sf311_cache["issues"]),
                "geographic_patterns": result.geographic_patterns,
                "insights": result.insights,
                "cache_info": {
                    "total_cached_issues": len(sf311_cache["issues"]),
                    "total_cached_requests": len(sf311_cache["raw_requests"]),
                    "new_issues_added": len(issues),
                    "new_requests_added": len(raw_requests),
                    "last_updated": sf311_cache["last_updated"]
                }
            },
            "execution_time": result.execution_time
        }
        
    except Exception as e:
        logger.error(f"SF311 agent launch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SF311 agent failed: {str(e)}")

@router.post("/agents/sf311/neighborhood-insights")
async def get_neighborhood_insights(
    pages: int = 10,
    min_severity: str = "low"
):
    """Get LLM-generated insights for SF311 data grouped by neighborhood"""
    try:
        logger.info(f"Generating neighborhood insights with {pages} pages")
        
        # Reset cache every time for neighborhood insights
        neighborhood_insights_cache["insights"] = []
        neighborhood_insights_cache["summary"] = None
        neighborhood_insights_cache["last_updated"] = None
        logger.info("Neighborhood insights cache reset")
        
        # Create SF311 agent and task
        sf311_agent = SF311Agent()
        sf311_task = SF311Task(
            task_id=f"sf311_insights_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            agent_id="sf311_agent",
            data={},
            pages=pages,
            filter_types=None,
            min_severity=min_severity
        )
        
        # Execute the agent
        result = await sf311_agent.execute(sf311_task)
        
        # Group data by neighborhood
        neighborhood_data = {}
        if result.data.get('filtered_data'):
            for request in result.data['filtered_data']:
                neighborhood = request.get('neighborhood') or 'San Francisco'
                if neighborhood not in neighborhood_data:
                    neighborhood_data[neighborhood] = {
                        'requests': [],
                        'issue_types': {},
                        'severity_counts': {'high': 0, 'medium': 0, 'low': 0},
                        'total_requests': 0
                    }
                
                neighborhood_data[neighborhood]['requests'].append(request)
                neighborhood_data[neighborhood]['total_requests'] += 1
                
                # Count issue types
                offense_type = request.get('offense_type', 'Unknown')
                neighborhood_data[neighborhood]['issue_types'][offense_type] = \
                    neighborhood_data[neighborhood]['issue_types'].get(offense_type, 0) + 1
                
                # Count severity
                severity = request.get('severity', 'low')
                neighborhood_data[neighborhood]['severity_counts'][severity] += 1
        
        # Generate insights for each neighborhood
        neighborhood_insights = []
        for neighborhood, data in neighborhood_data.items():
            if data['total_requests'] == 0:
                continue
                
            # Get sample requests (up to 3 per neighborhood)
            sample_requests = data['requests'][:3]
            
            # Create insight data
            insight_data = {
                'neighborhood': neighborhood,
                'total_requests': data['total_requests'],
                'top_issue_types': dict(sorted(data['issue_types'].items(), 
                                             key=lambda x: x[1], reverse=True)[:3]),
                'severity_distribution': data['severity_counts'],
                'sample_requests': [
                    {
                        'offense_type': req.get('offense_type', ''),
                        'description': req.get('description', '')[:100] + '...' if len(req.get('description', '')) > 100 else req.get('description', ''),
                        'severity': req.get('severity', 'low'),
                        'address': req.get('address', '')
                    }
                    for req in sample_requests
                ]
            }
            
            # Generate LLM insights for this neighborhood
            try:
                from app.services.claude_service import ClaudeService
                claude_service = ClaudeService()
                
                prompt = f"""
                Analyze the following SF311 data for {neighborhood} neighborhood in San Francisco:
                
                Total Requests: {data['total_requests']}
                Top Issue Types: {data['issue_types']}
                Severity Distribution: {data['severity_counts']}
                Sample Requests: {sample_requests}
                
                Provide 2-3 key insights about this neighborhood's community issues. Write each insight as a complete, well-formed sentence that:
                1. Identifies patterns or trends in the data
                2. Explains potential root causes or context
                3. Suggests actionable recommendations
                
                Write insights as natural sentences, not bullet points or analysis summaries. Each insight should be 1-2 sentences long and provide meaningful context about the neighborhood's challenges.
                
                Format as a JSON object with 'insights' array containing the sentences.
                """
                
                llm_response = await claude_service.generate_response(prompt)
                
                # Try to parse JSON response - handle markdown formatting
                try:
                    import json
                    import re
                    
                    # Remove markdown code blocks if present
                    cleaned_response = llm_response.strip()
                    if cleaned_response.startswith('```json'):
                        cleaned_response = cleaned_response[7:]  # Remove ```json
                    if cleaned_response.endswith('```'):
                        cleaned_response = cleaned_response[:-3]  # Remove ```
                    
                    # Clean up any extra whitespace
                    cleaned_response = cleaned_response.strip()
                    
                    parsed_response = json.loads(cleaned_response)
                    insights = parsed_response.get('insights', [])
                    
                    # Ensure insights is a list of strings
                    if isinstance(insights, list):
                        insight_data['llm_insights'] = insights
                    else:
                        insight_data['llm_insights'] = [str(insights)]
                        
                except Exception as parse_error:
                    logger.warning(f"Failed to parse LLM response for {neighborhood}: {parse_error}")
                    # Fallback: try to extract insights from the raw response
                    try:
                        # Look for insights array in the response
                        insights_match = re.search(r'"insights":\s*\[(.*?)\]', llm_response, re.DOTALL)
                        if insights_match:
                            insights_text = insights_match.group(1)
                            # Split by quotes and clean up
                            insights = []
                            for insight in insights_text.split('",'):
                                insight = insight.strip().strip('"').strip()
                                if insight:
                                    insights.append(insight)
                            insight_data['llm_insights'] = insights
                        else:
                            # Last resort: use the whole response as a single insight
                            insight_data['llm_insights'] = [llm_response]
                    except:
                        insight_data['llm_insights'] = [llm_response]
                    
            except Exception as e:
                logger.warning(f"Failed to generate LLM insights for {neighborhood}: {str(e)}")
                top_issues = list(data['issue_types'].keys())[:2]
                has_high_severity = data['severity_counts']['high'] > 0
                
                insight_data['llm_insights'] = [
                    f"The {neighborhood} neighborhood shows {data['total_requests']} community service requests, with {top_issues[0]} being the most common issue type.",
                    f"The severity distribution indicates {'significant high-priority concerns' if has_high_severity else 'moderate priority issues'} that require attention from city services."
                ]
            
            neighborhood_insights.append(insight_data)
        
        # Sort by total requests
        neighborhood_insights.sort(key=lambda x: x['total_requests'], reverse=True)
        
        # Store in cache
        neighborhood_insights_cache["insights"] = neighborhood_insights
        neighborhood_insights_cache["summary"] = {
            "total_neighborhoods": len(neighborhood_insights),
            "total_requests": sum(data['total_requests'] for data in neighborhood_data.values()),
            "most_active_neighborhood": neighborhood_insights[0]['neighborhood'] if neighborhood_insights else None
        }
        neighborhood_insights_cache["last_updated"] = datetime.utcnow().isoformat()
        
        return {
            "success": True,
            "neighborhood_insights": neighborhood_insights_cache["insights"],
            "summary": neighborhood_insights_cache["summary"]
        }
        
    except Exception as e:
        logger.error(f"Neighborhood insights generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Neighborhood insights failed: {str(e)}")

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

