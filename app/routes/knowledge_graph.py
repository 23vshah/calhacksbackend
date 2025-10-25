"""
Knowledge Graph API endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from app.database import get_db
from app.database import CommunityIssue, Location, IssueRelationship, IssueCluster
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/knowledge-graph/nodes")
async def get_graph_nodes(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    source: Optional[str] = Query(None, description="Filter by source"),
    neighborhood: Optional[str] = Query(None, description="Filter by neighborhood"),
    limit: int = Query(100, description="Maximum number of nodes to return")
):
    """Get knowledge graph nodes (issues)"""
    try:
        db = next(get_db())
        
        # Build query
        query = db.query(CommunityIssue)
        
        # Apply filters
        if severity:
            query = query.filter(CommunityIssue.severity == severity)
        if source:
            query = query.filter(CommunityIssue.source == source)
        if neighborhood:
            query = query.join(Location).filter(Location.neighborhood == neighborhood)
        
        # Get issues
        issues = query.limit(limit).all()
        
        # Format response
        nodes = []
        for issue in issues:
            node = {
                "id": f"{issue.source}_{issue.id}",
                "title": issue.title,
                "source": issue.source,
                "location": issue.location.neighborhood if issue.location else "Unknown",
                "severity": issue.severity,
                "keywords": issue.metadata_json.get("keywords", []) if issue.metadata_json else [],
                "coordinates": [issue.location.latitude, issue.location.longitude] if issue.location else None,
                "metadata": issue.metadata_json or {},
                "created_at": issue.created_at.isoformat() if issue.created_at else None
            }
            nodes.append(node)
        
        return {
            "nodes": nodes,
            "total": len(nodes),
            "filters": {
                "severity": severity,
                "source": source,
                "neighborhood": neighborhood
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get graph nodes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-graph/edges")
async def get_graph_edges(
    min_strength: float = Query(0.5, description="Minimum relationship strength"),
    relationship_type: Optional[str] = Query(None, description="Filter by relationship type"),
    limit: int = Query(200, description="Maximum number of edges to return")
):
    """Get knowledge graph edges (relationships)"""
    try:
        db = next(get_db())
        
        # Build query
        query = db.query(IssueRelationship)
        
        # Apply filters
        query = query.filter(IssueRelationship.strength >= min_strength)
        if relationship_type:
            query = query.filter(IssueRelationship.relationship_type == relationship_type)
        
        # Get relationships
        relationships = query.limit(limit).all()
        
        # Format response
        edges = []
        for rel in relationships:
            edge = {
                "id": f"edge_{rel.id}",
                "source": f"{rel.issue_1.source}_{rel.issue_1.id}",
                "target": f"{rel.issue_2.source}_{rel.issue_2.id}",
                "type": rel.relationship_type,
                "strength": rel.strength,
                "reason": f"{rel.relationship_type} relationship between issues",
                "created_at": rel.created_at.isoformat() if rel.created_at else None
            }
            edges.append(edge)
        
        return {
            "edges": edges,
            "total": len(edges),
            "filters": {
                "min_strength": min_strength,
                "relationship_type": relationship_type
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get graph edges: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-graph/clusters")
async def get_graph_clusters(
    min_issues: int = Query(2, description="Minimum number of issues per cluster"),
    limit: int = Query(20, description="Maximum number of clusters to return")
):
    """Get knowledge graph clusters"""
    try:
        db = next(get_db())
        
        # Get clusters
        clusters = db.query(IssueCluster).filter(
            IssueCluster.issue_count >= min_issues
        ).limit(limit).all()
        
        # Format response
        cluster_data = []
        for cluster in clusters:
            cluster_info = {
                "id": f"cluster_{cluster.id}",
                "name": cluster.cluster_name,
                "nodes": [f"{cluster.representative_issue.source}_{cluster.representative_issue.id}"],
                "themes": cluster.pattern_description.split(", ") if cluster.pattern_description else [],
                "geographic": [cluster.representative_issue.location.neighborhood] if cluster.representative_issue.location else [],
                "severity_distribution": {
                    "high": 0,
                    "medium": 0,
                    "low": 0
                },
                "issue_count": cluster.issue_count,
                "severity_score": cluster.severity_score,
                "created_at": cluster.created_at.isoformat() if cluster.created_at else None
            }
            cluster_data.append(cluster_info)
        
        return {
            "clusters": cluster_data,
            "total": len(cluster_data),
            "filters": {
                "min_issues": min_issues
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get graph clusters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-graph/query")
async def query_knowledge_graph(
    query_type: str = Query(..., description="Query type: spatial, temporal, similarity, cluster"),
    neighborhood: Optional[str] = Query(None, description="Neighborhood for spatial queries"),
    coordinates: Optional[str] = Query(None, description="Coordinates as 'lat,lon' for spatial queries"),
    radius: Optional[float] = Query(1.0, description="Radius in miles for spatial queries"),
    start_date: Optional[str] = Query(None, description="Start date for temporal queries"),
    end_date: Optional[str] = Query(None, description="End date for temporal queries"),
    timeframe: Optional[str] = Query("week", description="Timeframe: week, month, year"),
    issue_id: Optional[str] = Query(None, description="Issue ID for similarity queries"),
    min_similarity: Optional[float] = Query(0.7, description="Minimum similarity score"),
    cluster_id: Optional[int] = Query(None, description="Cluster ID for cluster queries"),
    cluster_name: Optional[str] = Query(None, description="Cluster name for cluster queries")
):
    """Query the knowledge graph with different types of queries"""
    try:
        db = next(get_db())
        
        # Build query parameters based on query type
        if query_type == "spatial":
            params = {
                "neighborhood": neighborhood,
                "coordinates": coordinates.split(",") if coordinates else None,
                "radius": radius
            }
            return await _spatial_query(db, params)
        elif query_type == "temporal":
            params = {
                "start_date": start_date,
                "end_date": end_date,
                "timeframe": timeframe
            }
            return await _temporal_query(db, params)
        elif query_type == "similarity":
            params = {
                "issue_id": issue_id,
                "min_similarity": min_similarity
            }
            return await _similarity_query(db, params)
        elif query_type == "cluster":
            params = {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name
            }
            return await _cluster_query(db, params)
        else:
            raise HTTPException(status_code=400, detail="Invalid query type")
            
    except Exception as e:
        logger.error(f"Failed to query knowledge graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def _spatial_query(db: Session, params: Dict[str, Any]):
    """Spatial queries - find issues by location"""
    neighborhood = params.get("neighborhood")
    coordinates = params.get("coordinates")
    radius = params.get("radius", 1.0)  # miles
    
    query = db.query(CommunityIssue)
    
    if neighborhood:
        query = query.join(Location).filter(Location.neighborhood == neighborhood)
    
    if coordinates:
        lat, lon = coordinates
        # Simple bounding box query (in production, use proper geospatial queries)
        query = query.join(Location).filter(
            and_(
                Location.latitude.between(lat - radius/69, lat + radius/69),
                Location.longitude.between(lon - radius/69, lon + radius/69)
            )
        )
    
    issues = query.all()
    
    return {
        "query_type": "spatial",
        "results": [
            {
                "id": f"{issue.source}_{issue.id}",
                "title": issue.title,
                "location": issue.location.neighborhood if issue.location else "Unknown",
                "coordinates": [issue.location.latitude, issue.location.longitude] if issue.location else None
            }
            for issue in issues
        ],
        "total": len(issues)
    }

async def _temporal_query(db: Session, params: Dict[str, Any]):
    """Temporal queries - find issues by time"""
    start_date = params.get("start_date")
    end_date = params.get("end_date")
    timeframe = params.get("timeframe", "week")
    
    query = db.query(CommunityIssue)
    
    if start_date:
        query = query.filter(CommunityIssue.created_at >= start_date)
    if end_date:
        query = query.filter(CommunityIssue.created_at <= end_date)
    
    if timeframe == "week":
        query = query.filter(CommunityIssue.created_at >= datetime.utcnow() - timedelta(days=7))
    elif timeframe == "month":
        query = query.filter(CommunityIssue.created_at >= datetime.utcnow() - timedelta(days=30))
    
    issues = query.all()
    
    return {
        "query_type": "temporal",
        "results": [
            {
                "id": f"{issue.source}_{issue.id}",
                "title": issue.title,
                "created_at": issue.created_at.isoformat() if issue.created_at else None,
                "severity": issue.severity
            }
            for issue in issues
        ],
        "total": len(issues)
    }

async def _similarity_query(db: Session, params: Dict[str, Any]):
    """Similarity queries - find similar issues"""
    issue_id = params.get("issue_id")
    min_similarity = params.get("min_similarity", 0.7)
    
    if not issue_id:
        raise HTTPException(status_code=400, detail="issue_id is required for similarity queries")
    
    # Find relationships with high similarity
    relationships = db.query(IssueRelationship).filter(
        or_(
            IssueRelationship.issue_id_1 == issue_id,
            IssueRelationship.issue_id_2 == issue_id
        ),
        IssueRelationship.relationship_type == "similar",
        IssueRelationship.strength >= min_similarity
    ).all()
    
    return {
        "query_type": "similarity",
        "results": [
            {
                "id": f"edge_{rel.id}",
                "source": f"{rel.issue_1.source}_{rel.issue_1.id}",
                "target": f"{rel.issue_2.source}_{rel.issue_2.id}",
                "similarity": rel.strength,
                "reason": "Similar issue content and context"
            }
            for rel in relationships
        ],
        "total": len(relationships)
    }

async def _cluster_query(db: Session, params: Dict[str, Any]):
    """Cluster queries - find issues in clusters"""
    cluster_id = params.get("cluster_id")
    cluster_name = params.get("cluster_name")
    
    query = db.query(IssueCluster)
    
    if cluster_id:
        query = query.filter(IssueCluster.id == cluster_id)
    if cluster_name:
        query = query.filter(IssueCluster.cluster_name.ilike(f"%{cluster_name}%"))
    
    clusters = query.all()
    
    return {
        "query_type": "cluster",
        "results": [
            {
                "id": f"cluster_{cluster.id}",
                "name": cluster.cluster_name,
                "issue_count": cluster.issue_count,
                "severity_score": cluster.severity_score,
                "pattern": cluster.pattern_description
            }
            for cluster in clusters
        ],
        "total": len(clusters)
    }

@router.get("/knowledge-graph/metrics")
async def get_graph_metrics():
    """Get knowledge graph metrics and statistics"""
    try:
        db = next(get_db())
        
        # Node metrics
        total_nodes = db.query(CommunityIssue).count()
        reddit_nodes = db.query(CommunityIssue).filter(CommunityIssue.source == "reddit").count()
        sf311_nodes = db.query(CommunityIssue).filter(CommunityIssue.source == "311").count()
        
        # Edge metrics
        total_edges = db.query(IssueRelationship).count()
        geographic_edges = db.query(IssueRelationship).filter(IssueRelationship.relationship_type == "geographic").count()
        temporal_edges = db.query(IssueRelationship).filter(IssueRelationship.relationship_type == "temporal").count()
        similar_edges = db.query(IssueRelationship).filter(IssueRelationship.relationship_type == "similar").count()
        
        # Cluster metrics
        total_clusters = db.query(IssueCluster).count()
        largest_cluster = db.query(IssueCluster).order_by(IssueCluster.issue_count.desc()).first()
        
        # Average relationship strength
        avg_strength = db.query(func.avg(IssueRelationship.strength)).scalar() or 0.0
        
        return {
            "nodes": {
                "total": total_nodes,
                "reddit": reddit_nodes,
                "sf311": sf311_nodes,
                "distribution": {
                    "reddit": (reddit_nodes / total_nodes * 100) if total_nodes > 0 else 0,
                    "sf311": (sf311_nodes / total_nodes * 100) if total_nodes > 0 else 0
                }
            },
            "edges": {
                "total": total_edges,
                "geographic": geographic_edges,
                "temporal": temporal_edges,
                "similar": similar_edges,
                "distribution": {
                    "geographic": (geographic_edges / total_edges * 100) if total_edges > 0 else 0,
                    "temporal": (temporal_edges / total_edges * 100) if total_edges > 0 else 0,
                    "similar": (similar_edges / total_edges * 100) if total_edges > 0 else 0
                },
                "average_strength": round(avg_strength, 2)
            },
            "clusters": {
                "total": total_clusters,
                "largest": {
                    "name": largest_cluster.cluster_name if largest_cluster else None,
                    "size": largest_cluster.issue_count if largest_cluster else 0
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get graph metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
