#!/usr/bin/env python3
"""
Populate the knowledge graph with sample data for testing
"""

import asyncio
import json
from datetime import datetime
from app.database import AsyncSessionLocal, CommunityIssue, Location, IssueRelationship, IssueCluster
from sqlalchemy import text

async def populate_sample_data():
    """Populate the database with sample data"""
    
    print("Populating database with sample data...")
    
    async with AsyncSessionLocal() as db:
        # Create sample locations
        locations = [
            {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "address": "123 Market St",
                "neighborhood": "SOMA",
                "city": "San Francisco",
                "county": "San Francisco"
            },
            {
                "latitude": 37.7849,
                "longitude": -122.4094,
                "address": "456 Mission St",
                "neighborhood": "Mission District",
                "city": "San Francisco",
                "county": "San Francisco"
            },
            {
                "latitude": 37.7649,
                "longitude": -122.4294,
                "address": "789 Folsom St",
                "neighborhood": "SOMA",
                "city": "San Francisco",
                "county": "San Francisco"
            }
        ]
        
        location_objects = []
        for loc_data in locations:
            location = Location(**loc_data)
            db.add(location)
            location_objects.append(location)
        
        await db.commit()
        print(f"Created {len(location_objects)} locations")
        
        # Create sample community issues
        issues = [
            {
                "title": "BART delays causing major commute issues",
                "description": "Another day of BART delays. This is getting ridiculous. 30+ minute delays every morning.",
                "severity": "high",
                "source": "reddit",
                "source_id": "reddit_001",
                "location_id": location_objects[0].id,
                "metadata_json": json.dumps({
                    "subreddit": "sftransportation",
                    "score": 25,
                    "created_utc": 1703123456,
                    "url": "https://reddit.com/r/sftransportation/comments/abc123",
                    "keywords": ["transportation", "bart", "delays", "commute"]
                }),
                "embedding_vector": json.dumps([0.1, 0.2, 0.3, 0.4, 0.5])
            },
            {
                "title": "Homeless encampment growing in Mission District",
                "description": "The encampment under the 101 overpass has grown significantly. Safety concerns for residents.",
                "severity": "medium",
                "source": "reddit",
                "source_id": "reddit_002",
                "location_id": location_objects[1].id,
                "metadata_json": json.dumps({
                    "subreddit": "sanfrancisco",
                    "score": 18,
                    "created_utc": 1703120000,
                    "url": "https://reddit.com/r/sanfrancisco/comments/def456",
                    "keywords": ["homeless", "safety", "mission", "encampment"]
                }),
                "embedding_vector": json.dumps([0.2, 0.3, 0.4, 0.5, 0.6])
            },
            {
                "title": "Graffiti on commercial building",
                "description": "Multiple tags on commercial building facade at 2972 16th St",
                "severity": "medium",
                "source": "311",
                "source_id": "311_001",
                "location_id": location_objects[1].id,
                "metadata_json": json.dumps({
                    "request_type": "Graffiti",
                    "status": "Open",
                    "created_date": "2024-01-15",
                    "keywords": ["graffiti", "vandalism", "cleanup"]
                }),
                "embedding_vector": json.dumps([0.3, 0.4, 0.5, 0.6, 0.7])
            },
            {
                "title": "Street cleaning needed",
                "description": "Accumulated debris and litter on sidewalk",
                "severity": "low",
                "source": "311",
                "source_id": "311_002",
                "location_id": location_objects[2].id,
                "metadata_json": json.dumps({
                    "request_type": "Street Cleaning",
                    "status": "Open",
                    "created_date": "2024-01-16",
                    "keywords": ["cleaning", "litter", "sidewalk"]
                }),
                "embedding_vector": json.dumps([0.4, 0.5, 0.6, 0.7, 0.8])
            },
            {
                "title": "Public transportation reliability issues",
                "description": "Frequent delays and cancellations affecting daily commuters",
                "severity": "high",
                "source": "reddit",
                "source_id": "reddit_003",
                "location_id": location_objects[0].id,
                "metadata_json": json.dumps({
                    "subreddit": "sftransportation",
                    "score": 32,
                    "created_utc": 1703125000,
                    "url": "https://reddit.com/r/sftransportation/comments/ghi789",
                    "keywords": ["transportation", "reliability", "delays", "commute"]
                }),
                "embedding_vector": json.dumps([0.15, 0.25, 0.35, 0.45, 0.55])
            }
        ]
        
        issue_objects = []
        for issue_data in issues:
            issue = CommunityIssue(**issue_data)
            db.add(issue)
            issue_objects.append(issue)
        
        await db.commit()
        print(f"Created {len(issue_objects)} community issues")
        
        # Create sample relationships
        relationships = [
            {
                "issue_id_1": issue_objects[0].id,
                "issue_id_2": issue_objects[4].id,
                "relationship_type": "similar",
                "strength": 0.85
            },
            {
                "issue_id_1": issue_objects[1].id,
                "issue_id_2": issue_objects[2].id,
                "relationship_type": "geographic",
                "strength": 0.75
            },
            {
                "issue_id_1": issue_objects[2].id,
                "issue_id_2": issue_objects[3].id,
                "relationship_type": "temporal",
                "strength": 0.60
            }
        ]
        
        for rel_data in relationships:
            relationship = IssueRelationship(**rel_data)
            db.add(relationship)
        
        await db.commit()
        print(f"Created {len(relationships)} relationships")
        
        # Create sample clusters
        clusters = [
            {
                "cluster_name": "Transportation Issues",
                "representative_issue_id": issue_objects[0].id,
                "pattern_description": "Public transportation delays, reliability issues, and commuter complaints",
                "issue_count": 2,
                "severity_score": 0.85
            },
            {
                "cluster_name": "Neighborhood Maintenance",
                "representative_issue_id": issue_objects[2].id,
                "pattern_description": "Graffiti, street cleaning, and general neighborhood upkeep issues",
                "issue_count": 2,
                "severity_score": 0.50
            }
        ]
        
        for cluster_data in clusters:
            cluster = IssueCluster(**cluster_data)
            db.add(cluster)
        
        await db.commit()
        print(f"Created {len(clusters)} clusters")
        
        # Verify data
        result = await db.execute(text("SELECT COUNT(*) FROM community_issues"))
        issues_count = result.scalar()
        print(f"Total issues: {issues_count}")
        
        result = await db.execute(text("SELECT COUNT(*) FROM issue_relationships"))
        relationships_count = result.scalar()
        print(f"Total relationships: {relationships_count}")
        
        result = await db.execute(text("SELECT COUNT(*) FROM issue_clusters"))
        clusters_count = result.scalar()
        print(f"Total clusters: {clusters_count}")
        
        result = await db.execute(text("SELECT COUNT(*) FROM locations"))
        locations_count = result.scalar()
        print(f"Total locations: {locations_count}")

if __name__ == "__main__":
    asyncio.run(populate_sample_data())
