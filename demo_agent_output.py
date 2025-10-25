#!/usr/bin/env python3
"""
Demo script showing what the agent system stores and outputs
This shows the expected data structures and knowledge graph
"""

import json
from datetime import datetime
from typing import Dict, List, Any

def show_expected_output():
    """Show what the agent system produces"""
    
    print("=" * 80)
    print("AGENT SYSTEM EXPECTED OUTPUT DEMO")
    print("=" * 80)
    
    # 1. Reddit Agent Output
    print("\n1. REDDIT AGENT OUTPUT")
    print("-" * 40)
    
    reddit_output = {
        "agent_id": "reddit_agent",
        "status": "success",
        "execution_time": 2.3,
        "data": {
            "city": "San Francisco",
            "subreddits": ["sftransportation", "sanfrancisco", "bayarea"],
            "posts_analyzed": 45,
            "issues": [
                {
                    "title": "BART delays causing major commute issues",
                    "description": "Another day of BART delays. This is getting ridiculous. 30+ minute delays every morning.",
                    "severity": "high",
                    "source": "reddit",
                    "source_id": "abc123",
                    "location": {
                        "city": "San Francisco",
                        "neighborhood": "SOMA",
                        "coordinates": None,
                        "address": None
                    },
                    "metadata": {
                        "subreddit": "sftransportation",
                        "score": 25,
                        "created_utc": 1703123456,
                        "url": "https://reddit.com/r/sftransportation/comments/abc123"
                    }
                },
                {
                    "title": "Homeless encampment growing in Mission District",
                    "description": "The encampment under the 101 overpass has grown significantly. Safety concerns for residents.",
                    "severity": "medium",
                    "source": "reddit",
                    "source_id": "def456",
                    "location": {
                        "city": "San Francisco",
                        "neighborhood": "Mission District",
                        "coordinates": None,
                        "address": None
                    },
                    "metadata": {
                        "subreddit": "sanfrancisco",
                        "score": 18,
                        "created_utc": 1703120000,
                        "url": "https://reddit.com/r/sanfrancisco/comments/def456"
                    }
                }
            ]
        },
        "insights": [
            "Found 3 relevant subreddits for San Francisco",
            "Identified 2 community issues",
            "Most active subreddit: sftransportation"
        ]
    }
    
    print(json.dumps(reddit_output, indent=2))
    
    # 2. SF311 Agent Output
    print("\n\n2. SF311 AGENT OUTPUT")
    print("-" * 40)
    
    sf311_output = {
        "agent_id": "sf311_agent",
        "status": "success",
        "execution_time": 1.8,
        "data": {
            "total_requests": 50,
            "filtered_requests": 35,
            "issues": [
                {
                    "title": "Graffiti",
                    "description": "Multiple tags on commercial building facade",
                    "severity": "medium",
                    "source": "311",
                    "source_id": "#101002857943",
                    "location": {
                        "coordinates": "(37.76517043, -122.41906416)",
                        "address": "2972 16th St",
                        "neighborhood": "Mission District",
                        "city": "San Francisco"
                    },
                    "metadata": {
                        "offense_type": "Graffiti",
                        "geographic_analysis": {
                            "neighborhood": "Mission District",
                            "geographic_priority": "medium"
                        }
                    }
                },
                {
                    "title": "Street or sidewalk cleaning",
                    "description": "One mattress dumped on the sidewalk with some small loose litter nearby",
                    "severity": "low",
                    "source": "311",
                    "source_id": "#101002857947",
                    "location": {
                        "coordinates": "(37.78115948, -122.46452375)",
                        "address": "4214 Geary Blvd",
                        "neighborhood": "Richmond",
                        "city": "San Francisco"
                    },
                    "metadata": {
                        "offense_type": "Street or sidewalk cleaning",
                        "geographic_analysis": {
                            "neighborhood": "Richmond",
                            "geographic_priority": "low"
                        }
                    }
                }
            ],
            "geographic_patterns": {
                "neighborhood_counts": {
                    "Mission District": 12,
                    "Tenderloin": 8,
                    "SOMA": 6,
                    "Richmond": 4
                },
                "most_active_neighborhood": "Mission District",
                "offense_type_distribution": {
                    "Graffiti": 15,
                    "Street or sidewalk cleaning": 10,
                    "Blocked driveway & illegal parking": 8
                }
            }
        },
        "insights": [
            "Processed 50 311 requests",
            "Identified 35 community issues",
            "Top issue type: Graffiti",
            "Most active neighborhood: Mission District"
        ]
    }
    
    print(json.dumps(sf311_output, indent=2))
    
    # 3. Knowledge Graph Output
    print("\n\n3. KNOWLEDGE GRAPH AGENT OUTPUT")
    print("-" * 40)
    
    kg_output = {
        "agent_id": "knowledge_graph_agent",
        "status": "success",
        "execution_time": 3.1,
        "data": {
            "relationships_created": 3,
            "clusters_identified": 2,
            "relationships": [
                {
                    "issue_1": {
                        "title": "BART delays causing major commute issues",
                        "source": "reddit"
                    },
                    "issue_2": {
                        "title": "Blocked driveway & illegal parking",
                        "source": "311"
                    },
                    "similarity": 0.73,
                    "relationship_type": "geographic",
                    "strength": 0.73
                },
                {
                    "issue_1": {
                        "title": "Homeless encampment growing in Mission District",
                        "source": "reddit"
                    },
                    "issue_2": {
                        "title": "Street or sidewalk cleaning",
                        "source": "311"
                    },
                    "similarity": 0.65,
                    "relationship_type": "temporal",
                    "strength": 0.65
                }
            ],
            "clusters": [
                {
                    "cluster_id": 0,
                    "issue_count": 3,
                    "representative_issue": {
                        "title": "Graffiti",
                        "description": "Multiple tags on commercial building facade"
                    },
                    "common_themes": ["graffiti", "vandalism", "commercial"],
                    "geographic_distribution": {
                        "neighborhoods": {"Mission District": 2, "SOMA": 1},
                        "most_common_neighborhood": "Mission District"
                    },
                    "severity_distribution": {"high": 0, "medium": 2, "low": 1}
                },
                {
                    "cluster_id": 1,
                    "issue_count": 2,
                    "representative_issue": {
                        "title": "BART delays causing major commute issues",
                        "description": "Another day of BART delays"
                    },
                    "common_themes": ["transportation", "delays", "commute"],
                    "geographic_distribution": {
                        "neighborhoods": {"SOMA": 1, "Mission District": 1},
                        "most_common_neighborhood": "SOMA"
                    },
                    "severity_distribution": {"high": 1, "medium": 1, "low": 0}
                }
            ]
        },
        "insights": [
            "Created 3 relationships between issues",
            "Identified 2 issue clusters",
            "Generated 4 insights",
            "Most connected issue type: Transportation"
        ]
    }
    
    print(json.dumps(kg_output, indent=2))
    
    # 4. Database Storage Structure
    print("\n\n4. DATABASE STORAGE STRUCTURE")
    print("-" * 40)
    
    db_structure = {
        "community_issues": [
            {
                "id": 1,
                "title": "BART delays causing major commute issues",
                "description": "Another day of BART delays. This is getting ridiculous.",
                "severity": "high",
                "source": "reddit",
                "source_id": "abc123",
                "location_id": 1,
                "created_at": "2024-01-01T10:00:00Z",
                "metadata_json": '{"subreddit": "sftransportation", "score": 25}',
                "embedding_vector": "[0.1, 0.2, 0.3, ...]"  # Vector for similarity
            }
        ],
        "locations": [
            {
                "id": 1,
                "latitude": 37.7749,
                "longitude": -122.4194,
                "address": "Mission District",
                "neighborhood": "Mission District",
                "city": "San Francisco",
                "county": "San Francisco County"
            }
        ],
        "issue_relationships": [
            {
                "id": 1,
                "issue_id_1": 1,
                "issue_id_2": 2,
                "relationship_type": "geographic",
                "strength": 0.73,
                "created_at": "2024-01-01T10:05:00Z"
            }
        ],
        "issue_clusters": [
            {
                "id": 1,
                "cluster_name": "Transportation Issues",
                "representative_issue_id": 1,
                "pattern_description": "BART delays and transportation problems",
                "issue_count": 3,
                "severity_score": 0.8
            }
        ],
        "agent_memory": [
            {
                "id": 1,
                "agent_id": "reddit_agent",
                "memory_type": "successful_search",
                "data_json": '{"keywords": ["san francisco", "transportation"], "success_rate": 0.85}',
                "created_at": "2024-01-01T10:00:00Z"
            }
        ]
    }
    
    print(json.dumps(db_structure, indent=2))
    
    # 5. Real-Time Map Data
    print("\n\n5. REAL-TIME MAP DATA (for frontend)")
    print("-" * 40)
    
    map_data = {
        "city": "San Francisco",
        "issues": [
            {
                "id": "transportation_cluster_001",
                "type": "Transportation",
                "severity": "high",
                "location": [37.7749, -122.4194],
                "neighborhood": "Mission District",
                "count": 3,
                "trend": "increasing",
                "related_reddit_posts": 2,
                "311_reports": 1,
                "pattern": "BART delays and commute issues"
            },
            {
                "id": "graffiti_cluster_002",
                "type": "Graffiti",
                "severity": "medium",
                "location": [37.7651, -122.4190],
                "neighborhood": "Mission District",
                "count": 5,
                "trend": "stable",
                "related_reddit_posts": 1,
                "311_reports": 4,
                "pattern": "Commercial building facades"
            }
        ],
        "hotspots": [
            {
                "neighborhood": "Mission District",
                "issue_types": ["transportation", "graffiti", "homelessness"],
                "severity_score": 0.7,
                "trend": "worsening",
                "issue_count": 8
            },
            {
                "neighborhood": "Tenderloin",
                "issue_types": ["homelessness", "graffiti", "litter"],
                "severity_score": 0.9,
                "trend": "worsening",
                "issue_count": 12
            }
        ],
        "insights": [
            "Transportation issues concentrated in Mission District",
            "Graffiti complaints increase 40% after major events",
            "Homelessness issues show seasonal patterns"
        ],
        "total_issues": 20
    }
    
    print(json.dumps(map_data, indent=2))
    
    # 6. Knowledge Graph Visualization
    print("\n\n6. KNOWLEDGE GRAPH VISUALIZATION")
    print("-" * 40)
    
    print("""
    KNOWLEDGE GRAPH STRUCTURE:
    
    Issues (Nodes):
    ┌─────────────────────────────────────────────────────────────┐
    │  Reddit: "BART delays" ────┐                               │
    │  [SOMA, Transportation]    │                               │
    └────────────────────────────┼───────────────────────────────┘
                                 │
    ┌─────────────────────────────────────────────────────────────┐
                                 │  Geographic Relationship (0.73)      │
                                 │  Temporal Relationship (0.65)       │
                                 │                                   │
    ┌────────────────────────────┼───────────────────────────────┐   │
    │  311: "Blocked driveway"  │                               │   │
    │  [Mission, Parking]        │                               │   │
    └────────────────────────────┼───────────────────────────────┘   │
                                 │                                   │
    ┌────────────────────────────┼───────────────────────────────┐   │
    │  Reddit: "Homeless camp"  │                               │   │
    │  [Mission, Homelessness]  │                               │   │
    └────────────────────────────┼───────────────────────────────┘   │
                                 │                                   │
    ┌────────────────────────────┼───────────────────────────────┐   │
    │  311: "Street cleaning"   │                               │   │
    │  [Richmond, Litter]        │                               │   │
    └────────────────────────────┴───────────────────────────────┘
    
    Clusters:
    • Transportation Cluster: BART delays + Parking issues
    • Graffiti Cluster: Multiple graffiti reports
    • Homelessness Cluster: Encampments + Street cleaning
    
    Relationships:
    • Geographic: Same neighborhood issues
    • Temporal: Issues reported around same time
    • Similar: Similar issue types
    • Related: Cross-platform correlation
    """)
    
    print("\n" + "=" * 80)
    print("SUMMARY: What the Agent System Produces")
    print("=" * 80)
    print("""
    📊 DATA STORAGE:
    • CommunityIssue: Individual issues with location, severity, source
    • Location: Geographic data with coordinates and neighborhoods  
    • IssueRelationship: Connections between issues (geographic, temporal, similar)
    • IssueCluster: Grouped issues with patterns and themes
    • AgentMemory: Learning data for adaptive search strategies
    
    🧠 KNOWLEDGE GRAPH:
    • Nodes: Individual issues from Reddit and 311
    • Edges: Relationships (geographic, temporal, similarity)
    • Clusters: Grouped issues with common themes
    • Patterns: Recurring issue types and locations
    
    🗺️ REAL-TIME MAP DATA:
    • Issues with coordinates for map visualization
    • Hotspots by neighborhood with severity scores
    • Trends and patterns for insights
    • Cross-platform correlations (Reddit + 311)
    
    📈 INSIGHTS:
    • Geographic patterns (which neighborhoods have most issues)
    • Temporal patterns (when issues occur)
    • Cross-platform correlations (Reddit discussions + 311 reports)
    • Predictive insights (trending issues, seasonal patterns)
    """)

if __name__ == "__main__":
    show_expected_output()
