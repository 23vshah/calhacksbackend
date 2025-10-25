#!/usr/bin/env python3
"""
Demo showing knowledge graph structure and querying
"""

def show_knowledge_graph_demo():
    print("=" * 80)
    print("KNOWLEDGE GRAPH STRUCTURE & QUERYING DEMO")
    print("=" * 80)
    
    print("\n1. GRAPH STRUCTURE:")
    print("-" * 40)
    print("""
    NODES (Issues):
    - reddit_001: "BART delays causing major commute issues"
      * Source: reddit
      * Location: SOMA, San Francisco
      * Severity: high
      * Keywords: [transportation, BART, delays]
      * Coordinates: (37.7749, -122.4194)
    
    - sf311_001: "Graffiti"
      * Source: 311
      * Location: Mission District, San Francisco
      * Severity: medium
      * Keywords: [graffiti, vandalism, commercial]
      * Coordinates: (37.7651, -122.4190)
    """)
    
    print("\n2. GRAPH EDGES (Relationships):")
    print("-" * 40)
    print("""
    GEOGRAPHIC RELATIONSHIPS:
    - reddit_001 <-> sf311_002 (same neighborhood: SOMA)
      * Strength: 0.73
      * Type: geographic
      * Reason: Both issues in SOMA area
    
    TEMPORAL RELATIONSHIPS:
    - reddit_003 <-> sf311_004 (reported same week)
      * Strength: 0.65
      * Type: temporal
      * Reason: Issues reported around same time
    
    SIMILARITY RELATIONSHIPS:
    - sf311_001 <-> sf311_005 (both graffiti issues)
      * Strength: 0.89
      * Type: similar
      * Reason: Same issue type and description
    """)
    
    print("\n3. GRAPH CLUSTERS:")
    print("-" * 40)
    print("""
    TRANSPORTATION CLUSTER:
    - Nodes: reddit_001, sf311_002, reddit_006
    - Common themes: [BART, delays, commute, parking]
    - Geographic: SOMA, Mission District
    - Severity: high (1), medium (2)
    
    GRAFFITI CLUSTER:
    - Nodes: sf311_001, sf311_005, reddit_007
    - Common themes: [graffiti, vandalism, commercial]
    - Geographic: Mission District, Castro
    - Severity: medium (3)
    
    HOMELESSNESS CLUSTER:
    - Nodes: reddit_003, sf311_004, reddit_008
    - Common themes: [homeless, encampment, tent, shelter]
    - Geographic: Tenderloin, Mission District
    - Severity: high (1), medium (2)
    """)
    
    print("\n4. GRAPH QUERYING:")
    print("-" * 40)
    print("""
    SPATIAL QUERIES:
    GET /api/knowledge-graph/query?query_type=spatial&neighborhood=Mission District
    - Returns: All issues in Mission District
    
    GET /api/knowledge-graph/query?query_type=spatial&coordinates=[37.7749,-122.4194]&radius=1.0
    - Returns: All issues within 1 mile of coordinates
    
    TEMPORAL QUERIES:
    GET /api/knowledge-graph/query?query_type=temporal&timeframe=week
    - Returns: All issues from the last week
    
    GET /api/knowledge-graph/query?query_type=temporal&start_date=2024-01-01&end_date=2024-01-31
    - Returns: All issues in January 2024
    
    SIMILARITY QUERIES:
    GET /api/knowledge-graph/query?query_type=similarity&issue_id=reddit_001&min_similarity=0.7
    - Returns: Issues similar to reddit_001 with 70%+ similarity
    
    CLUSTER QUERIES:
    GET /api/knowledge-graph/query?query_type=cluster&cluster_name=Transportation
    - Returns: All issues in the Transportation cluster
    """)
    
    print("\n5. API ENDPOINTS:")
    print("-" * 40)
    print("""
    GET /api/knowledge-graph/nodes
    - Parameters: severity, source, neighborhood, limit
    - Returns: List of graph nodes (issues)
    
    GET /api/knowledge-graph/edges
    - Parameters: min_strength, relationship_type, limit
    - Returns: List of graph edges (relationships)
    
    GET /api/knowledge-graph/clusters
    - Parameters: min_issues, limit
    - Returns: List of graph clusters
    
    GET /api/knowledge-graph/metrics
    - Returns: Graph statistics and metrics
    
    GET /api/knowledge-graph/query
    - Parameters: query_type, query_params
    - Returns: Query results based on type
    """)
    
    print("\n6. FRONTEND VISUALIZATION:")
    print("-" * 40)
    print("""
    KNOWLEDGE GRAPH COMPONENT:
    - Interactive graph visualization
    - Node filtering by severity, source, neighborhood
    - Edge filtering by relationship type and strength
    - Cluster analysis and visualization
    - Real-time metrics and statistics
    
    GRAPH VIEWS:
    - Graph View: Interactive node-edge visualization
    - Clusters View: Cluster analysis and patterns
    - Timeline View: Temporal analysis of issues
    
    INTERACTIVE FEATURES:
    - Click nodes to see details
    - Click clusters to analyze patterns
    - Filter by severity, source, location
    - Zoom and pan for exploration
    - Export data for analysis
    """)
    
    print("\n7. EXAMPLE QUERIES:")
    print("-" * 40)
    print("""
    # Get all high severity issues
    GET /api/knowledge-graph/nodes?severity=high
    
    # Get all Reddit issues in Mission District
    GET /api/knowledge-graph/nodes?source=reddit&neighborhood=Mission District
    
    # Get all geographic relationships
    GET /api/knowledge-graph/edges?relationship_type=geographic
    
    # Get clusters with at least 3 issues
    GET /api/knowledge-graph/clusters?min_issues=3
    
    # Find issues similar to a specific issue
    GET /api/knowledge-graph/query?query_type=similarity&issue_id=reddit_001&min_similarity=0.8
    
    # Get all issues in the last week
    GET /api/knowledge-graph/query?query_type=temporal&timeframe=week
    """)
    
    print("\n8. GRAPH METRICS:")
    print("-" * 40)
    print("""
    NODE METRICS:
    - Total nodes: 50 (25 Reddit + 25 SF311)
    - Node types: transportation, graffiti, homelessness, etc.
    - Severity distribution: high (10), medium (25), low (15)
    - Geographic coverage: 8 SF neighborhoods
    
    EDGE METRICS:
    - Total relationships: 75
    - Geographic relationships: 30 (40%)
    - Temporal relationships: 20 (27%)
    - Similarity relationships: 25 (33%)
    - Average relationship strength: 0.72
    
    CLUSTER METRICS:
    - Total clusters: 5
    - Largest cluster: Transportation (8 nodes)
    - Most connected node: sf311_001 (6 relationships)
    - Cluster density: 0.65
    """)
    
    print("\n" + "=" * 80)
    print("SUMMARY: KNOWLEDGE GRAPH CAPABILITIES")
    print("=" * 80)
    print("""
    GRAPH STRUCTURE:
    - Nodes: Individual issues from Reddit and SF311
    - Edges: Relationships with strength scores
    - Clusters: Grouped issues with common themes
    - Metadata: Rich information for each node
    
    QUERYING CAPABILITIES:
    - Spatial: Find issues by location
    - Temporal: Find issues by time
    - Similarity: Find similar issues
    - Cluster: Analyze issue clusters
    
    VISUALIZATION:
    - Interactive graph visualization
    - Multiple view modes (graph, clusters, timeline)
    - Real-time filtering and exploration
    - Export capabilities for analysis
    
    API INTEGRATION:
    - RESTful endpoints for all graph operations
    - Flexible querying with multiple parameters
    - Real-time metrics and statistics
    - Easy integration with frontend applications
    """)

if __name__ == "__main__":
    show_knowledge_graph_demo()
