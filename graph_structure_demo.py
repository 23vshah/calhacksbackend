#!/usr/bin/env python3
"""
Demo showing the knowledge graph structure and visualization
"""

def show_graph_structure():
    print("=" * 80)
    print("KNOWLEDGE GRAPH STRUCTURE & VISUALIZATION")
    print("=" * 80)
    
    print("\n1. GRAPH NODES (Issues):")
    print("-" * 40)
    print("""
    REDDIT NODES:
    - Node ID: reddit_001
    - Title: "BART delays causing major commute issues"
    - Source: reddit
    - Location: SOMA, San Francisco
    - Severity: high
    - Keywords: ["transportation", "BART", "delays"]
    - Metadata: {subreddit: "sftransportation", score: 25}
    
    SF311 NODES:
    - Node ID: sf311_001  
    - Title: "Graffiti"
    - Source: 311
    - Location: Mission District, San Francisco
    - Severity: medium
    - Keywords: ["graffiti", "vandalism", "commercial"]
    - Metadata: {offense_type: "Graffiti", coordinates: "(37.7651, -122.4190)"}
    """)
    
    print("\n2. GRAPH EDGES (Relationships):")
    print("-" * 40)
    print("""
    GEOGRAPHIC RELATIONSHIPS:
    - reddit_001 ↔ sf311_002 (same neighborhood: SOMA)
    - Strength: 0.73
    - Type: geographic
    - Reason: Both issues in SOMA area
    
    TEMPORAL RELATIONSHIPS:
    - reddit_003 ↔ sf311_004 (reported same week)
    - Strength: 0.65
    - Type: temporal  
    - Reason: Issues reported around same time
    
    SIMILARITY RELATIONSHIPS:
    - sf311_001 ↔ sf311_005 (both graffiti issues)
    - Strength: 0.89
    - Type: similar
    - Reason: Same issue type and description
    """)
    
    print("\n3. GRAPH CLUSTERS:")
    print("-" * 40)
    print("""
    TRANSPORTATION CLUSTER:
    - Nodes: reddit_001, sf311_002, reddit_006
    - Common themes: ["BART", "delays", "commute", "parking"]
    - Geographic: SOMA, Mission District
    - Severity: high (1), medium (2)
    
    GRAFFITI CLUSTER:
    - Nodes: sf311_001, sf311_005, reddit_007
    - Common themes: ["graffiti", "vandalism", "commercial"]
    - Geographic: Mission District, Castro
    - Severity: medium (3)
    
    HOMELESSNESS CLUSTER:
    - Nodes: reddit_003, sf311_004, reddit_008
    - Common themes: ["homeless", "encampment", "tent", "shelter"]
    - Geographic: Tenderloin, Mission District
    - Severity: high (1), medium (2)
    """)
    
    print("\n4. GRAPH VISUALIZATION (ASCII):")
    print("-" * 40)
    print("""
    KNOWLEDGE GRAPH STRUCTURE:
    
    ┌─────────────────────────────────────────────────────────────┐
    │  REDDIT NODES (Community Discussions)                     │
    └─────────────────────────────────────────────────────────────┘
    
    reddit_001: "BART delays" [SOMA, Transportation, HIGH]
         │
         │ Geographic (0.73)
         │
    ┌────▼────┐    ┌─────────────────────────────────────────┐
    │ sf311_002│    │  SF311 NODES (Official Reports)        │
    │"Parking" │    └─────────────────────────────────────────┘
    │[SOMA,    │
    │ Parking, │
    │ MEDIUM]  │
    └────┬────┘
         │
         │ Temporal (0.65)
         │
    ┌────▼────┐
    │ reddit_003│
    │"Homeless"│
    │[Mission,  │
    │ Homeless,│
    │ HIGH]    │
    └────┬────┘
         │
         │ Similar (0.89)
         │
    ┌────▼────┐
    │ sf311_004│
    │"Cleaning"│
    │[Mission, │
    │ Litter,  │
    │ LOW]     │
    └──────────┘
    
    CLUSTERS:
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │ Transportation  │  │    Graffiti     │  │  Homelessness   │
    │ Cluster         │  │    Cluster      │  │    Cluster      │
    │ 3 nodes         │  │ 3 nodes         │  │ 3 nodes         │
    │ SOMA + Mission  │  │ Mission + Castro│  │ Tenderloin +    │
    │ High severity   │  │ Medium severity  │  │ Mission         │
    └─────────────────┘  └─────────────────┘  └─────────────────┘
    """)
    
    print("\n5. GRAPH QUERY EXAMPLES:")
    print("-" * 40)
    print("""
    SPATIAL QUERIES:
    - "Show all issues in Mission District"
    - "Find issues within 1 mile of coordinates (37.7749, -122.4194)"
    - "Get hotspots by neighborhood"
    
    TEMPORAL QUERIES:
    - "Show issues reported in the last week"
    - "Find issues that occurred around the same time"
    - "Get trending issues over time"
    
    SIMILARITY QUERIES:
    - "Find issues similar to 'graffiti'"
    - "Show related transportation problems"
    - "Get issues with similar severity"
    
    CLUSTER QUERIES:
    - "Show all issues in the transportation cluster"
    - "Find the largest issue cluster"
    - "Get cluster patterns and themes"
    """)
    
    print("\n6. GRAPH METRICS:")
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

if __name__ == "__main__":
    show_graph_structure()

