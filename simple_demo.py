#!/usr/bin/env python3
"""
Simple demo showing agent system output
"""

def show_simple_demo():
    print("=" * 60)
    print("AGENT SYSTEM OUTPUT DEMO")
    print("=" * 60)
    
    print("\n1. REDDIT AGENT FINDS:")
    print("   - BART delays in SOMA (Reddit post)")
    print("   - Homeless encampment in Mission District (Reddit post)")
    print("   - Traffic issues in Castro (Reddit post)")
    
    print("\n2. SF311 AGENT FINDS:")
    print("   - Graffiti at 2972 16th St, Mission District")
    print("   - Mattress dumping at 4214 Geary Blvd, Richmond")
    print("   - Blocked driveway at 1316 Pine St")
    
    print("\n3. KNOWLEDGE GRAPH CONNECTS:")
    print("   - BART delays (Reddit) <-> Blocked driveway (311)")
    print("     Relationship: Geographic (same area)")
    print("   - Homeless encampment (Reddit) <-> Street cleaning (311)")
    print("     Relationship: Temporal (reported around same time)")
    
    print("\n4. CLUSTERS IDENTIFIED:")
    print("   - Transportation Cluster: BART delays + Parking issues")
    print("   - Graffiti Cluster: Multiple graffiti reports")
    print("   - Homelessness Cluster: Encampments + Street cleaning")
    
    print("\n5. DATABASE STORES:")
    print("   - CommunityIssue: Individual issues with location")
    print("   - Location: Coordinates and neighborhoods")
    print("   - IssueRelationship: Connections between issues")
    print("   - IssueCluster: Grouped issues with patterns")
    print("   - AgentMemory: Learning data for adaptation")
    
    print("\n6. MAP DATA FOR FRONTEND:")
    print("   - Issues with coordinates for visualization")
    print("   - Hotspots by neighborhood (Mission District: 8 issues)")
    print("   - Trends and patterns for insights")
    print("   - Cross-platform correlations")
    
    print("\n7. KNOWLEDGE GRAPH STRUCTURE:")
    print("""
    Issues (Nodes):
    Reddit: "BART delays" ────┐
    [SOMA, Transportation]    │
                               │ Geographic Relationship (0.73)
                               │
    311: "Blocked driveway"  │
    [Mission, Parking]        │
                               │
    Reddit: "Homeless camp"  │
    [Mission, Homelessness]  │
                               │ Temporal Relationship (0.65)
                               │
    311: "Street cleaning"   │
    [Richmond, Litter]        │
    """)
    
    print("\n8. INSIGHTS GENERATED:")
    print("   - Mission District has most issues (12 total)")
    print("   - Transportation problems concentrated in SOMA")
    print("   - Graffiti increases 40% after major events")
    print("   - Homelessness and litter show temporal correlation")
    
    print("\n9. REAL-TIME FEATURES:")
    print("   - Live updates as new issues are found")
    print("   - Adaptive search strategies that learn")
    print("   - Geographic intelligence with neighborhood detection")
    print("   - Cross-platform correlation (Reddit + 311)")
    
    print("\n" + "=" * 60)
    print("SUMMARY: What You Get")
    print("=" * 60)
    print("""
    DATA STORAGE:
    - Individual issues with location, severity, source
    - Geographic data with coordinates and neighborhoods
    - Relationships between issues (geographic, temporal, similar)
    - Clusters of related issues with patterns
    - Learning data for adaptive search strategies
    
    KNOWLEDGE GRAPH:
    - Nodes: Individual issues from Reddit and 311
    - Edges: Relationships with strength scores
    - Clusters: Grouped issues with common themes
    - Patterns: Recurring issue types and locations
    
    REAL-TIME MAP DATA:
    - Issues with coordinates for map visualization
    - Hotspots by neighborhood with severity scores
    - Trends and patterns for insights
    - Cross-platform correlations (Reddit + 311)
    
    INSIGHTS:
    - Geographic patterns (which neighborhoods have most issues)
    - Temporal patterns (when issues occur)
    - Cross-platform correlations (Reddit discussions + 311 reports)
    - Predictive insights (trending issues, seasonal patterns)
    """)

if __name__ == "__main__":
    show_simple_demo()

