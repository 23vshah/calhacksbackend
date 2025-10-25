#!/usr/bin/env python3
"""
Demo script for Checkpoint 2: Adaptive City Goals
Shows how to train Theages on city goals and get goal-aligned recommendations
"""

import asyncio
import json
import requests
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:8000/api"

async def demo_adaptive_goals():
    """Demonstrate the adaptive city goals feature"""
    
    print("🎯 Checkpoint 2 Demo: Adaptive City Goals")
    print("=" * 50)
    
    # Step 1: Train Oakland with some goals
    print("\n1. Training Oakland with city goals...")
    
    oakland_goals = {
        "city_name": "Oakland",
        "goals": [
            {
                "city_name": "Oakland",
                "goal_title": "Reduce Homelessness",
                "goal_description": "Comprehensive approach to reduce homelessness by 20% through housing-first initiatives, job training, and mental health services",
                "target_metric": "Reduce homelessness by 20%",
                "target_value": 20.0,
                "target_unit": "percentage",
                "priority_level": "high",
                "deadline": (datetime.now() + timedelta(days=365)).isoformat(),
                "metadata_json": json.dumps({
                    "focus_areas": ["housing", "mental_health", "employment"],
                    "target_population": "unsheltered_individuals"
                })
            },
            {
                "city_name": "Oakland",
                "goal_title": "Improve Public Transit",
                "goal_description": "Enhance public transit efficiency and accessibility to reduce car dependency and improve air quality",
                "target_metric": "Increase transit ridership by 15%",
                "target_value": 15.0,
                "target_unit": "percentage",
                "priority_level": "medium",
                "deadline": (datetime.now() + timedelta(days=730)).isoformat(),
                "metadata_json": json.dumps({
                    "focus_areas": ["transit", "environment", "accessibility"],
                    "target_population": "all_residents"
                })
            }
        ],
        "policy_documents": [
            {
                "source": "HUD",
                "title": "Housing First Best Practices",
                "content": "Housing First is an evidence-based approach that provides permanent housing as quickly as possible for people experiencing homelessness, then provides supportive services as needed. This approach has been proven to reduce homelessness, improve health outcomes, and lower costs to public systems.",
                "document_type": "best_practice",
                "geographic_scope": "national",
                "topic_tags": ["housing", "homelessness", "social_services"]
            },
            {
                "source": "Urban Institute",
                "title": "Transit-Oriented Development Strategies",
                "content": "Transit-oriented development (TOD) creates compact, walkable, mixed-use communities centered around high-quality public transportation. This approach reduces car dependency, improves air quality, and creates more affordable housing options near transit hubs.",
                "document_type": "research",
                "geographic_scope": "national",
                "topic_tags": ["transit", "development", "sustainability"]
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/goals/train", json=oakland_goals)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Successfully trained {result['goals_created']} goals and {result['policies_created']} policies")
            print(f"📊 Vector index now has {result['vector_index_stats']['total_vectors']} vectors")
        else:
            print(f"❌ Error training goals: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Error training goals: {e}")
        return
    
    # Step 2: Get city stats
    print("\n2. Checking Oakland's training status...")
    
    try:
        response = requests.get(f"{BASE_URL}/goals/Oakland/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Training Status: {stats['training_status']}")
            print(f"📈 Active Goals: {stats['goals']['active']}")
            print(f"🎯 Completed Goals: {stats['goals']['completed']}")
            print(f"🧠 Vector Index: {stats['vector_index']['total_vectors']} vectors")
        else:
            print(f"❌ Error getting stats: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
    
    # Step 3: Get goal-aligned recommendations
    print("\n3. Getting goal-aligned recommendations for a problem...")
    
    problem_data = {
        "problem_description": "Rising unemployment in industrial zones, particularly affecting young adults aged 18-29",
        "current_data": {
            "unemployment_rate": 12.5,
            "youth_unemployment": 18.2,
            "industrial_vacancy": 25.3,
            "transit_accessibility": "limited"
        }
    }
    
    try:
        params = {
            "problem_description": problem_data["problem_description"],
            "current_data": json.dumps(problem_data["current_data"]),
            "max_recommendations": "3"
        }
        
        response = requests.get(f"{BASE_URL}/goals/Oakland/recommendations", params=params)
        if response.status_code == 200:
            recommendations = response.json()
            print(f"✅ Found {recommendations['total_found']} goal-aligned recommendations")
            
            for i, rec in enumerate(recommendations['recommendations'], 1):
                print(f"\n📋 Recommendation {i}:")
                print(f"   🎯 Goal: {rec['city_goal']['metadata']['original_text'][:80]}...")
                print(f"   📚 Policy: {rec['policy_document']['metadata']['source']} - {rec['policy_document']['metadata']['original_text'][:60]}...")
                print(f"   🎯 Match Score: {rec['similarity_score']:.2%}")
                print(f"   💡 Solution: {rec['recommendation_text'][:100]}...")
        else:
            print(f"❌ Error getting recommendations: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error getting recommendations: {e}")
    
    # Step 4: Show the RAG synthesis
    print("\n4. Generating comprehensive goal-aligned report...")
    
    try:
        synthesis_request = {
            "city_name": "Oakland",
            "problem_description": problem_data["problem_description"],
            "current_data": problem_data["current_data"],
            "max_recommendations": 3
        }
        
        response = requests.post(f"{BASE_URL}/goals/Oakland/synthesize", json=synthesis_request)
        if response.status_code == 200:
            synthesis = response.json()
            print(f"✅ Generated comprehensive synthesis for {synthesis['city_name']}")
            print(f"🎯 City Goals: {len(synthesis['city_goals'])} active goals")
            print(f"📋 Recommendations: {len(synthesis['recommendations'])} goal-aligned solutions")
            print(f"🤖 AI Synthesis: Generated at {synthesis['generated_at']}")
        else:
            print(f"❌ Error generating synthesis: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error generating synthesis: {e}")
    
    print("\n🎉 Checkpoint 2 Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ City goals training with vector embeddings")
    print("✅ Policy document ingestion and indexing")
    print("✅ RAG-based goal-aligned recommendations")
    print("✅ AI synthesis combining goals with data analysis")
    print("\nNext Steps:")
    print("🌐 Visit http://localhost:5173/goals to manage city goals in the UI")
    print("🗺️ Visit http://localhost:5173 to see goal-aligned recommendations in reports")

if __name__ == "__main__":
    asyncio.run(demo_adaptive_goals())

