#!/usr/bin/env python3
"""
Test LLM analysis on SF permit CSV
"""
import pandas as pd
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.services.llm_service import LLMService

async def test_sf_csv_llm():
    """Test LLM analysis on SF permit CSV"""
    
    print("🏙️ Testing LLM Analysis on SF Permit CSV")
    print("=" * 50)
    
    # Read the CSV and get a sample
    df = pd.read_csv('sf_permit_events_large.csv')
    sample_size = min(50, len(df))
    sample_data = df.sample(n=sample_size, random_state=42).to_csv(index=False)
    
    print(f"📊 Sample data ({sample_size} rows):")
    print(sample_data[:500] + "...")
    print("\n" + "="*50)
    
    # Test LLM analysis with Claude
    llm_service = LLMService(provider="claude")
    
    try:
        print("🤖 Running Claude analysis...")
        profile = await llm_service.analyze_schema(sample_data)
        
        print("✅ LLM Analysis Results:")
        print(f"Data Type: {profile.data_type}")
        print(f"Geographic Level: {profile.geographic_level}")
        print(f"Time Granularity: {profile.time_granularity}")
        print(f"Geographic Column: {profile.geographic_column}")
        print(f"Time Column: {profile.time_column}")
        
        print(f"\n🌍 Geographic Hierarchy:")
        hierarchy = profile.geographic_hierarchy
        print(f"  Neighborhood: {hierarchy.neighborhood}")
        print(f"  City: {hierarchy.city}")
        print(f"  County: {hierarchy.county}")
        print(f"  State: {hierarchy.state}")
        print(f"  Region: {hierarchy.region}")
        
        print(f"\n📊 Metrics ({len(profile.metrics)}):")
        for metric in profile.metrics:
            print(f"  - {metric['column']} → {metric['normalized_name']} ({metric['unit']})")
        
        print(f"\n🏷️ Dimensions ({len(profile.dimensions)}):")
        for dim in profile.dimensions:
            print(f"  - {dim}")
        
        # Check if it correctly identified SF
        print(f"\n🔍 Geographic Analysis:")
        if profile.geographic_level == "neighborhood":
            print("✅ Correctly identified as neighborhood-level data")
        else:
            print(f"❌ Expected 'neighborhood', got '{profile.geographic_level}'")
        
        if "Inner Sunset" in sample_data:
            print("✅ Sample contains SF neighborhood data")
        else:
            print("❌ Sample doesn't contain SF neighborhood data")
        
        return profile
        
    except Exception as e:
        print(f"❌ LLM analysis failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(test_sf_csv_llm())
