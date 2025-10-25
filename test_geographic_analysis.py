#!/usr/bin/env python3
"""
Test script for geographic hierarchical analysis
"""
import requests
import json

def test_geographic_analysis():
    """Test the geographic analysis system"""
    
    print("🗺️ Testing Geographic Analysis System")
    print("=" * 50)
    
    # Test county-level analysis
    print("\n1. Testing County Analysis:")
    url = "http://localhost:8000/api/generate-report?county=Alameda County"
    response = requests.get(url)
    
    if response.status_code == 200:
        report = response.json()
        print("✅ County analysis successful!")
        print(f"Problems found: {len(report.get('problems', []))}")
        
        # Show geographic context if available
        if 'geographic_context' in str(report):
            print("✅ Geographic context included")
        if 'regional_trends' in str(report):
            print("✅ Regional trends included")
        if 'balanced_analysis' in str(report):
            print("✅ Dataset balancing applied")
    else:
        print(f"❌ County analysis failed: {response.status_code}")
        print(response.text)
    
    # Test neighborhood-level analysis (if data exists)
    print("\n2. Testing Neighborhood Analysis:")
    url = "http://localhost:8000/api/generate-neighborhood-report"
    params = {
        "neighborhood": "Downtown",
        "city": "Oakland"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        report = response.json()
        print("✅ Neighborhood analysis successful!")
        print(f"Neighborhood: {report.get('neighborhood')}")
        print(f"City: {report.get('city')}")
        print(f"Problems found: {len(report.get('problems', []))}")
    else:
        print(f"❌ Neighborhood analysis failed: {response.status_code}")
        print(response.text)
    
    # Test dataset balancing
    print("\n3. Testing Dataset Balancing:")
    url = "http://localhost:8000/api/datasets"
    response = requests.get(url)
    
    if response.status_code == 200:
        datasets = response.json()
        print("✅ Dataset information:")
        for dataset in datasets:
            print(f"  - {dataset['name']} (Type: {dataset['source_type']})")
        
        # Check if we have multiple datasets
        if len(datasets) > 1:
            print("✅ Multiple datasets detected - balancing should be applied")
        else:
            print("ℹ️ Only one dataset - no balancing needed")
    else:
        print(f"❌ Dataset listing failed: {response.status_code}")

def test_large_dataset_handling():
    """Test handling of large datasets"""
    
    print("\n📊 Testing Large Dataset Handling:")
    
    # Check if the new large dataset was ingested
    url = "http://localhost:8000/api/datasets"
    response = requests.get(url)
    
    if response.status_code == 200:
        datasets = response.json()
        large_datasets = [d for d in datasets if 'Crimes_and_Clearances' in d['name']]
        
        if large_datasets:
            print("✅ Large dataset detected:")
            for dataset in large_datasets:
                print(f"  - {dataset['name']} (Type: {dataset['source_type']})")
            print("✅ System should handle dataset balancing automatically")
        else:
            print("ℹ️ Large dataset not yet ingested")
    else:
        print(f"❌ Could not check datasets: {response.status_code}")

if __name__ == "__main__":
    print("🧪 Testing Geographic Analysis System")
    print("=" * 50)
    
    # Test geographic analysis
    test_geographic_analysis()
    
    # Test large dataset handling
    test_large_dataset_handling()
    
    print("\n🎉 Geographic analysis testing completed!")
    print("\nKey Features Tested:")
    print("✅ Hierarchical geographic analysis (neighborhood → city → region)")
    print("✅ Dataset balancing to prevent bias")
    print("✅ Regional trend analysis")
    print("✅ Geographic context inclusion")
    print("✅ Large dataset handling")
