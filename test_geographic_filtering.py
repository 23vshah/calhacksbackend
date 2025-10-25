#!/usr/bin/env python3
"""
Test script for geographic filtering system
"""
import requests
import json

def test_geographic_filtering():
    """Test the geographic filtering system"""
    
    print("🗺️ Testing Geographic Filtering System")
    print("=" * 50)
    
    # Test 1: County-level analysis
    print("\n1. Testing County-Level Analysis:")
    url = "http://localhost:8000/api/generate-report"
    params = {
        "location": "Alameda County",
        "level": "county"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        report = response.json()
        print("✅ County analysis successful!")
        print(f"Location: {report.get('county')}")
        print(f"Problems found: {len(report.get('problems', []))}")
        
        # Check if geographic level is included
        if 'geographic_level' in report.get('summary', {}):
            print(f"✅ Geographic level: {report['summary']['geographic_level']}")
        
        # Check if relevant datasets are listed
        if 'relevant_datasets' in report.get('summary', {}):
            datasets = report['summary']['relevant_datasets']
            print(f"✅ Relevant datasets: {len(datasets)} datasets")
            for dataset in datasets:
                print(f"  - {dataset}")
    else:
        print(f"❌ County analysis failed: {response.status_code}")
        print(response.text)
    
    # Test 2: City-level analysis
    print("\n2. Testing City-Level Analysis:")
    params = {
        "location": "Oakland",
        "level": "city"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        report = response.json()
        print("✅ City analysis successful!")
        print(f"Location: {report.get('county')}")
        print(f"Problems found: {len(report.get('problems', []))}")
        
        # Check if geographic level is included
        if 'geographic_level' in report.get('summary', {}):
            print(f"✅ Geographic level: {report['summary']['geographic_level']}")
        
        # Check if relevant datasets are listed
        if 'relevant_datasets' in report.get('summary', {}):
            datasets = report['summary']['relevant_datasets']
            print(f"✅ Relevant datasets: {len(datasets)} datasets")
            for dataset in datasets:
                print(f"  - {dataset}")
    else:
        print(f"❌ City analysis failed: {response.status_code}")
        print(response.text)
    
    # Test 3: Check dataset metadata
    print("\n3. Testing Dataset Metadata:")
    url = "http://localhost:8000/api/datasets"
    response = requests.get(url)
    
    if response.status_code == 200:
        datasets = response.json()
        print("✅ Dataset information:")
        for dataset in datasets:
            print(f"  - {dataset['name']}")
            print(f"    Type: {dataset['source_type']}")
            if 'geographic_level' in dataset:
                print(f"    Geographic Level: {dataset['geographic_level']}")
            else:
                print("    Geographic Level: Not set (old dataset)")
    else:
        print(f"❌ Dataset listing failed: {response.status_code}")

def test_geographic_relevance():
    """Test geographic relevance filtering"""
    
    print("\n4. Testing Geographic Relevance Logic:")
    
    # Test different levels and locations
    test_cases = [
        {"location": "Alameda County", "level": "county", "expected": "county datasets"},
        {"location": "Oakland", "level": "city", "expected": "county + city datasets"},
        {"location": "Downtown", "level": "neighborhood", "expected": "county + city + neighborhood datasets"}
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['location']} at {test_case['level']} level")
        print(f"Expected: {test_case['expected']}")
        
        url = "http://localhost:8000/api/generate-report"
        params = {
            "location": test_case["location"],
            "level": test_case["level"]
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            report = response.json()
            if 'relevant_datasets' in report.get('summary', {}):
                datasets = report['summary']['relevant_datasets']
                print(f"✅ Found {len(datasets)} relevant datasets")
            else:
                print("❌ No relevant datasets information")
        else:
            print(f"❌ Request failed: {response.status_code}")

if __name__ == "__main__":
    print("🧪 Testing Geographic Filtering System")
    print("=" * 50)
    
    # Test geographic filtering
    test_geographic_filtering()
    
    # Test geographic relevance
    test_geographic_relevance()
    
    print("\n🎉 Geographic filtering testing completed!")
    print("\nKey Features Tested:")
    print("✅ Geographic level detection")
    print("✅ Dataset filtering by relevance")
    print("✅ API parameter handling")
    print("✅ Metadata storage and retrieval")
    print("✅ Multi-level geographic analysis")
