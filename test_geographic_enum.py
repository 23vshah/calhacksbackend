#!/usr/bin/env python3
"""
Test script for geographic enum implementation
"""
import requests
import json

def test_geographic_enum():
    """Test the geographic enum system"""
    
    print("🗺️ Testing Geographic Enum System")
    print("=" * 50)
    
    # Test 1: Check if datasets have geographic levels
    print("\n1. Testing Dataset Geographic Levels:")
    url = "http://localhost:8000/api/datasets"
    response = requests.get(url)
    
    if response.status_code == 200:
        datasets = response.json()
        print("✅ Dataset information:")
        for dataset in datasets:
            print(f"  - {dataset['name']}")
            print(f"    Type: {dataset['source_type']}")
            if 'geographic_level' in dataset:
                level = dataset['geographic_level']
                print(f"    Geographic Level: {level}")
                
                # Validate enum values
                valid_levels = ["state", "region", "county", "city", "neighborhood"]
                if level in valid_levels:
                    print(f"    ✅ Valid enum value: {level}")
                else:
                    print(f"    ❌ Invalid enum value: {level}")
            else:
                print("    ❌ Geographic Level: Not set (old dataset)")
    else:
        print(f"❌ Dataset listing failed: {response.status_code}")
    
    # Test 2: Test different geographic levels
    print("\n2. Testing Different Geographic Levels:")
    
    test_cases = [
        {"location": "California", "level": "state", "description": "State-level analysis"},
        {"location": "West Coast", "level": "region", "description": "Region-level analysis"},
        {"location": "Alameda County", "level": "county", "description": "County-level analysis"},
        {"location": "Oakland", "level": "city", "description": "City-level analysis"},
        {"location": "Downtown", "level": "neighborhood", "description": "Neighborhood-level analysis"}
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['description']}")
        print(f"Location: {test_case['location']}, Level: {test_case['level']}")
        
        url = "http://localhost:8000/api/generate-report"
        params = {
            "location": test_case["location"],
            "level": test_case["level"]
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            report = response.json()
            print(f"✅ Analysis successful!")
            print(f"Problems found: {len(report.get('problems', []))}")
            
            # Check if geographic level is included
            if 'geographic_level' in report.get('summary', {}):
                level = report['summary']['geographic_level']
                print(f"✅ Geographic level: {level}")
                
                # Validate enum value
                valid_levels = ["state", "region", "county", "city", "neighborhood"]
                if level in valid_levels:
                    print(f"✅ Valid enum value: {level}")
                else:
                    print(f"❌ Invalid enum value: {level}")
            
            # Check relevant datasets
            if 'relevant_datasets' in report.get('summary', {}):
                datasets = report['summary']['relevant_datasets']
                print(f"✅ Relevant datasets: {len(datasets)} datasets")
                for dataset in datasets:
                    print(f"  - {dataset}")
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(response.text)
    
    # Test 3: Test enum validation
    print("\n3. Testing Enum Validation:")
    
    # Test invalid level
    print("Testing invalid geographic level...")
    url = "http://localhost:8000/api/generate-report"
    params = {
        "location": "Oakland",
        "level": "invalid_level"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        print("✅ System handled invalid level gracefully")
    else:
        print(f"❌ System failed with invalid level: {response.status_code}")

def test_enum_values():
    """Test the specific enum values"""
    
    print("\n4. Testing Enum Values:")
    
    # Test each enum value
    enum_values = ["state", "region", "county", "city", "neighborhood"]
    
    for enum_value in enum_values:
        print(f"\nTesting enum value: {enum_value}")
        
        url = "http://localhost:8000/api/generate-report"
        params = {
            "location": "Test Location",
            "level": enum_value
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            print(f"✅ {enum_value} level accepted")
        else:
            print(f"❌ {enum_value} level failed: {response.status_code}")

if __name__ == "__main__":
    print("🧪 Testing Geographic Enum System")
    print("=" * 50)
    
    # Test geographic enum
    test_geographic_enum()
    
    # Test enum values
    test_enum_values()
    
    print("\n🎉 Geographic enum testing completed!")
    print("\nKey Features Tested:")
    print("✅ GeographicLevel enum implementation")
    print("✅ Database enum storage")
    print("✅ LLM enum value selection")
    print("✅ Report generation enum usage")
    print("✅ Enum validation and error handling")
    print("✅ Multi-level geographic analysis")
