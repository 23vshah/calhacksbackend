#!/usr/bin/env python3
"""
Test script for CSV ingestion
"""
import requests
import json

def test_csv_ingestion():
    """Test the CSV ingestion endpoint with all SF datasets"""
    
    url = "http://localhost:8000/api/ingest-data"
    
    # SF datasets to process
    sf_datasets = [
        {
            "file": "sf_311_events_large.csv",
            "name": "SF_311_Events",
            "type": "311"
        },
        {
            "file": "sf_crime_events_large.csv", 
            "name": "SF_Crime_Events",
            "type": "crime"
        },
        {
            "file": "sf_neighborhood_indicators_large.csv",
            "name": "SF_Neighborhood_Indicators", 
            "type": "demographics"
        },
        {
            "file": "sf_permit_events_large.csv",
            "name": "SF_Permit_Events",
            "type": "housing"
        }
    ]
    
    dataset_ids = []
    
    for dataset in sf_datasets:
        print(f"\n📁 Processing {dataset['name']}...")
        
        try:
            with open(dataset["file"], "rb") as f:
                files = {"file": (dataset["file"], f, "text/csv")}
                data = {
                    "dataset_name": dataset["name"],
                    "source_type": dataset["type"]
                }
                
                print(f"Uploading {dataset['file']}...")
                response = requests.post(url, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ {dataset['name']} ingestion successful!")
                    print(f"Dataset ID: {result['dataset_id']}")
                    print(f"Rows processed: {result['rows_processed']}")
                    print(f"Message: {result['message']}")
                    dataset_ids.append(result['dataset_id'])
                else:
                    print(f"❌ {dataset['name']} ingestion failed: {response.status_code}")
                    print(response.text)
                    
        except FileNotFoundError:
            print(f"❌ File not found: {dataset['file']}")
        except Exception as e:
            print(f"❌ Error processing {dataset['name']}: {e}")
    
    print(f"\n🎉 Processed {len(dataset_ids)} SF datasets successfully!")
    return dataset_ids

def test_report_generation(location="Butte County", level="county"):
    """Test the report generation endpoint"""
    
    url = f"http://localhost:8000/api/generate-report?location={location}&level={level}"
    
    print(f"\nGenerating report for {location} at {level} level...")
    response = requests.get(url)
    
    if response.status_code == 200:
        report = response.json()
        print("✅ Report generation successful!")
        print(f"Location: {report['county']}")
        print(f"Problems found: {len(report['problems'])}")
        
        # Check if geographic level is included
        if 'geographic_level' in report.get('summary', {}):
            print(f"Geographic Level: {report['summary']['geographic_level']}")
        
        # Check relevant datasets
        if 'relevant_datasets' in report.get('summary', {}):
            datasets = report['summary']['relevant_datasets']
            print(f"Relevant datasets: {len(datasets)} datasets")
            for dataset in datasets:
                print(f"  - {dataset}")
        
        for i, problem in enumerate(report['problems'], 1):
            print(f"\nProblem {i}:")
            print(f"  Title: {problem['title']}")
            print(f"  Severity: {problem['severity']}")
            print(f"  Description: {problem['description']}")
            print(f"  Solution: {problem['solution']['title']}")
        
        return report
    else:
        print(f"❌ Report generation failed: {response.status_code}")
        print(response.text)
        return None

def test_llm_info():
    """Test LLM provider info"""
    
    url = "http://localhost:8000/llm-info"
    
    print("\nChecking LLM provider...")
    response = requests.get(url)
    
    if response.status_code == 200:
        info = response.json()
        print("✅ LLM info successful!")
        print(f"Provider: {info.get('provider', 'unknown')}")
        print(f"Model: {info.get('model', 'unknown')}")
        return info
    else:
        print(f"❌ LLM info failed: {response.status_code}")
        print(response.text)
        return None

def test_list_datasets():
    """Test listing datasets"""
    
    url = "http://localhost:8000/api/datasets"
    
    print("\nListing datasets...")
    response = requests.get(url)
    
    if response.status_code == 200:
        datasets = response.json()
        print("✅ Dataset listing successful!")
        print(f"Found {len(datasets)} datasets:")
        
        for dataset in datasets:
            print(f"  - {dataset['name']} (ID: {dataset['id']}, Type: {dataset['source_type']})")
        
        return datasets
    else:
        print(f"❌ Dataset listing failed: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    print("🧪 Testing Theages Backend API with SF Datasets")
    print("=" * 50)
    
    # Test LLM provider info
    test_llm_info()
    
    # Test CSV ingestion for all SF datasets
    # dataset_ids = test_csv_ingestion()
    dataset_ids = True
    if dataset_ids:
        # Test listing datasets
        test_list_datasets()
        
        # Test report generation with different SF locations and levels
        print("\n" + "="*50)
        print("Testing SF geographic analysis:")
        
        # Test SF city level
        test_report_generation("San Francisco", "city")
        
        # Test SF neighborhood level
        test_report_generation("Inner Sunset", "neighborhood")
        
        # Test SF neighborhood level (different neighborhood)
        test_report_generation("Mission District", "neighborhood")
    
    print("\n🎉 SF dataset testing completed!")
