#!/usr/bin/env python3
"""
Test script for SF311 API endpoint
"""

import requests
import json
import time

def test_sf311_endpoint():
    """Test the SF311 agent launch endpoint"""
    
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/api/agents/sf311/launch"
    
    print("🚀 Testing SF311 Agent Endpoint")
    print(f"📡 URL: {endpoint}")
    print("-" * 50)
    
    # Test parameters
    params = {
        "pages": 1,  # Start with fewer pages for testing
        "min_severity": "low"
    }
    
    try:
        print("⏳ Launching SF311 agent...")
        start_time = time.time()
        
        response = requests.post(endpoint, params=params, timeout=300)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"⏱️  Request completed in {execution_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(data)
            
            print("✅ Success!")
            print(f"📈 Issues found: {len(data.get('issues', []))}")
            print(f"📋 Raw requests: {len(data.get('raw_requests', []))}")
            
            if data.get('summary'):
                summary = data['summary']
                print(f"📊 Total requests processed: {summary.get('total_requests', 0)}")
                print(f"🔍 Filtered requests: {summary.get('filtered_requests', 0)}")
                print(f"🎯 Issues identified: {summary.get('issues_identified', 0)}")
            print(issues)
            # Show sample issues
            issues = data.get('issues', [])
            if issues:
                print("\n🔍 Sample Issues:")
                for i, issue in enumerate(issues[:3]):  # Show first 3
                    print(f"  {i+1}. {issue.get('title', 'Unknown')}")
                    print(f"     Severity: {issue.get('severity', 'Unknown')}")
                    print(f"     Neighborhood: {issue.get('neighborhood', 'Unknown')}")
                    if issue.get('coordinates'):
                        coords = issue['coordinates']
                        print(f"     Coordinates: {coords[1]:.6f}, {coords[0]:.6f}")
                    print()
            
            # Show sample raw requests
            requests_data = data.get('raw_requests', [])
            if requests_data:
                print("📋 Sample Raw Requests:")
                for i, req in enumerate(requests_data[:3]):  # Show first 3
                    print(f"  {i+1}. {req.get('offense_type', 'Unknown')}")
                    print(f"     Address: {req.get('address', 'Unknown')}")
                    print(f"     Severity: {req.get('severity', 'Unknown')}")
                    print()
            
            # Show insights
            insights = data.get('summary', {}).get('insights', [])
            if insights:
                print("💡 Insights:")
                for insight in insights:
                    print(f"  • {insight}")
            
        else:
            print("❌ Error!")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error!")
        print("Make sure the backend server is running on http://localhost:8000")
        print("Start it with: cd backend && python -m uvicorn app.main:app --reload")
        
    except requests.exceptions.Timeout:
        print("⏰ Request timed out!")
        print("The SF311 agent might be taking longer than expected.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def test_health_endpoint():
    """Test the health endpoint"""
    
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/health"
    
    print("\n🏥 Testing Health Endpoint")
    print(f"📡 URL: {endpoint}")
    print("-" * 30)
    
    try:
        response = requests.get(endpoint, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is healthy!")
            print(f"📊 Status: {data.get('status', 'Unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")

if __name__ == "__main__":
    print("🧪 SF311 API Endpoint Tester")
    print("=" * 50)
    
    # Test health first
    test_health_endpoint()
    
    # Test SF311 endpoint
    test_sf311_endpoint()
    
    print("\n✨ Testing completed!")
