#!/usr/bin/env python3
"""
Simple test to verify Claude API is working
"""
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

def test_claude_api():
    """Test Claude API with a simple request"""
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
    
    print(f"Testing Claude API with model: {model}")
    print(f"API Key present: {'Yes' if api_key else 'No'}")
    
    if not api_key:
        print("❌ No API key found!")
        return False
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Simple test request
        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello and return a simple JSON: {\"status\": \"working\"}"}]
        )
        
        content = response.content[0].text
        print(f"✅ Claude API working!")
        print(f"Response: {content}")
        return True
        
    except Exception as e:
        print(f"❌ Claude API failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_claude_api()
