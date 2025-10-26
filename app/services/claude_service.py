import json
import os
from typing import Dict, List, Any
import anthropic
from dotenv import load_dotenv
from app.models import NormalizationProfile, MetricWeight

# Load environment variables
load_dotenv()

class ClaudeService:
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    
    async def analyze_schema(self, sample_data: str) -> NormalizationProfile:
        """Analyze dataset sample and return normalization profile"""
        
        prompt = f"""You are a data analyst. Analyze this dataset sample:
{sample_data}

Respond in JSON format:
{{
  "data_type": "crime" | "demographics" | "economic" | "housing" | "311" | "other",
  "geographic_level": "county" | "city" | "zip" | "address",
  "time_granularity": "yearly" | "monthly" | "daily" | "none",
  "metrics": [
    {{"column": "VIOLENT", "normalized_name": "violent_arrests", "description": "Violent crime arrests", "unit": "count"}},
    {{"column": "PROPERTY", "normalized_name": "property_arrests", "description": "Property crime arrests", "unit": "count"}},
    ...
  ],
  "dimensions": ["GENDER", "RACE", "AGE_GROUP"],
  "geographic_column": "COUNTY",
  "time_column": "YEAR"
}}

Be specific about what each column measures and provide clear normalized names."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response
            content = response.content[0].text
            profile_data = json.loads(content)
            return NormalizationProfile(**profile_data)
            
        except Exception as e:
            raise Exception(f"Claude schema analysis failed: {str(e)}")
    
    async def calculate_weights(self, metrics: List[str]) -> List[MetricWeight]:
        """Calculate importance weights for metrics"""
        
        metrics_list = ", ".join(metrics)
        
        prompt = f"""You are an urban planning expert. Given these metrics across various datasets:
{metrics_list}

Assign importance weights (0-1) for detecting community problems:
- Economic decline
- Public safety issues  
- Housing crisis
- Infrastructure gaps

Return JSON array:
[
  {{"metric_name": "violent_arrests", "weight": 0.9, "reasoning": "Strong indicator of public safety issues"}},
  {{"metric_name": "property_arrests", "weight": 0.7, "reasoning": "Moderate indicator of economic distress"}},
  ...
]

Provide weights between 0-1 and clear reasoning for each."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            weights_data = json.loads(content)
            return [MetricWeight(**weight) for weight in weights_data]
            
        except Exception as e:
            raise Exception(f"Claude weight calculation failed: {str(e)}")
    
    async def synthesize_problems(self, county_data: Dict[str, Any], county: str) -> List[Dict[str, Any]]:
        """Synthesize problems and solutions from county data"""
        
        # Format data for Claude
        data_summary = []
        for metric, value in county_data.items():
            data_summary.append(f"- {metric}: {value}")
        
        data_text = "\n".join(data_summary)
        
        prompt = f"""Based on this {county} data:
{data_text}

Generate 1-3 most pressing community problems. For each problem:

1. Problem title (concise, 5-8 words)
2. Problem description (2-3 sentences explaining the issue)
3. Actionable solution (specific policy recommendation)
4. Expected impact (quantifiable outcome)

Return JSON array:
[
  {{
    "title": "Elevated Youth Arrest Rates",
    "description": "18-29 age group shows 150% higher arrest rates than state average, indicating potential systemic issues with youth engagement and opportunity.",
    "solution": {{
      "title": "Youth Intervention Programs",
      "description": "Implement community-based diversion programs and mentorship initiatives targeting at-risk youth.",
      "estimated_cost": "$2.5M annually",
      "expected_impact": "20-30% reduction in youth recidivism within 2 years"
    }},
    "severity": "high",
    "metrics": {{
      "violent_arrests": 949,
      "property_arrests": 1593,
      "state_comparison": "+45%"
    }}
  }}
]

Focus on actionable, data-driven insights."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            problems = json.loads(content)
            return problems
            
        except Exception as e:
            raise Exception(f"Claude problem synthesis failed: {str(e)}")
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a generic response from Claude"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            return content
            
        except Exception as e:
            raise Exception(f"Claude response generation failed: {str(e)}")

