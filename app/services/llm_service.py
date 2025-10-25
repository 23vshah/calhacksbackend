import json
import os
import re
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import anthropic
import openai
from app.models import NormalizationProfile, MetricWeight

def extract_json_from_markdown(content: str) -> str:
    """Extract JSON from markdown code blocks"""
    content = content.strip()
    
    # Remove markdown code blocks
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
    elif content.startswith("```"):
        content = content[3:]  # Remove ```
    
    if content.endswith("```"):
        content = content[:-3]  # Remove ```
    
    content = content.strip()
    
    # Remove any leading text before JSON
    # Look for the first { or [ that starts a JSON structure
    json_start = -1
    for i, char in enumerate(content):
        if char in '{[':
            json_start = i
            break
    
    if json_start >= 0:
        content = content[json_start:]
    
    # Try to find complete JSON structure
    json_match = re.search(r'(\[.*\]|\{.*\})', content, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    return content

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def analyze_schema(self, sample_data: str) -> NormalizationProfile:
        pass
    
    @abstractmethod
    async def calculate_weights(self, metrics: List[str]) -> List[MetricWeight]:
        pass
    
    @abstractmethod
    async def synthesize_problems(self, county_data: Dict[str, Any], county: str) -> List[Dict[str, Any]]:
        pass

class ClaudeProvider(LLMProvider):
    """Claude API provider"""
    
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
    
    def analyze_schema(self, sample_data: str) -> NormalizationProfile:
        """Analyze dataset sample and return normalization profile"""
        
        prompt = f"""You are a data analyst. Analyze this dataset sample:
{sample_data}

Respond in JSON format:
{{
  "data_type": "crime" | "demographics" | "economic" | "housing" | "311" | "other",
  "geographic_level": "state" | "region" | "county" | "city" | "neighborhood",
  "geographic_hierarchy": {{
    "neighborhood": "specific neighborhood name or null",
    "city": "city name or null", 
    "county": "county name or null",
    "state": "state name or null",
    "region": "region name or null"
  }},
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

IMPORTANT: 
1. For geographic_level, choose EXACTLY ONE from: "state", "region", "county", "city", "neighborhood"
2. For geographic_hierarchy, fill in the COMPLETE hierarchy based on the data:
   - If data is neighborhood-level (e.g., "Inner Sunset"), fill: neighborhood="Inner Sunset", city="San Francisco", county="San Francisco County", state="California"
   - If data is city-level (e.g., "Oakland"), fill: city="Oakland", county="Alameda County", state="California" 
   - If data is county-level (e.g., "Alameda County"), fill: county="Alameda County", state="California"
   - If data is state-level, fill: state="California"
   - Use null for levels not applicable

Geographic level means what each row's data represents. Is it county-wide data or city etc.
This is important because it affects how we aggregate the data.
- "state": Data covers entire states (e.g., California, Texas)
- "region": Data covers regions inside the state (e.g., West Coast, Northeast within California, Sacremento Valley)
- "county": Data covers counties (e.g., Alameda County, Los Angeles County)
- "city": Data covers cities (e.g., Oakland, San Francisco)
- "neighborhood": Data covers neighborhoods/districts (e.g., Downtown, Mission District)

Be specific about what each column measures and provide clear normalized names."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            print(f"Claude response: {content[:500]}...")  # Debug: print first 500 chars
            
            if not content.strip():
                raise Exception("Empty response from Claude")
            
            # Extract JSON from markdown code blocks if present
            content = extract_json_from_markdown(content)
            
            profile_data = json.loads(content)
            return NormalizationProfile(**profile_data)
            
        except json.JSONDecodeError as e:
            raise Exception(f"Claude schema analysis failed: Invalid JSON response. Content: {content[:200]}...")
        except Exception as e:
            raise Exception(f"Claude schema analysis failed: {str(e)}")
    
    def calculate_weights(self, metrics: List[str]) -> List[MetricWeight]:
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
            print(f"Claude weights response: {content[:500]}...")  # Debug: print first 500 chars
            
            if not content.strip():
                raise Exception("Empty response from Claude")
            
            # Extract JSON from markdown code blocks if present
            content = extract_json_from_markdown(content)
            
            weights_data = json.loads(content)
            return [MetricWeight(**weight) for weight in weights_data]
            
        except json.JSONDecodeError as e:
            raise Exception(f"Claude weight calculation failed: Invalid JSON response. Content: {content[:200]}...")
        except Exception as e:
            raise Exception(f"Claude weight calculation failed: {str(e)}")
    
    def synthesize_problems(self, county_data: Dict[str, Any], county: str) -> List[Dict[str, Any]]:
        """Synthesize problems and solutions from county data"""
        
        print(f"🔍 Claude Provider - Processing data for {county}")
        print(f"Raw county_data: {county_data}")
        
        data_summary = []
        for metric, value in county_data.items():
            data_summary.append(f"- {metric}: {value}")
        
        data_text = "\n".join(data_summary)
        print(f"📝 Data summary being sent to Claude:")
        print(f"{data_text[:500]}...")
        
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
            print(f"Claude problems response: {content[:500]}...")  # Debug: print first 500 chars
            
            if not content.strip():
                raise Exception("Empty response from Claude")
            
            # Extract JSON from markdown code blocks if present
            content = extract_json_from_markdown(content)
            
            problems = json.loads(content)
            return problems
            
        except json.JSONDecodeError as e:
            raise Exception(f"Claude problem synthesis failed: Invalid JSON response. Content: {content[:200]}...")
        except Exception as e:
            raise Exception(f"Claude problem synthesis failed: {str(e)}")
    
    def _generate_strategy_sync(self, prompt: str) -> Dict[str, Any]:
        """Generate aggregation strategy (Claude sync version)"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            content = extract_json_from_markdown(content)
            return json.loads(content)
            
        except Exception as e:
            raise Exception(f"Claude strategy generation failed: {str(e)}")
    
    def _analyze_data_sync(self, prompt: str) -> Dict[str, Any]:
        """Analyze aggregated data (Claude sync version)"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            print(f"Claude analysis response: {content[:500]}...")  # Debug
            
            content = extract_json_from_markdown(content)
            
            if not content.strip():
                raise Exception("Empty response from Claude")
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            raise Exception(f"Claude data analysis failed: Invalid JSON response. Content: {content[:200]}...")
        except Exception as e:
            raise Exception(f"Claude data analysis failed: {str(e)}")

class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self):
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    async def analyze_schema(self, sample_data: str) -> NormalizationProfile:
        """Analyze dataset sample and return normalization profile"""
        
        prompt = f"""You are a data analyst. Analyze this dataset sample:
{sample_data}

Respond in JSON format:
{{
  "data_type": "crime" | "demographics" | "economic" | "housing" | "311" | "other",
  "geographic_level": "state" | "region" | "county" | "city" | "neighborhood",
  "geographic_hierarchy": {{
    "neighborhood": "specific neighborhood name or null",
    "city": "city name or null", 
    "county": "county name or null",
    "state": "state name or null",
    "region": "region name or null"
  }},
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

IMPORTANT: 
1. For geographic_level, choose EXACTLY ONE from: "state", "region", "county", "city", "neighborhood"
2. For geographic_hierarchy, fill in the COMPLETE hierarchy based on the data:
   - If data is neighborhood-level (e.g., "Inner Sunset"), fill: neighborhood="Inner Sunset", city="San Francisco", county="San Francisco County", state="California"
   - If data is city-level (e.g., "Oakland"), fill: city="Oakland", county="Alameda County", state="California" 
   - If data is county-level (e.g., "Alameda County"), fill: county="Alameda County", state="California"
   - If data is state-level, fill: state="California"
   - Use null for levels not applicable

- "state": Data covers entire states (e.g., California, Texas)
- "region": Data covers multi-state regions (e.g., West Coast, Northeast)
- "county": Data covers counties (e.g., Alameda County, Los Angeles County)
- "city": Data covers cities (e.g., Oakland, San Francisco)
- "neighborhood": Data covers neighborhoods/districts (e.g., Downtown, Mission District)

Be specific about what each column measures and provide clear normalized names."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            profile_data = json.loads(content)
            return NormalizationProfile(**profile_data)
            
        except Exception as e:
            raise Exception(f"OpenAI schema analysis failed: {str(e)}")
    
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
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            weights_data = json.loads(content)
            return [MetricWeight(**weight) for weight in weights_data]
            
        except Exception as e:
            raise Exception(f"OpenAI weight calculation failed: {str(e)}")
    
    async def synthesize_problems(self, county_data: Dict[str, Any], county: str) -> List[Dict[str, Any]]:
        """Synthesize problems and solutions from county data"""
        
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
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            problems = json.loads(content)
            return problems
            
        except Exception as e:
            raise Exception(f"OpenAI problem synthesis failed: {str(e)}")
    
    async def _generate_strategy_async(self, prompt: str) -> Dict[str, Any]:
        """Generate aggregation strategy (OpenAI async version)"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            content = extract_json_from_markdown(content)
            return json.loads(content)
            
        except Exception as e:
            raise Exception(f"OpenAI strategy generation failed: {str(e)}")
    
    async def _analyze_data_async(self, prompt: str) -> Dict[str, Any]:
        """Analyze aggregated data (OpenAI async version)"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            content = extract_json_from_markdown(content)
            return json.loads(content)
            
        except Exception as e:
            raise Exception(f"OpenAI data analysis failed: {str(e)}")

class LLMService:
    """Main LLM service that can switch between providers"""
    
    def __init__(self, provider: str = None):
        """
        Initialize LLM service with specified provider
        
        Args:
            provider: "claude", "openai", or None (use LLM_PROVIDER env var or auto-detect)
        """
        if provider is None:
            # Check LLM_PROVIDER environment variable first
            provider = os.getenv("LLM_PROVIDER")
            
            # If not set, auto-detect based on available API keys
            if provider is None:
                if os.getenv("ANTHROPIC_API_KEY"):
                    provider = "claude"
                elif os.getenv("OPENAI_API_KEY"):
                    provider = "openai"
                else:
                    raise Exception("No API keys found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        
        self.provider_name = provider
        
        if provider == "claude":
            self.provider = ClaudeProvider()
        elif provider == "openai":
            self.provider = OpenAIProvider()
        else:
            raise Exception(f"Unknown provider: {provider}")
    
    async def analyze_schema(self, sample_data: str) -> NormalizationProfile:
        """Analyze dataset sample and return normalization profile"""
        if self.provider_name == "claude":
            return self.provider.analyze_schema(sample_data)
        else:
            return await self.provider.analyze_schema(sample_data)
    
    async def calculate_weights(self, metrics: List[str]) -> List[MetricWeight]:
        """Calculate importance weights for metrics"""
        if self.provider_name == "claude":
            return self.provider.calculate_weights(metrics)
        else:
            return await self.provider.calculate_weights(metrics)
    
    async def synthesize_problems(self, county_data: Dict[str, Any], county: str) -> List[Dict[str, Any]]:
        """Synthesize problems and solutions from county data"""
        
        # Debug: Print what's being sent to LLM
        print(f"🤖 LLM Input for {county}:")
        print(f"County data type: {type(county_data)}")
        print(f"County data keys: {list(county_data.keys()) if isinstance(county_data, dict) else 'Not a dict'}")
        print(f"County data sample: {str(county_data)[:300]}...")
        
        if self.provider_name == "claude":
            return self.provider.synthesize_problems(county_data, county)
        else:
            return await self.provider.synthesize_problems(county_data, county)
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the current provider"""
        return {
            "provider": self.provider_name,
            "model": getattr(self.provider, 'model', 'claude-3-haiku-20240307')
        }
    
    async def _generate_aggregation_strategy(self, prompt: str) -> Dict[str, Any]:
        """Generate aggregation strategy using LLM"""
        if self.provider_name == "claude":
            return self.provider._generate_strategy_sync(prompt)
        else:
            return await self.provider._generate_strategy_async(prompt)
    
    async def _analyze_aggregated_data(self, prompt: str) -> Dict[str, Any]:
        """Analyze aggregated data and request drill-downs"""
        if self.provider_name == "claude":
            return self.provider._analyze_data_sync(prompt)
        else:
            return await self.provider._analyze_data_async(prompt)
