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
        
        # Extract and format data more intelligently
        data_summary = []
        context_info = []
        
        # Handle different data structures
        if isinstance(county_data, dict):
            # Check for hierarchical data structure
            if 'target_data' in county_data:
                # This is hierarchical analysis data
                target_data = county_data.get('target_data', {})
                geographic_context = county_data.get('geographic_context', {})
                regional_trends = county_data.get('regional_trends', {})
                balanced_analysis = county_data.get('balanced_analysis', {})
                
                data_summary.append(f"=== PRIMARY DATA FOR {county.upper()} ===")
                for metric, value in target_data.items():
                    if isinstance(value, dict):
                        data_summary.append(f"- {metric}: {value.get('total', value.get('value', 'N/A'))} (trend: {value.get('trend', 'unknown')})")
                    else:
                        data_summary.append(f"- {metric}: {value}")
                
                if geographic_context:
                    data_summary.append(f"\n=== GEOGRAPHIC CONTEXT ===")
                    for key, value in geographic_context.items():
                        data_summary.append(f"- {key}: {value}")
                
                if regional_trends:
                    data_summary.append(f"\n=== REGIONAL TRENDS ===")
                    for key, value in regional_trends.items():
                        data_summary.append(f"- {key}: {value}")
                
                if balanced_analysis:
                    data_summary.append(f"\n=== BALANCED ANALYSIS ===")
                    for key, value in balanced_analysis.items():
                        data_summary.append(f"- {key}: {value}")
                        
            else:
                # Simple key-value data
                for metric, value in county_data.items():
                    data_summary.append(f"- {metric}: {value}")
        
        data_text = "\n".join(data_summary)
        print(f"📝 Data summary being sent to Claude:")
        print(f"{data_text[:500]}...")
        
        prompt = f"""You are an expert urban planning analyst specializing in community development and policy recommendations. Based on this comprehensive data analysis for {county}:

{data_text}

Generate 2-4 most pressing community problems that require immediate policy intervention. For each problem, provide:

1. **Problem Analysis**: Clear identification of the issue with specific data points
2. **Root Cause Analysis**: Why this problem exists in this community
3. **Policy Solution**: Evidence-based policy recommendation with implementation steps
4. **Expected Outcomes**: Quantifiable impact metrics

CRITICAL: Return ONLY a valid JSON array. No markdown, no explanations, no additional text. Start with [ and end with ].

Return JSON array with this EXACT structure:
[
  {{
    "title": "Concise Problem Title (5-8 words)",
    "description": "Detailed 2-3 sentence explanation of the problem, including specific data points and why it's concerning for this community. Reference actual numbers from the data.",
    "severity": "high|medium|low",
    "solution": {{
      "title": "Specific Policy Solution Name",
      "description": "Detailed implementation plan with specific steps, partnerships needed, and timeline. Reference successful implementations in similar communities.",
      "estimated_cost": "Realistic cost estimate with funding sources (e.g., '$2.5M annually from state grants and local budget')",
      "expected_impact": "Specific, measurable outcomes with timeline (e.g., '25% reduction in youth arrests within 18 months')"
    }},
    "metrics": {{
      "current_value": "Current metric value from data",
      "target_value": "Realistic target after intervention",
      "comparison": "How this compares to regional/state averages",
      "trend": "Current trend direction"
    }}
  }}
]

**Requirements:**
- Use actual data points from the analysis
- Focus on problems that can be addressed through local government action
- Solutions must be implementable by city/county government
- Include specific metrics and comparisons
- Prioritize problems with the highest community impact
- Consider equity and social justice implications

**Severity Guidelines:**
- HIGH: Immediate threat to public safety, health, or economic stability
- MEDIUM: Significant quality of life impact or economic concern
- LOW: Important but not urgent community improvement opportunity

Generate actionable, data-driven policy recommendations that a local government could implement within 6-18 months."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,  # Increased token limit
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            print(f"Claude problems response: {content[:500]}...")  # Debug: print first 500 chars
            
            if not content.strip():
                raise Exception("Empty response from Claude")
            
            # Extract JSON from markdown code blocks if present
            content = extract_json_from_markdown(content)
            
            # Try to fix common JSON issues
            content = self._fix_json_response(content)
            
            problems = json.loads(content)
            print(f"✅ Successfully parsed {len(problems)} problems from Claude")
            return problems
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON Decode Error: {str(e)}")
            print(f"❌ Full response content: {content}")
            raise Exception(f"Claude problem synthesis failed: Invalid JSON response. Error: {str(e)}. Content: {content[:500]}...")
        except Exception as e:
            print(f"❌ General Error: {str(e)}")
            # Only print content if it exists (not for connection errors)
            if 'content' in locals():
                print(f"❌ Full response content: {content}")
            else:
                print(f"❌ No response content available (connection error)")
            raise Exception(f"Claude problem synthesis failed: {str(e)}")
    
    def _fix_json_response(self, content: str) -> str:
        """Fix common JSON issues in Claude responses"""
        content = content.strip()
        
        # Remove any text after the JSON array/object
        # Find the last complete JSON structure
        if content.startswith('['):
            # Find the matching closing bracket
            bracket_count = 0
            last_valid_pos = -1
            for i, char in enumerate(content):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        last_valid_pos = i
                        break
            
            if last_valid_pos > 0:
                content = content[:last_valid_pos + 1]
        
        # Remove trailing commas before closing brackets
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        # Ensure it's a valid JSON array
        if not content.startswith('['):
            # If it's a single object, wrap it in an array
            if content.startswith('{'):
                content = '[' + content + ']'
        
        return content
    
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
        
        # Extract and format data more intelligently
        data_summary = []
        
        # Handle different data structures
        if isinstance(county_data, dict):
            # Check for hierarchical data structure
            if 'target_data' in county_data:
                # This is hierarchical analysis data
                target_data = county_data.get('target_data', {})
                geographic_context = county_data.get('geographic_context', {})
                regional_trends = county_data.get('regional_trends', {})
                balanced_analysis = county_data.get('balanced_analysis', {})
                
                data_summary.append(f"=== PRIMARY DATA FOR {county.upper()} ===")
                for metric, value in target_data.items():
                    if isinstance(value, dict):
                        data_summary.append(f"- {metric}: {value.get('total', value.get('value', 'N/A'))} (trend: {value.get('trend', 'unknown')})")
                    else:
                        data_summary.append(f"- {metric}: {value}")
                
                if geographic_context:
                    data_summary.append(f"\n=== GEOGRAPHIC CONTEXT ===")
                    for key, value in geographic_context.items():
                        data_summary.append(f"- {key}: {value}")
                
                if regional_trends:
                    data_summary.append(f"\n=== REGIONAL TRENDS ===")
                    for key, value in regional_trends.items():
                        data_summary.append(f"- {key}: {value}")
                
                if balanced_analysis:
                    data_summary.append(f"\n=== BALANCED ANALYSIS ===")
                    for key, value in balanced_analysis.items():
                        data_summary.append(f"- {key}: {value}")
                        
            else:
                # Simple key-value data
                for metric, value in county_data.items():
                    data_summary.append(f"- {metric}: {value}")
        
        data_text = "\n".join(data_summary)
        
        prompt = f"""You are an expert urban planning analyst specializing in community development and policy recommendations. Based on this comprehensive data analysis for {county}:

{data_text}

Generate 2-4 most pressing community problems that require immediate policy intervention. For each problem, provide:

1. **Problem Analysis**: Clear identification of the issue with specific data points
2. **Root Cause Analysis**: Why this problem exists in this community
3. **Policy Solution**: Evidence-based policy recommendation with implementation steps
4. **Expected Outcomes**: Quantifiable impact metrics

CRITICAL: Return ONLY a valid JSON array. No markdown, no explanations, no additional text. Start with [ and end with ].

Return JSON array with this EXACT structure:
[
  {{
    "title": "Concise Problem Title (5-8 words)",
    "description": "Detailed 2-3 sentence explanation of the problem, including specific data points and why it's concerning for this community. Reference actual numbers from the data.",
    "severity": "high|medium|low",
    "solution": {{
      "title": "Specific Policy Solution Name",
      "description": "Detailed implementation plan with specific steps, partnerships needed, and timeline. Reference successful implementations in similar communities.",
      "estimated_cost": "Realistic cost estimate with funding sources (e.g., '$2.5M annually from state grants and local budget')",
      "expected_impact": "Specific, measurable outcomes with timeline (e.g., '25% reduction in youth arrests within 18 months')"
    }},
    "metrics": {{
      "current_value": "Current metric value from data",
      "target_value": "Realistic target after intervention",
      "comparison": "How this compares to regional/state averages",
      "trend": "Current trend direction"
    }}
  }}
]

**Requirements:**
- Use actual data points from the analysis
- Focus on problems that can be addressed through local government action
- Solutions must be implementable by city/county government
- Include specific metrics and comparisons
- Prioritize problems with the highest community impact
- Consider equity and social justice implications

**Severity Guidelines:**
- HIGH: Immediate threat to public safety, health, or economic stability
- MEDIUM: Significant quality of life impact or economic concern
- LOW: Important but not urgent community improvement opportunity

Generate actionable, data-driven policy recommendations that a local government could implement within 6-18 months."""

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
