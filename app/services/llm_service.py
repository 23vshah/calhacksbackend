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
    
    def synthesize_problems(self, county_data: Dict[str, Any], county: str) -> Dict[str, Any]:
        """Synthesize comprehensive city report with separate calls for city info and problems"""
        
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
        
        # FIRST CALL: Generate comprehensive city information
        city_info_prompt = f"""You are an expert urban planning analyst. Based on this data analysis for {county}:

{data_text}

Generate comprehensive city information including demographics, economic indicators, and key metrics. Return ONLY valid JSON with this EXACT structure:

{{
  "name": "{county}",
  "county": "{county} County",
  "riskLevel": "high|medium|low",
  "metrics": {{
    "population": 1234567,
    "medianIncome": 75000,
    "crimeRate": 0.045,
    "foreclosureRate": 0.012,
    "vacancyRate": 0.08,
    "unemploymentRate": 0.055,
    "homeValue": 850000,
    "rentBurden": 0.35,
    "educationLevel": 0.78,
    "povertyRate": 0.15,
    "airQuality": 65,
    "treeCanopy": 0.25,
    "transitAccess": 0.68,
    "walkability": 72,
    "bikeability": 58
  }},
  "demographics": {{
    "ageDistribution": {{
      "under18": 0.22,
      "18to34": 0.28,
      "35to54": 0.25,
      "55to64": 0.12,
      "over65": 0.13
    }},
    "raceEthnicity": {{
      "white": 0.45,
      "hispanic": 0.28,
      "black": 0.12,
      "asian": 0.10,
      "other": 0.05
    }},
    "householdComposition": {{
      "familyHouseholds": 0.65,
      "nonFamilyHouseholds": 0.35,
      "singleParent": 0.15
    }}
  }},
  "economicIndicators": {{
    "gdp": 125000000000,
    "medianRent": 2800,
    "medianHomePrice": 850000,
    "jobGrowth": 0.025,
    "businessCount": 45000,
    "startupDensity": 0.12,
    "ventureCapital": 2500000000
  }},
  "infrastructureMetrics": {{
    "roadCondition": 0.72,
    "bridgeCondition": 0.68,
    "waterSystem": 0.85,
    "sewerSystem": 0.78,
    "publicTransit": 0.65,
    "broadbandAccess": 0.92,
    "renewableEnergy": 0.35
  }},
  "socialIndicators": {{
    "lifeExpectancy": 78.5,
    "infantMortality": 4.2,
    "highSchoolGraduation": 0.88,
    "collegeGraduation": 0.45,
    "foodSecurity": 0.85,
    "homelessness": 0.008,
    "mentalHealthAccess": 0.72
  }}
}}

Use realistic values based on typical urban areas. Focus on metrics that would be relevant for policy analysis."""
        
        # SECOND CALL: Generate detailed problems and solutions
        problems_prompt = f"""You are an expert urban planning analyst specializing in community development and policy recommendations. Based on this comprehensive data analysis for {county}:

{data_text}

Generate 2-3 most pressing community problems that require immediate policy intervention. For each problem, provide comprehensive details including implementation phases, cost breakdown, success metrics, and stakeholder information.

CRITICAL: Return ONLY a valid JSON array. No markdown, no explanations, no additional text. Start with [ and end with ].

Return JSON array with this EXACT structure:
[
  {{
    "title": "Concise Problem Title (5-8 words)",
    "description": "Detailed 2-3 sentence explanation of the problem, including specific data points and why it's concerning for this community. Reference actual numbers from the data.",
    "severity": "high|medium|low",
    "category": "housing|economic|safety|environment|infrastructure|social",
    "solution": {{
      "title": "Specific Policy Solution Name",
      "description": "Detailed implementation plan with specific steps, partnerships needed, and timeline. Reference successful implementations in similar communities.",
      "estimated_cost": "Realistic cost estimate with funding sources (e.g., '$2.5M annually from state grants and local budget')",
      "expected_impact": "Specific, measurable outcomes with timeline (e.g., '25% reduction in youth arrests within 18 months')",
      "timeline": "6-18 months",
      "impact": "Specific measurable outcomes",
      "steps": [
        "Step 1: Establish community advisory board",
        "Step 2: Partner with local non-profits",
        "Step 3: Develop mentorship matching system",
        "Step 4: Create job training programs",
        "Step 5: Implement early intervention protocols"
      ],
      "costBreakdown": [
        {{"item": "Staff salaries", "amount": "$1.2M"}},
        {{"item": "Program materials", "amount": "$300K"}},
        {{"item": "Community outreach", "amount": "$200K"}},
        {{"item": "Evaluation and monitoring", "amount": "$150K"}}
      ],
      "implementationPhases": [
        {{
          "phase": "Planning and Setup",
          "duration": "2-3 months",
          "milestones": [
            "Establish governance structure",
            "Secure funding commitments",
            "Hire key staff"
          ]
        }},
        {{
          "phase": "Pilot Implementation",
          "duration": "3-6 months",
          "milestones": [
            "Launch pilot program",
            "Begin community outreach",
            "Establish partnerships"
          ]
        }}
      ],
      "successMetrics": [
        "25% reduction in target metric within 18 months",
        "80% participant satisfaction rate",
        "90% program completion rate",
        "15% improvement in community perception"
      ],
      "similarCities": [
        {{"city": "Portland, OR", "outcome": "Reduced youth crime by 30% in 2 years"}},
        {{"city": "Austin, TX", "outcome": "Improved community engagement by 40%"}}
      ],
      "requiredDepartments": [
        "City Planning",
        "Community Development",
        "Police Department",
        "Social Services"
      ],
      "stakeholders": [
        "Local government officials",
        "Community organizations",
        "Residents and families",
        "Local businesses",
        "Educational institutions"
      ],
      "fundingSources": [
        "State community development grants",
        "Federal HUD funding",
        "Local budget allocation",
        "Private foundation grants"
      ],
      "risks": [
        "Community resistance to new programs",
        "Funding shortfalls",
        "Staff turnover",
        "Regulatory compliance issues"
      ]
    }},
    "metrics": {{
      "current_value": "Current metric value from data (use percentages for rates like unemployment, crime rates per 1000, rent burden; use raw numbers for counts like violations)",
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
- Provide comprehensive implementation details
- IMPORTANT: For metric values, use realistic percentages (0-100%) for rates and reasonable numbers for counts

**Severity Guidelines:**
- HIGH: Immediate threat to public safety, health, or economic stability
- MEDIUM: Significant quality of life impact or economic concern
- LOW: Important but not urgent community improvement opportunity

Generate actionable, data-driven policy recommendations that a local government could implement within 6-18 months."""

        try:
            # FIRST CALL: Get city information
            print("🏙️ Making first LLM call for city information...")
            city_response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                messages=[{"role": "user", "content": city_info_prompt}]
            )
            
            city_content = city_response.content[0].text
            print(f"Claude city info response: {city_content[:500]}...")
            
            if not city_content.strip():
                raise Exception("Empty city info response from Claude")
            
            # Extract JSON from markdown code blocks if present
            city_content = extract_json_from_markdown(city_content)
            city_info = json.loads(city_content)
            print(f"✅ Successfully parsed city info from Claude")
            
            # SECOND CALL: Get problems and solutions
            print("🔧 Making second LLM call for problems and solutions...")
            problems_response = self.client.messages.create(
                model=self.model,
                max_tokens=20000,  # Increased token limit
                messages=[{"role": "user", "content": problems_prompt}]
            )
            
            problems_content = problems_response.content[0].text
            print(f"Claude problems response: {problems_content[:500]}...")
            
            if not problems_content.strip():
                raise Exception("Empty problems response from Claude")
            
            # Extract JSON from markdown code blocks if present
            problems_content = extract_json_from_markdown(problems_content)
            
            # Try to fix common JSON issues
            problems_content = self._fix_json_response(problems_content)
            
            # Try to parse JSON with multiple strategies
            problems = None
            parse_attempts = [
                problems_content,
                problems_content + ']',  # Try adding closing bracket
                problems_content[:-1] + ']' if problems_content.endswith(',') else problems_content + ']',  # Remove trailing comma and add bracket
            ]
            
            for i, attempt_content in enumerate(parse_attempts):
                try:
                    problems = json.loads(attempt_content)
                    print(f"✅ Successfully parsed problems on attempt {i+1}")
                    break
                except json.JSONDecodeError as e:
                    print(f"❌ Parse attempt {i+1} failed: {str(e)}")
                    if i == len(parse_attempts) - 1:  # Last attempt
                        raise e
                    continue
            
            if problems is None:
                raise Exception("All JSON parsing attempts failed")
            
            print(f"✅ Successfully parsed {len(problems)} problems from Claude")
            
            # Combine both responses
            combined_response = {
                "cityInfo": city_info,
                "problems": problems
            }
            
            return combined_response
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON Decode Error: {str(e)}")
            if 'city_content' in locals():
                print(f"❌ City content: {city_content[:500]}...")
            if 'problems_content' in locals():
                print(f"❌ Problems content: {problems_content[:500]}...")
            raise Exception(f"Claude report synthesis failed: Invalid JSON response. Error: {str(e)}")
        except Exception as e:
            print(f"❌ General Error: {str(e)}")
            raise Exception(f"Claude report synthesis failed: {str(e)}")
    
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
        
        # Fix common JSON issues
        # Remove any incomplete objects at the end
        if content.count('{') > content.count('}'):
            # Find the last complete object
            brace_count = 0
            last_complete_pos = -1
            for i, char in enumerate(content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_complete_pos = i
                        break
            
            if last_complete_pos > 0:
                # Find the end of this object in the array
                content = content[:last_complete_pos + 1]
                # Add closing bracket if needed
                if not content.endswith(']'):
                    content += ']'
        
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
    
    async def synthesize_problems(self, county_data: Dict[str, Any], county: str) -> Dict[str, Any]:
        """Synthesize comprehensive city report with separate calls for city info and problems"""
        
        print(f"🔍 OpenAI Provider - Processing data for {county}")
        print(f"Raw county_data: {county_data}")
        
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
        print(f"📝 Data summary being sent to OpenAI:")
        print(f"{data_text[:500]}...")
        
        # FIRST CALL: Generate comprehensive city information
        city_info_prompt = f"""You are an expert urban planning analyst. Based on this data analysis for {county}:

{data_text}

Generate comprehensive city information including demographics, economic indicators, and key metrics. Return ONLY valid JSON with this EXACT structure:

{{
  "name": "{county}",
  "county": "{county} County",
  "riskLevel": "high|medium|low",
  "metrics": {{
    "population": 1234567,
    "medianIncome": 75000,
    "crimeRate": 0.045,
    "foreclosureRate": 0.012,
    "vacancyRate": 0.08,
    "unemploymentRate": 0.055,
    "homeValue": 850000,
    "rentBurden": 0.35,
    "educationLevel": 0.78,
    "povertyRate": 0.15,
    "airQuality": 65,
    "treeCanopy": 0.25,
    "transitAccess": 0.68,
    "walkability": 72,
    "bikeability": 58
  }},
  "demographics": {{
    "ageDistribution": {{
      "under18": 0.22,
      "18to34": 0.28,
      "35to54": 0.25,
      "55to64": 0.12,
      "over65": 0.13
    }},
    "raceEthnicity": {{
      "white": 0.45,
      "hispanic": 0.28,
      "black": 0.12,
      "asian": 0.10,
      "other": 0.05
    }},
    "householdComposition": {{
      "familyHouseholds": 0.65,
      "nonFamilyHouseholds": 0.35,
      "singleParent": 0.15
    }}
  }},
  "economicIndicators": {{
    "gdp": 125000000000,
    "medianRent": 2800,
    "medianHomePrice": 850000,
    "jobGrowth": 0.025,
    "businessCount": 45000,
    "startupDensity": 0.12,
    "ventureCapital": 2500000000
  }},
  "infrastructureMetrics": {{
    "roadCondition": 0.72,
    "bridgeCondition": 0.68,
    "waterSystem": 0.85,
    "sewerSystem": 0.78,
    "publicTransit": 0.65,
    "broadbandAccess": 0.92,
    "renewableEnergy": 0.35
  }},
  "socialIndicators": {{
    "lifeExpectancy": 78.5,
    "infantMortality": 4.2,
    "highSchoolGraduation": 0.88,
    "collegeGraduation": 0.45,
    "foodSecurity": 0.85,
    "homelessness": 0.008,
    "mentalHealthAccess": 0.72
  }}
}}

Use realistic values based on typical urban areas. Focus on metrics that would be relevant for policy analysis."""
        
        # SECOND CALL: Generate detailed problems and solutions
        problems_prompt = f"""You are an expert urban planning analyst specializing in community development and policy recommendations. Based on this comprehensive data analysis for {county}:

{data_text}

Generate 3-5 most pressing community problems that require immediate policy intervention. For each problem, provide comprehensive details including implementation phases, cost breakdown, success metrics, and stakeholder information.

CRITICAL: Return ONLY a valid JSON array. No markdown, no explanations, no additional text. Start with [ and end with ].

Return JSON array with this EXACT structure:
[
  {{
    "title": "Concise Problem Title (5-8 words)",
    "description": "Detailed 2-3 sentence explanation of the problem, including specific data points and why it's concerning for this community. Reference actual numbers from the data.",
    "severity": "high|medium|low",
    "category": "housing|economic|safety|environment|infrastructure|social",
    "solution": {{
      "title": "Specific Policy Solution Name",
      "description": "Detailed implementation plan with specific steps, partnerships needed, and timeline. Reference successful implementations in similar communities.",
      "estimated_cost": "Realistic cost estimate with funding sources (e.g., '$2.5M annually from state grants and local budget')",
      "expected_impact": "Specific, measurable outcomes with timeline (e.g., '25% reduction in youth arrests within 18 months')",
      "timeline": "6-18 months",
      "impact": "Specific measurable outcomes",
      "steps": [
        "Step 1: Establish community advisory board",
        "Step 2: Partner with local non-profits",
        "Step 3: Develop mentorship matching system",
        "Step 4: Create job training programs",
        "Step 5: Implement early intervention protocols"
      ],
      "costBreakdown": [
        {{"item": "Staff salaries", "amount": "$1.2M"}},
        {{"item": "Program materials", "amount": "$300K"}},
        {{"item": "Community outreach", "amount": "$200K"}},
        {{"item": "Evaluation and monitoring", "amount": "$150K"}}
      ],
      "implementationPhases": [
        {{
          "phase": "Planning and Setup",
          "duration": "2-3 months",
          "milestones": [
            "Establish governance structure",
            "Secure funding commitments",
            "Hire key staff"
          ]
        }},
        {{
          "phase": "Pilot Implementation",
          "duration": "3-6 months",
          "milestones": [
            "Launch pilot program",
            "Begin community outreach",
            "Establish partnerships"
          ]
        }}
      ],
      "successMetrics": [
        "25% reduction in target metric within 18 months",
        "80% participant satisfaction rate",
        "90% program completion rate",
        "15% improvement in community perception"
      ],
      "similarCities": [
        {{"city": "Portland, OR", "outcome": "Reduced youth crime by 30% in 2 years"}},
        {{"city": "Austin, TX", "outcome": "Improved community engagement by 40%"}}
      ],
      "requiredDepartments": [
        "City Planning",
        "Community Development",
        "Police Department",
        "Social Services"
      ],
      "stakeholders": [
        "Local government officials",
        "Community organizations",
        "Residents and families",
        "Local businesses",
        "Educational institutions"
      ],
      "fundingSources": [
        "State community development grants",
        "Federal HUD funding",
        "Local budget allocation",
        "Private foundation grants"
      ],
      "risks": [
        "Community resistance to new programs",
        "Funding shortfalls",
        "Staff turnover",
        "Regulatory compliance issues"
      ]
    }},
    "metrics": {{
      "current_value": "Current metric value from data (use percentages for rates like unemployment, crime rates per 1000, rent burden; use raw numbers for counts like violations)",
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
- Provide comprehensive implementation details
- IMPORTANT: For metric values, use realistic percentages (0-100%) for rates and reasonable numbers for counts

**Severity Guidelines:**
- HIGH: Immediate threat to public safety, health, or economic stability
- MEDIUM: Significant quality of life impact or economic concern
- LOW: Important but not urgent community improvement opportunity

Generate actionable, data-driven policy recommendations that a local government could implement within 6-18 months."""

        try:
            # FIRST CALL: Get city information
            print("🏙️ Making first OpenAI call for city information...")
            city_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": city_info_prompt}],
                max_tokens=4000,
                temperature=0.1
            )
            
            city_content = city_response.choices[0].message.content
            print(f"OpenAI city info response: {city_content[:500]}...")
            
            if not city_content.strip():
                raise Exception("Empty city info response from OpenAI")
            
            # Extract JSON from markdown code blocks if present
            city_content = extract_json_from_markdown(city_content)
            city_info = json.loads(city_content)
            print(f"✅ Successfully parsed city info from OpenAI")
            
            # SECOND CALL: Get problems and solutions
            print("🔧 Making second OpenAI call for problems and solutions...")
            problems_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": problems_prompt}],
                max_tokens=12000,  # Increased token limit
                temperature=0.1
            )
            
            problems_content = problems_response.choices[0].message.content
            print(f"OpenAI problems response: {problems_content[:500]}...")
            
            if not problems_content.strip():
                raise Exception("Empty problems response from OpenAI")
            
            # Extract JSON from markdown code blocks if present
            problems_content = extract_json_from_markdown(problems_content)
            
            # Try to fix common JSON issues
            problems_content = self._fix_json_response(problems_content)
            
            # Try to parse JSON with multiple strategies
            problems = None
            parse_attempts = [
                problems_content,
                problems_content + ']',  # Try adding closing bracket
                problems_content[:-1] + ']' if problems_content.endswith(',') else problems_content + ']',  # Remove trailing comma and add bracket
            ]
            
            for i, attempt_content in enumerate(parse_attempts):
                try:
                    problems = json.loads(attempt_content)
                    print(f"✅ Successfully parsed problems on attempt {i+1}")
                    break
                except json.JSONDecodeError as e:
                    print(f"❌ Parse attempt {i+1} failed: {str(e)}")
                    if i == len(parse_attempts) - 1:  # Last attempt
                        raise e
                    continue
            
            if problems is None:
                raise Exception("All JSON parsing attempts failed")
            
            print(f"✅ Successfully parsed {len(problems)} problems from OpenAI")
            
            # Combine both responses
            combined_response = {
                "cityInfo": city_info,
                "problems": problems
            }
            
            return combined_response
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON Decode Error: {str(e)}")
            if 'city_content' in locals():
                print(f"❌ City content: {city_content[:500]}...")
            if 'problems_content' in locals():
                print(f"❌ Problems content: {problems_content[:500]}...")
            raise Exception(f"OpenAI report synthesis failed: Invalid JSON response. Error: {str(e)}")
        except Exception as e:
            print(f"❌ General Error: {str(e)}")
            raise Exception(f"OpenAI report synthesis failed: {str(e)}")
    
    def _fix_json_response(self, content: str) -> str:
        """Fix common JSON issues in OpenAI responses"""
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
    
    async def synthesize_problems(self, county_data: Dict[str, Any], county: str) -> Dict[str, Any]:
        """Synthesize comprehensive city report with separate calls for city info and problems"""
        
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
