"""
SF311 Intelligence Agent
Smart 311 data analysis and pattern recognition
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.services.agent_framework import BaseAgent, AgentTask, AgentResult, AgentStatus
from app.services.llm_service import LLMService
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../parsers'))

# Import the 311 parser function
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../parsers/311parser'))
    import demo_exact_request as sf311_parser
    get_sf311_offenses = sf311_parser.get_sf311_offenses
except ImportError:
    # Fallback - create a mock function for testing
    def get_sf311_offenses(page=1):
        """Mock SF311 data - returns 10 items per page with realistic SF coordinates"""
        offense_types = ["Graffiti", "Street Cleaning", "Blocked Driveway", "Pothole", "Broken Streetlight", 
                        "Illegal Dumping", "Noise Complaint", "Parking Violation", "Sidewalk Repair", "Tree Maintenance"]
        
        # Realistic SF coordinates for different neighborhoods
        sf_locations = [
            {"lat": 37.7749, "lng": -122.4194, "neighborhood": "Mission District", "address": "Mission St"},
            {"lat": 37.7849, "lng": -122.4094, "neighborhood": "SOMA", "address": "Howard St"},
            {"lat": 37.7611, "lng": -122.4369, "neighborhood": "Castro", "address": "Castro St"},
            {"lat": 37.7699, "lng": -122.4469, "neighborhood": "Haight", "address": "Haight St"},
            {"lat": 37.7804, "lng": -122.4602, "neighborhood": "Richmond", "address": "Geary Blvd"},
            {"lat": 37.7599, "lng": -122.4148, "neighborhood": "Sunset", "address": "Irving St"},
            {"lat": 37.8024, "lng": -122.4058, "neighborhood": "Marina", "address": "Chestnut St"},
            {"lat": 37.7946, "lng": -122.4094, "neighborhood": "Nob Hill", "address": "California St"}
        ]
        
        return [
            {
                "offense_type": offense_types[i % len(offense_types)],
                "description": f"311 request #{i+1} for {offense_types[i % len(offense_types)].lower()} in {sf_locations[i % len(sf_locations)]['neighborhood']}",
                "offense_id": f"311_{page}_{i+1:03d}",
                "coordinates": f"({sf_locations[i % len(sf_locations)]['lat']}, {sf_locations[i % len(sf_locations)]['lng']})",
                "address": f"{100+i} {sf_locations[i % len(sf_locations)]['address']}"
            }
            for i in range(10)
        ]

logger = logging.getLogger(__name__)

@dataclass
class SF311Task:
    """SF311 agent task"""
    task_id: str
    agent_id: str
    data: Dict[str, Any]
    pages: int = 5  # Number of pages to fetch
    filter_types: Optional[List[str]] = None  # Specific offense types to focus on
    min_severity: str = "low"  # Minimum severity filter
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class SF311Result:
    """SF311 agent result"""
    task_id: str
    agent_id: str
    status: AgentStatus
    data: Dict[str, Any]
    total_requests: int
    filtered_requests: int
    issues_identified: List[Dict[str, Any]]
    geographic_patterns: Dict[str, Any]
    temporal_patterns: Dict[str, Any]
    insights: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    execution_time: float = 0.0
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class SF311Agent(BaseAgent):
    """Intelligent SF311 agent for official city data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("sf311_agent", config)
        self.llm_service = LLMService()
        
        # High-priority offense types for San Francisco
        self.priority_types = [
            'Graffiti', 'Street or sidewalk cleaning', 'Blocked driveway & illegal parking',
            'Homeless encampment', 'Illegal dumping', 'Noise complaint',
            'Street light out', 'Pothole', 'Sewer overflow'
        ]
        
        # Geographic patterns for SF neighborhoods
        self.sf_neighborhoods = {
            'Mission District': {'lat_range': (37.75, 37.77), 'lng_range': (-122.42, -122.40)},
            'Tenderloin': {'lat_range': (37.78, 37.79), 'lng_range': (-122.42, -122.41)},
            'SOMA': {'lat_range': (37.77, 37.79), 'lng_range': (-122.40, -122.38)},
            'Castro': {'lat_range': (37.76, 37.78), 'lng_range': (-122.44, -122.42)},
            'Richmond': {'lat_range': (37.78, 37.80), 'lng_range': (-122.48, -122.46)},
            'Sunset': {'lat_range': (37.75, 37.77), 'lng_range': (-122.50, -122.48)}
        }
    
    async def execute(self, task: SF311Task) -> SF311Result:
        """Execute SF311 intelligence task"""
        logger.info(f"SF311 agent starting with {task.pages} pages")
        
        try:
            # Step 1: Fetch 311 data
            raw_data = await self._fetch_311_data(task)
            logger.info(f"Fetched {len(raw_data)} 311 requests")
            
            # Step 2: Smart filtering and analysis
            filtered_data = await self._filter_and_analyze(raw_data, task)
            logger.info(f"Filtered to {len(filtered_data)} relevant requests")
            
            # Step 3: Issue identification
            issues = await self._identify_issues(filtered_data)
            logger.info(f"Identified {len(issues)} community issues")
            
            # Step 4: Pattern analysis
            geographic_patterns = await self._analyze_geographic_patterns(filtered_data)
            temporal_patterns = await self._analyze_temporal_patterns(filtered_data)
            
            # Step 5: Generate insights
            insights = await self._generate_insights(issues, geographic_patterns, temporal_patterns)
            
            return SF311Result(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=AgentStatus.SUCCESS,
                data={
                    "raw_data": raw_data,
                    "filtered_data": filtered_data,
                    "issues": issues,
                    "geographic_patterns": geographic_patterns,
                    "temporal_patterns": temporal_patterns
                },
                total_requests=len(raw_data),
                filtered_requests=len(filtered_data),
                issues_identified=issues,
                geographic_patterns=geographic_patterns,
                temporal_patterns=temporal_patterns,
                insights=[
                    f"Processed {len(raw_data)} 311 requests",
                    f"Identified {len(issues)} community issues",
                    f"Top issue type: {self._get_top_issue_type(filtered_data)}",
                    f"Most active neighborhood: {geographic_patterns.get('most_active_neighborhood', 'Unknown')}"
                ]
            )
            
        except Exception as e:
            logger.error(f"SF311 agent failed: {str(e)}")
            raise
    
    async def _fetch_311_data(self, task: SF311Task) -> List[Dict[str, Any]]:
        """Simple 311 data fetching - no quality threshold"""
        all_data = []
        
        logger.info(f"Starting 311 data fetch for {task.pages} pages")
        
        for page in range(1, task.pages + 1):
            try:
                page_data = get_sf311_offenses(page=page)
                all_data.extend(page_data)
                logger.info(f"Fetched page {page}: {len(page_data)} requests")
                
            except Exception as e:
                logger.warning(f"Failed to fetch page {page}: {str(e)}")
                continue
        
        logger.info(f"Fetch complete: {len(all_data)} total requests from {task.pages} pages")
        return all_data
    
    async def _filter_and_analyze(self, raw_data: List[Dict[str, Any]], task: SF311Task) -> List[Dict[str, Any]]:
        """Smart filtering and analysis of 311 data"""
        filtered = []
        
        for request in raw_data:
            # Apply filters
            if not self._passes_filters(request, task):
                continue
            
            # Enhance with analysis
            enhanced_request = await self._enhance_request(request)
            filtered.append(enhanced_request)
        
        return filtered
    
    def _passes_filters(self, request: Dict[str, Any], task: SF311Task) -> bool:
        """Check if request passes filters"""
        offense_type = request.get('offense_type', '').lower()
        description = request.get('description', '').lower()
        
        # Filter by offense type if specified
        if task.filter_types:
            if not any(filter_type.lower() in offense_type for filter_type in task.filter_types):
                return False
        
        # More flexible priority type matching
        priority_keywords = [
            'graffiti', 'street', 'sidewalk', 'cleaning', 'blocked', 'driveway', 
            'parking', 'homeless', 'encampment', 'dumping', 'noise', 'light', 
            'pothole', 'sewer', 'overflow', 'illegal'
        ]
        
        # Check if any priority keywords match
        if not any(keyword in offense_type or keyword in description for keyword in priority_keywords):
            return False
        
        # Filter by severity
        severity = self._assess_severity(request)
        if severity == 'low' and task.min_severity == 'medium':
            return False
        if severity in ['low', 'medium'] and task.min_severity == 'high':
            return False
        
        return True
    
    async def _enhance_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance request with additional analysis"""
        enhanced = request.copy()
        
        # Add severity assessment
        enhanced['severity'] = self._assess_severity(request)
        
        # Add neighborhood detection
        enhanced['neighborhood'] = self._detect_neighborhood(request)
        
        # Add geographic analysis
        enhanced['geographic_analysis'] = self._analyze_geographic_context(request)
        
        # Add temporal analysis
        enhanced['temporal_analysis'] = self._analyze_temporal_context(request)
        
        return enhanced
    
    def _assess_severity(self, request: Dict[str, Any]) -> str:
        """Assess severity of 311 request"""
        offense_type = request.get('offense_type', '').lower()
        description = request.get('description', '').lower()
        
        # High severity indicators
        high_severity_terms = [
            'homeless encampment', 'sewer overflow', 'illegal dumping',
            'blocked driveway', 'graffiti', 'noise complaint'
        ]
        
        if any(term in offense_type or term in description for term in high_severity_terms):
            return 'high'
        
        # Medium severity indicators
        medium_severity_terms = [
            'street cleaning', 'sidewalk cleaning', 'street light',
            'pothole', 'parking'
        ]
        
        if any(term in offense_type or term in description for term in medium_severity_terms):
            return 'medium'
        
        return 'low'
    
    def _detect_neighborhood(self, request: Dict[str, Any]) -> Optional[str]:
        """Detect neighborhood from coordinates or address"""
        coordinates = request.get('coordinates', '')
        address = request.get('address', '')
        
        # Try to extract coordinates
        if coordinates:
            try:
                # Parse coordinates like "(37.7749, -122.4194)"
                coords_match = re.search(r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', coordinates)
                if coords_match:
                    lat = float(coords_match.group(1))
                    lng = float(coords_match.group(2))
                    
                    # Check against neighborhood boundaries
                    for neighborhood, bounds in self.sf_neighborhoods.items():
                        if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and
                            bounds['lng_range'][0] <= lng <= bounds['lng_range'][1]):
                            return neighborhood
            except (ValueError, IndexError):
                pass
        
        # Fallback to address analysis
        if address:
            address_lower = address.lower()
            for neighborhood in self.sf_neighborhoods.keys():
                if neighborhood.lower().replace(' ', '') in address_lower.replace(' ', ''):
                    return neighborhood
        
        return None
    
    def _analyze_geographic_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze geographic context"""
        neighborhood = self._detect_neighborhood(request)
        
        return {
            "neighborhood": neighborhood,
            "coordinates": request.get('coordinates', ''),
            "address": request.get('address', ''),
            "geographic_priority": self._assess_geographic_priority(neighborhood)
        }
    
    def _analyze_temporal_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal context"""
        # This would be enhanced with actual timestamp data
        return {
            "time_of_day": "unknown",  # Would extract from actual data
            "day_of_week": "unknown",
            "seasonal_pattern": "unknown"
        }
    
    def _assess_geographic_priority(self, neighborhood: Optional[str]) -> str:
        """Assess geographic priority based on neighborhood"""
        if not neighborhood:
            return "unknown"
        
        # High-priority neighborhoods for SF
        high_priority = ['Tenderloin', 'SOMA', 'Mission District']
        medium_priority = ['Castro', 'Richmond', 'Sunset']
        
        if neighborhood in high_priority:
            return "high"
        elif neighborhood in medium_priority:
            return "medium"
        else:
            return "low"
    
    async def _identify_issues(self, filtered_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert filtered 311 data into issues format for frontend"""
        issues = []
        
        if not filtered_data:
            return issues
        
        # Convert each filtered 311 request into an issue format
        for i, request in enumerate(filtered_data):
            # Extract coordinates if available - check both direct coordinates and geographic_analysis
            coordinates = None
            neighborhood = request.get('neighborhood')
            
            # First try direct coordinates
            if request.get('coordinates'):
                coords_str = request['coordinates']
                try:
                    # Parse coordinates like "(37.7749, -122.4194)"
                    coords_match = re.search(r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', coords_str)
                    if coords_match:
                        lat = float(coords_match.group(1))
                        lng = float(coords_match.group(2))
                        coordinates = [lng, lat]  # [longitude, latitude] for frontend
                except (ValueError, IndexError):
                    pass
            
            # If no direct coordinates, try geographic_analysis
            if not coordinates and request.get('geographic_analysis', {}).get('coordinates'):
                coords_str = request['geographic_analysis']['coordinates']
                try:
                    # Parse coordinates like "(37.7749, -122.4194)"
                    coords_match = re.search(r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', coords_str)
                    if coords_match:
                        lat = float(coords_match.group(1))
                        lng = float(coords_match.group(2))
                        coordinates = [lng, lat]  # [longitude, latitude] for frontend
                except (ValueError, IndexError):
                    pass
            
            # Extract neighborhood from geographic_analysis if not set
            if not neighborhood and request.get('geographic_analysis', {}).get('neighborhood'):
                neighborhood = request['geographic_analysis']['neighborhood']
            
            # Create issue from 311 request
            issue = {
                "id": request.get('offense_id', f"sf311_{i}"),
                "title": request.get('offense_type', 'SF311 Request'),
                "description": request.get('description', 'No description available'),
                "severity": request.get('severity', 'low'),
                "source": "311",
                "coordinates": coordinates,
                "neighborhood": neighborhood,
                "metadata": {
                    "offense_id": request.get('offense_id', ''),
                    "address": request.get('address', ''),
                    "offense_type": request.get('offense_type', ''),
                    "geographic_analysis": request.get('geographic_analysis', {}),
                    "temporal_analysis": request.get('temporal_analysis', {}),
                    "analysis_type": "direct_311_request"
                }
            }
            issues.append(issue)
        
        return issues
    
    async def _llm_analyze_311_issues(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to analyze 311 requests and identify community issues"""
        
        # Prepare data for LLM analysis
        requests_summary = []
        for req in requests[:15]:  # Limit to top 15 for analysis
            requests_summary.append({
                "type": req.get('offense_type', ''),
                "description": req.get('description', ''),
                "location": req.get('address', ''),
                "neighborhood": req.get('neighborhood', ''),
                "severity": req.get('severity', 'low')
            })
        
        requests_text = "\n".join([
            f"Type: {req['type']}\nDescription: {req['description']}\nLocation: {req['location']}\nNeighborhood: {req['neighborhood']}\nSeverity: {req['severity']}\n"
            for req in requests_summary
        ])
        
        prompt = f"""You are an expert urban planning analyst specializing in San Francisco. Analyze these 311 service requests and identify the most pressing community issues.

311 Service Requests:
{requests_text}

Analyze the data and identify 2-4 most significant community issues. For each issue:

1. **Issue Title**: Concise problem description (5-8 words)
2. **Description**: Detailed explanation with specific examples from the 311 data
3. **Severity**: high/medium/low based on frequency and impact
4. **Geographic Pattern**: Which neighborhoods are most affected
5. **Evidence**: Specific 311 requests that support this issue

Return ONLY a valid JSON array:
[
  {{
    "title": "Issue Title",
    "description": "Detailed description with specific examples from 311 data",
    "severity": "high|medium|low",
    "source": "311",
    "source_id": "311_analysis",
    "location": {{
      "city": "San Francisco",
      "neighborhood": "most affected neighborhood",
      "coordinates": null
    }},
    "metadata": {{
      "analysis_type": "llm_311_analysis",
      "evidence_requests": ["request1", "request2"],
      "confidence": 0.8,
      "geographic_pattern": "description of geographic pattern"
    }}
  }}
]

Focus on issues that:
- Appear multiple times in the data
- Have significant community impact
- Are actionable by city government
- Show clear geographic patterns
- Include specific examples from the 311 requests"""

        try:
            # Use the LLM service to analyze the requests
            response = self.llm_service.provider.client.messages.create(
                model=self.llm_service.provider.model,
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            # Extract JSON from response
            import json
            import re
            
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                issues = json.loads(json_match.group(0))
                logger.info(f"LLM identified {len(issues)} community issues from 311 data")
                return issues
            else:
                logger.warning("No valid JSON found in LLM response")
                return []
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            # Fallback to simple analysis
            return self._fallback_311_analysis(requests)
    
    def _fallback_311_analysis(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback simple 311 analysis without LLM"""
        issues = []
        
        # Group requests by type
        type_counts = {}
        for req in requests:
            req_type = req.get('offense_type', 'Unknown')
            if req_type not in type_counts:
                type_counts[req_type] = []
            type_counts[req_type].append(req)
        
        # Create issues for types with multiple requests
        for req_type, reqs in type_counts.items():
            if len(reqs) > 1:  # Only create issues for types with multiple requests
                issues.append({
                    "title": f"{req_type} Issues",
                    "description": f"Multiple {req_type} requests indicating ongoing community issues",
                    "severity": "high" if len(reqs) > 3 else "medium",
                    "source": "311",
                    "source_id": f"311_{req_type.lower().replace(' ', '_')}",
                    "location": {
                        "city": "San Francisco",
                        "neighborhood": reqs[0].get('neighborhood'),
                        "coordinates": None
                    },
                    "metadata": {
                        "analysis_type": "type_fallback",
                        "evidence_requests": [req.get('description', '')[:100] for req in reqs[:3]],
                        "confidence": 0.6,
                        "request_count": len(reqs)
                    }
                })
        
        return issues
    
    async def _analyze_geographic_patterns(self, filtered_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze geographic patterns"""
        neighborhood_counts = {}
        offense_type_counts = {}
        
        for request in filtered_data:
            neighborhood = request.get('neighborhood')
            offense_type = request.get('offense_type', '')
            
            if neighborhood:
                neighborhood_counts[neighborhood] = neighborhood_counts.get(neighborhood, 0) + 1
            
            offense_type_counts[offense_type] = offense_type_counts.get(offense_type, 0) + 1
        
        return {
            "neighborhood_counts": neighborhood_counts,
            "most_active_neighborhood": max(neighborhood_counts.keys(), key=neighborhood_counts.get) if neighborhood_counts else None,
            "offense_type_distribution": offense_type_counts,
            "geographic_hotspots": self._identify_geographic_hotspots(neighborhood_counts)
        }
    
    async def _analyze_temporal_patterns(self, filtered_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns"""
        # This would be enhanced with actual timestamp data
        return {
            "total_requests": len(filtered_data),
            "severity_distribution": self._calculate_severity_distribution(filtered_data),
            "temporal_insights": "Enhanced temporal analysis would require timestamp data"
        }
    
    def _identify_geographic_hotspots(self, neighborhood_counts: Dict[str, int]) -> List[Dict[str, Any]]:
        """Identify geographic hotspots"""
        if not neighborhood_counts:
            return []
        
        total_requests = sum(neighborhood_counts.values())
        hotspots = []
        
        for neighborhood, count in neighborhood_counts.items():
            if count > total_requests * 0.1:  # More than 10% of requests
                hotspots.append({
                    "neighborhood": neighborhood,
                    "request_count": count,
                    "percentage": (count / total_requests) * 100
                })
        
        return sorted(hotspots, key=lambda x: x['request_count'], reverse=True)
    
    def _calculate_severity_distribution(self, filtered_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate severity distribution"""
        distribution = {"high": 0, "medium": 0, "low": 0}
        for request in filtered_data:
            severity = request.get('severity', 'low')
            distribution[severity] += 1
        return distribution
    
    def _get_top_issue_type(self, filtered_data: List[Dict[str, Any]]) -> str:
        """Get top issue type"""
        if not filtered_data:
            return "Unknown"
        
        offense_types = [request.get('offense_type', '') for request in filtered_data]
        if not offense_types:
            return "Unknown"
        
        # Count offense types
        type_counts = {}
        for offense_type in offense_types:
            type_counts[offense_type] = type_counts.get(offense_type, 0) + 1
        
        return max(type_counts.keys(), key=type_counts.get)
    
    async def _generate_insights(self, issues: List[Dict[str, Any]], 
                               geographic_patterns: Dict[str, Any], 
                               temporal_patterns: Dict[str, Any]) -> List[str]:
        """Generate insights from analysis"""
        insights = []
        
        # Geographic insights
        if geographic_patterns.get('most_active_neighborhood'):
            insights.append(f"Most active neighborhood: {geographic_patterns['most_active_neighborhood']}")
        
        # Issue type insights
        if geographic_patterns.get('offense_type_distribution'):
            top_type = max(geographic_patterns['offense_type_distribution'].keys(), 
                          key=geographic_patterns['offense_type_distribution'].get)
            insights.append(f"Most common issue type: {top_type}")
        
        # Severity insights
        severity_dist = temporal_patterns.get('severity_distribution', {})
        if severity_dist.get('high', 0) > 0:
            insights.append(f"High severity issues detected: {severity_dist['high']}")
        
        return insights
    
    def _get_quality_threshold(self) -> float:
        """Get adaptive quality threshold based on past success"""
        successful_patterns = self.memory.get_successful_patterns()
        
        # Default quality threshold
        threshold = 0.7
        
        # Learn from successful patterns
        if successful_patterns:
            recent_success = successful_patterns[-1]
            if 'quality_threshold' in recent_success.get('pattern', {}):
                threshold = recent_success['pattern']['quality_threshold']
        
        return threshold
    
    def _assess_data_quality(self, data: List[Dict[str, Any]]) -> float:
        """Assess the quality of fetched 311 data"""
        if not data:
            return 0.0
        
        quality_score = 0.0
        total_requests = len(data)
        
        # Quality metrics
        with_coordinates = sum(1 for req in data if req.get('coordinates'))
        with_addresses = sum(1 for req in data if req.get('address'))
        with_descriptions = sum(1 for req in data if req.get('description'))
        priority_types = sum(1 for req in data if req.get('offense_type') in self.priority_types)
        
        # Calculate quality score (0-1)
        coordinate_quality = with_coordinates / total_requests if total_requests > 0 else 0
        address_quality = with_addresses / total_requests if total_requests > 0 else 0
        description_quality = with_descriptions / total_requests if total_requests > 0 else 0
        priority_quality = priority_types / total_requests if total_requests > 0 else 0
        
        # Weighted quality score
        quality_score = (
            coordinate_quality * 0.3 +
            address_quality * 0.2 +
            description_quality * 0.3 +
            priority_quality * 0.2
        )
        
        return quality_score
    
    def _learn_from_fetch(self, data: List[Dict[str, Any]], pages_fetched: int, quality_threshold: float):
        """Learn from 311 data fetch session"""
        if data:
            # Record successful fetch pattern
            pattern = {
                'pages_fetched': pages_fetched,
                'quality_threshold': quality_threshold,
                'data_quality': self._assess_data_quality(data),
                'total_requests': len(data)
            }
            
            # Create mock result for memory
            mock_result = type('MockResult', (), {
                'data': {'requests_found': len(data), 'quality': self._assess_data_quality(data)},
                'status': 'success'
            })()
            
            self.memory.record_success(pattern, mock_result)
    
    def _get_adaptive_filters(self) -> Dict[str, Any]:
        """Get adaptive filtering parameters based on past success"""
        successful_patterns = self.memory.get_successful_patterns()
        
        # Default filters
        filters = {
            'priority_types': self.priority_types,
            'min_severity': 'low',
            'geographic_priority': 'all'
        }
        
        # Learn from successful patterns
        if successful_patterns:
            recent_success = successful_patterns[-1]
            if 'filters' in recent_success.get('pattern', {}):
                filters.update(recent_success['pattern']['filters'])
        
        return filters
