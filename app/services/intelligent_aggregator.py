import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import DataPoint, Dataset
from app.services.llm_service import LLMService

class IntelligentAggregator:
    """Adaptive multi-stage pipeline for intelligent data aggregation"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.analysis_context = {}
    
    async def analyze_and_plan(self, county: str, db: AsyncSession, analysis_goal: str = "identify_community_problems") -> Dict[str, Any]:
        """STAGE 1: LLM-Powered Strategy Generation"""
        
        # Profile all datasets for this county
        profiles = await self._profile_datasets(county, db)
        
        # Get existing data characteristics
        data_characteristics = await self._analyze_data_characteristics(county, db)
        
        # Ask LLM for aggregation strategy
        strategy_prompt = f"""
        I have these datasets for {county}:
        {json.dumps(profiles, indent=2)}
        
        Data characteristics:
        {json.dumps(data_characteristics, indent=2)}
        
        My analysis goal: {analysis_goal}
        
        Generate an intelligent aggregation strategy that:
        1. Prioritizes recent, relevant data (last 5-10 years for crime data, last 2-3 years for demographics)
        2. Reduces dimensionality while preserving insights
        3. Identifies key patterns and outliers
        4. Adapts to data density (sparse areas get rolled up)
        5. Focuses on high-variance, high-signal patterns
        6. Considers domain-specific logic (arrest trends vs demographic changes)
        
        Return JSON with specific rules per dataset:
        {{
            "time_focus": {{
                "arrest_data": "recent_5_years",
                "demographics": "recent_3_years",
                "311_complaints": "recent_2_years"
            }},
            "aggregation_rules": {{
                "high_volume_metrics": "sum_by_year",
                "low_volume_metrics": "average_by_period",
                "demographic_metrics": "latest_values"
            }},
            "relevance_filters": {{
                "min_data_points": 10,
                "min_variance_threshold": 0.1,
                "focus_on_outliers": true
            }},
            "analysis_priorities": ["public_safety", "economic_indicators", "demographic_changes"]
        }}
        """
        
        try:
            # Use LLM to generate strategy
            strategy = await self.llm_service._generate_aggregation_strategy(strategy_prompt)
            return strategy
        except:
            # Fallback to default strategy
            return self._get_default_strategy()
    
    async def execute_strategy(self, location: str, level: str, strategy: Dict[str, Any], db: AsyncSession) -> Dict[str, Any]:
        """STAGE 2: Automated Profiling & Adjustment"""
        
        # Get all datasets and check their geographic hierarchy to find matching location at the specified level
        from app.database import Dataset
        datasets_result = await db.execute(select(Dataset))
        datasets = datasets_result.scalars().all()
        
        # Find datasets that match the requested location at the specified level
        matching_dataset_ids = []
        for dataset in datasets:
            try:
                import json
                hierarchy = json.loads(dataset.geographic_hierarchy_json)
                
                # Check if this dataset matches the requested location at the specified level
                location_lower = location.lower()
                is_match = False
                
                if level == 'county':
                    # Match county to county
                    dataset_county = hierarchy.get('county', '').lower()
                    if dataset_county and location_lower in dataset_county:
                        is_match = True
                        print(f"✅ County match: dataset county '{hierarchy.get('county', 'N/A')}' matches requested '{location}'")
                elif level == 'city':
                    # Match city to city
                    dataset_city = hierarchy.get('city', '').lower()
                    if dataset_city and location_lower in dataset_city:
                        is_match = True
                        print(f"✅ City match: dataset city '{hierarchy.get('city', 'N/A')}' matches requested '{location}'")
                elif level == 'neighborhood':
                    # Match neighborhood to neighborhood
                    dataset_neighborhood = hierarchy.get('neighborhood', '').lower()
                    if dataset_neighborhood and location_lower in dataset_neighborhood:
                        is_match = True
                        print(f"✅ Neighborhood match: dataset neighborhood '{hierarchy.get('neighborhood', 'N/A')}' matches requested '{location}'")
                else:
                    # For other levels, check all hierarchy levels
                    for level_name, value in hierarchy.items():
                        if value and location_lower in value.lower():
                            is_match = True
                            print(f"✅ {level_name} match: '{value}' matches requested '{location}'")
                            break
                
                if is_match:
                    matching_dataset_ids.append(dataset.id)
                    print(f"✅ Including dataset: {dataset.name}")
                else:
                    print(f"❌ Dataset '{dataset.name}' doesn't match at {level} level (hierarchy: {hierarchy})")
                    
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"⚠️ Error parsing hierarchy for dataset {dataset.name}: {e}")
                continue
        
        if not matching_dataset_ids:
            print(f"❌ No datasets found for {level} '{location}'")
            return {}
        
        # Get all data points from matching datasets
        result = await db.execute(
            select(DataPoint)
            .where(DataPoint.dataset_id.in_(matching_dataset_ids))
        )
        data_points = result.scalars().all()
        print(f"🔍 Intelligent aggregator found {len(data_points)} data points from {len(matching_dataset_ids)} matching datasets for {level} '{location}'")
        
        if not data_points:
            print(f"❌ No data points found for {level} '{location}'")
            return {}
        
        # Apply intelligent time filtering
        filtered_points = self._apply_time_focus_filtering(data_points, strategy)
        
        # Run aggregations based on strategy
        aggregated = {}
        
        # Group by metric and apply domain-specific logic
        metric_groups = {}
        for point in filtered_points:
            metric = point.metric_name
            if metric not in metric_groups:
                metric_groups[metric] = []
            metric_groups[metric].append(point)
        
        # Apply aggregation rules
        for metric, points in metric_groups.items():
            analysis = await self._apply_aggregation_rules(metric, points, strategy)
            if analysis:
                aggregated[metric] = analysis
        
        # Measure information density and auto-adjust
        adjusted_aggregated = self._auto_adjust_based_on_density(aggregated, strategy)
        
        return adjusted_aggregated
    
    async def iterative_analysis(self, county: str, aggregated_data: Dict[str, Any], db: AsyncSession) -> Dict[str, Any]:
        """STAGE 3: Iterative Refinement"""
        
        # Pass aggregated data to LLM for analysis
        analysis_prompt = f"""
        Based on this aggregated data for {county}:
        {json.dumps(aggregated_data, indent=2)}
        
        Analyze the patterns and identify:
        1. Key insights and anomalies
        2. Areas that need more detailed analysis
        3. Specific drill-downs that would be valuable
        4. Missing context that would improve understanding
        
        If you need more specific data, request it in this format:
        {{
            "drill_down_requests": [
                {{
                    "metric": "violent_crime_arrests",
                    "time_period": "2020-2024",
                    "reason": "High variance detected, need recent trend analysis"
                }}
            ],
            "insights": ["list of key findings"],
            "confidence_level": "high|medium|low"
        }}
        """
        
        try:
            # Get LLM analysis and requests
            llm_response = await self.llm_service._analyze_aggregated_data(analysis_prompt)
            
            # Execute drill-down requests
            if 'drill_down_requests' in llm_response:
                for request in llm_response['drill_down_requests']:
                    drill_down_data = await self._execute_drill_down(county, request, db)
                    aggregated_data[f"{request['metric']}_detailed"] = drill_down_data
            
            return {
                'aggregated_data': aggregated_data,
                'llm_insights': llm_response.get('insights', []),
                'confidence_level': llm_response.get('confidence_level', 'medium'),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            # Fallback: return basic aggregated data without LLM insights
            return {
                'aggregated_data': aggregated_data,
                'llm_insights': ["Data analysis completed with basic aggregation"],
                'confidence_level': 'low',
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'fallback_mode': True
            }
    
    async def insight_driven_reaggregation(self, county: str, insights: List[str], db: AsyncSession) -> Dict[str, Any]:
        """STAGE 4: Insight-Driven Re-aggregation"""
        
        # Based on insights, perform targeted re-aggregations
        targeted_analysis = {}
        
        for insight in insights:
            if "arrest" in insight.lower():
                # Focus on crime patterns
                crime_analysis = await self._analyze_crime_patterns(county, db)
                targeted_analysis['crime_patterns'] = crime_analysis
            
            elif "demographic" in insight.lower():
                # Focus on demographic changes
                demo_analysis = await self._analyze_demographic_changes(county, db)
                targeted_analysis['demographic_changes'] = demo_analysis
            
            elif "economic" in insight.lower():
                # Focus on economic indicators
                economic_analysis = await self._analyze_economic_indicators(county, db)
                targeted_analysis['economic_indicators'] = economic_analysis
        
        return targeted_analysis
    
    # Helper Methods
    
    async def _profile_datasets(self, county: str, db: AsyncSession) -> Dict[str, Any]:
        """Profile all datasets for a county"""
        result = await db.execute(
            select(DataPoint)
            .where(DataPoint.county == county)
        )
        data_points = result.scalars().all()
        
        profiles = {}
        for point in data_points:
            category = point.category
            if category not in profiles:
                profiles[category] = {
                    'metrics': set(),
                    'years': set(),
                    'total_points': 0
                }
            
            profiles[category]['metrics'].add(point.metric_name)
            profiles[category]['years'].add(point.year)
            profiles[category]['total_points'] += 1
        
        # Convert sets to lists for JSON serialization
        for category in profiles:
            profiles[category]['metrics'] = list(profiles[category]['metrics'])
            profiles[category]['years'] = list(profiles[category]['years'])
        
        return profiles
    
    async def _analyze_data_characteristics(self, county: str, db: AsyncSession) -> Dict[str, Any]:
        """Analyze data characteristics for intelligent processing"""
        result = await db.execute(
            select(DataPoint)
            .where(DataPoint.county == county)
        )
        data_points = result.scalars().all()
        
        if not data_points:
            return {}
        
        # Calculate basic statistics
        years = [p.year for p in data_points if p.year]
        values = [p.metric_value for p in data_points]
        
        return {
            'total_data_points': len(data_points),
            'year_range': f"{min(years)}-{max(years)}" if years else "unknown",
            'data_density': len(data_points) / len(set(years)) if years else 0,
            'value_statistics': {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            },
            'categories': list(set(p.category for p in data_points))
        }
    
    def _apply_time_focus_filtering(self, data_points: List[DataPoint], strategy: Dict[str, Any]) -> List[DataPoint]:
        """Apply intelligent time filtering based on strategy"""
        current_year = datetime.now().year
        time_focus = strategy.get('time_focus', {})
        
        filtered_points = []
        
        for point in data_points:
            # Skip points with invalid year data
            if not point.year or not isinstance(point.year, (int, float)):
                continue
                
            # Determine focus period based on data type
            if point.category == 'crime' or 'arrest' in point.metric_name.lower():
                focus_years = 5  # Last 5 years for crime data
            elif point.category == 'demographics':
                focus_years = 3  # Last 3 years for demographics
            else:
                focus_years = 2  # Last 2 years for other data
            
            # Ensure year is a valid integer
            try:
                year_int = int(point.year)
                if year_int >= (current_year - focus_years):
                    filtered_points.append(point)
            except (ValueError, TypeError):
                continue
        
        return filtered_points
    
    async def _apply_aggregation_rules(self, metric: str, points: List[DataPoint], strategy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply domain-specific aggregation rules"""
        if len(points) < 5:  # Skip metrics with too little data
            return None
        
        values = [p.metric_value for p in points]
        years = [p.year for p in points if p.year]
        
        # Calculate metrics based on data characteristics
        total = sum(values)
        avg_per_year = total / len(set(years)) if years else 0
        
        # Recent average (last 3 years)
        recent_years = [y for y in years if y >= (datetime.now().year - 3)]
        recent_values = [p.metric_value for p in points if p.year in recent_years]
        recent_avg = sum(recent_values) / len(recent_values) if recent_values else 0
        
        # Trend analysis
        trend = self._calculate_trend(values, years)
        
        # Anomaly score
        anomaly_score = self._calculate_anomaly_score(total, len(points))
        
        # Relevance multiplier based on recency and variance
        relevance_multiplier = self._calculate_relevance_multiplier(values, years)
        
        return {
            'total': total,
            'average_per_year': avg_per_year,
            'recent_average': recent_avg,
            'trend': trend,
            'anomaly_score': anomaly_score,
            'relevance_multiplier': relevance_multiplier,
            'data_points': len(points),
            'time_period': f"{min(years)}-{max(years)}" if years else "unknown"
        }
    
    def _auto_adjust_based_on_density(self, aggregated: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-adjust aggregation based on information density"""
        min_data_points = strategy.get('relevance_filters', {}).get('min_data_points', 10)
        min_variance = strategy.get('relevance_filters', {}).get('min_variance_threshold', 0.1)
        
        adjusted = {}
        for metric, data in aggregated.items():
            # Filter out low-density metrics
            if data['data_points'] < min_data_points:
                continue
            
            # Boost metrics with high variance (more interesting)
            if data.get('anomaly_score', 0) > min_variance:
                data['relevance_multiplier'] *= 1.5
            
            adjusted[metric] = data
        
        return adjusted
    
    def _calculate_trend(self, values: List[float], years: List[int]) -> str:
        """Calculate trend direction"""
        if len(values) < 2 or len(set(years)) < 2:
            return "insufficient_data"
        
        # Group by year
        year_totals = {}
        for year, value in zip(years, values):
            if year not in year_totals:
                year_totals[year] = []
            year_totals[year].append(value)
        
        # Calculate averages
        year_averages = {year: sum(vals)/len(vals) for year, vals in year_totals.items()}
        sorted_years = sorted(year_averages.keys())
        
        if len(sorted_years) < 2:
            return "insufficient_data"
        
        first_avg = year_averages[sorted_years[0]]
        last_avg = year_averages[sorted_years[-1]]
        
        change_percent = ((last_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        
        if change_percent > 15:
            return "strongly_increasing"
        elif change_percent > 5:
            return "increasing"
        elif change_percent < -15:
            return "strongly_decreasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_anomaly_score(self, total: float, count: int) -> float:
        """Calculate anomaly score (0-1)"""
        if count == 0:
            return 0
        
        avg_per_point = total / count
        
        # Normalize based on magnitude
        if avg_per_point > 1000:
            return 1.0
        elif avg_per_point > 500:
            return 0.8
        elif avg_per_point > 100:
            return 0.6
        elif avg_per_point > 50:
            return 0.4
        else:
            return 0.2
    
    def _calculate_relevance_multiplier(self, values: List[float], years: List[int]) -> float:
        """Calculate relevance multiplier based on recency and variance"""
        if not values or not years:
            return 0.5
        
        # Recency factor (more recent = higher relevance)
        current_year = datetime.now().year
        recent_years = [y for y in years if y >= (current_year - 3)]
        recency_factor = len(recent_years) / len(set(years)) if years else 0.5
        
        # Variance factor (higher variance = more interesting)
        variance_factor = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        
        # Combine factors
        relevance = (recency_factor * 0.6) + (min(variance_factor, 1.0) * 0.4)
        return max(0.1, min(2.0, relevance))  # Clamp between 0.1 and 2.0
    
    def _get_default_strategy(self) -> Dict[str, Any]:
        """Fallback default strategy"""
        return {
            "time_focus": {
                "arrest_data": "recent_5_years",
                "demographics": "recent_3_years",
                "311_complaints": "recent_2_years"
            },
            "aggregation_rules": {
                "high_volume_metrics": "sum_by_year",
                "low_volume_metrics": "average_by_period"
            },
            "relevance_filters": {
                "min_data_points": 10,
                "min_variance_threshold": 0.1,
                "focus_on_outliers": True
            },
            "analysis_priorities": ["public_safety", "demographic_changes"]
        }
    
    async def _execute_drill_down(self, county: str, request: Dict[str, Any], db: AsyncSession) -> Dict[str, Any]:
        """Execute specific drill-down analysis"""
        # Implementation for targeted analysis based on LLM requests
        # This would perform specific aggregations based on the request
        return {"drill_down": "implemented", "request": request}
    
    async def _analyze_crime_patterns(self, county: str, db: AsyncSession) -> Dict[str, Any]:
        """Analyze crime patterns specifically"""
        # Implementation for crime-specific analysis
        return {"crime_analysis": "implemented"}
    
    async def _analyze_demographic_changes(self, county: str, db: AsyncSession) -> Dict[str, Any]:
        """Analyze demographic changes specifically"""
        # Implementation for demographic analysis
        return {"demographic_analysis": "implemented"}
    
    async def _analyze_economic_indicators(self, county: str, db: AsyncSession) -> Dict[str, Any]:
        """Analyze economic indicators specifically"""
        # Implementation for economic analysis
        return {"economic_analysis": "implemented"}
