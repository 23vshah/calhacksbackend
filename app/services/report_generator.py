import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import DataPoint, GeneratedReport
from app.services.llm_service import LLMService
from app.services.intelligent_aggregator import IntelligentAggregator
from app.services.geographic_aggregator import GeographicAggregator
from app.models import CityReport, Problem, GeographicLevel

class ReportGeneratorService:
    def __init__(self):
        self.llm_service = LLMService()
        self.intelligent_aggregator = IntelligentAggregator()
        self.geographic_aggregator = GeographicAggregator()
    
    async def generate_report(self, location: str, level: str, db: AsyncSession) -> CityReport:
        """Generate or retrieve cached city report"""
        
        # Check for cached report first (DISABLED FOR DEBUGGING)
        # cached_report = await self._get_cached_report(location, db)
        # if cached_report:
        #     return cached_report
        print("🚫 Cache bypassed for debugging")
        
        # Filter datasets by geographic relevance
        relevant_datasets = await self._filter_datasets_by_geographic_relevance(location, level, db)
        print(f"🗺️ Found {len(relevant_datasets)} relevant datasets for {location} at {level} level")
        
        # Generate new report using intelligent aggregation
        location_data = await self._generate_intelligent_analysis(location, level, relevant_datasets, db)
        
        # Debug: Print what's being sent to LLM
        print(f"📊 Data being sent to LLM for {location}:")
        print(f"Data type: {type(location_data)}")
        print(f"Data keys: {list(location_data.keys()) if isinstance(location_data, dict) else 'Not a dict'}")
        print(f"Sample data: {str(location_data)[:500]}...")
        
        problems = await self.llm_service.synthesize_problems(location_data, location)
        
        # Create report
        report = CityReport(
            county=location,
            generated_at=datetime.utcnow(),
            cached=False,
            summary={
                "data_sources": await self._get_data_sources(location, db),
                "last_data_update": await self._get_last_update(location, db),
                "geographic_level": level,
                "relevant_datasets": [d.name for d in relevant_datasets]
            },
            problems=[self._format_problem(problem, location) for problem in problems]
        )
        
        # Cache the report
        await self._cache_report(report, db)
        
        return report
    
    async def _filter_datasets_by_geographic_relevance(self, location: str, level: str, db: AsyncSession) -> List[Any]:
        """Filter datasets by geographic relevance for the requested location and level"""
        
        from app.database import Dataset
        from sqlalchemy import select
        
        # Get all datasets
        result = await db.execute(select(Dataset))
        all_datasets = result.scalars().all()
        
        relevant_datasets = []
        
        for dataset in all_datasets:
            # Check if dataset is relevant for the requested level
            if self._is_dataset_relevant(dataset, location, level):
                relevant_datasets.append(dataset)
                print(f"✅ Including dataset: {dataset.name} (level: {dataset.geographic_level})")
            else:
                print(f"❌ Excluding dataset: {dataset.name} (level: {dataset.geographic_level})")
        
        return relevant_datasets
    
    def _is_dataset_relevant(self, dataset, location: str, level: str) -> bool:
        """Check if a dataset is relevant for the requested location and level"""
        
        # Convert string level to enum for comparison
        try:
            requested_level = GeographicLevel(level)
        except ValueError:
            # Invalid level, include all datasets
            return True
        
        # Check geographic level compatibility
        if not self._check_geographic_level_compatibility(dataset.geographic_level, requested_level):
            return False
        
        # Check geographic scope compatibility using hierarchy
        return self._check_geographic_scope_compatibility(dataset, location, level)
    
    def _check_geographic_level_compatibility(self, dataset_level: GeographicLevel, requested_level: GeographicLevel) -> bool:
        """Check if dataset level is compatible with requested level"""
        if requested_level == GeographicLevel.COUNTY:
            return dataset_level == GeographicLevel.COUNTY
        elif requested_level == GeographicLevel.CITY:
            # For city-level reports, include county, city, AND neighborhood data
            return dataset_level in [GeographicLevel.COUNTY, GeographicLevel.CITY, GeographicLevel.NEIGHBORHOOD]
        elif requested_level == GeographicLevel.NEIGHBORHOOD:
            return dataset_level in [GeographicLevel.COUNTY, GeographicLevel.CITY, GeographicLevel.NEIGHBORHOOD]
        elif requested_level == GeographicLevel.STATE:
            return dataset_level == GeographicLevel.STATE
        elif requested_level == GeographicLevel.REGION:
            return dataset_level == GeographicLevel.REGION
        else:
            return True
    
    def _check_geographic_scope_compatibility(self, dataset, location: str, level: str) -> bool:
        """Check if dataset geographic scope matches the requested location"""
        import json
        
        try:
            # Parse geographic hierarchy from dataset
            hierarchy_data = json.loads(dataset.geographic_hierarchy_json)
            
            # Check if the requested location matches any level in the hierarchy
            location_lower = location.lower()
            
            # Check each level of the hierarchy
            for hierarchy_level, hierarchy_value in hierarchy_data.items():
                if hierarchy_value and location_lower in hierarchy_value.lower():
                    return True
            
            # Special case: if requesting a city, include ALL datasets that have that city in their hierarchy
            if level == "city":
                # Check if the requested city appears anywhere in the hierarchy
                for hierarchy_level, hierarchy_value in hierarchy_data.items():
                    if hierarchy_value and location_lower in hierarchy_value.lower():
                        return True
                
                # Also check if the dataset contains neighborhoods from that city
                # (This is more permissive - include if city matches)
                if hierarchy_data.get("city") and location_lower in hierarchy_data["city"].lower():
                    return True
            
            # For neighborhood requests, be more specific
            if level == "neighborhood":
                # Check if the specific neighborhood matches
                for hierarchy_level, hierarchy_value in hierarchy_data.items():
                    if hierarchy_value and location_lower in hierarchy_value.lower():
                        return True
            
            return False
            
        except (json.JSONDecodeError, AttributeError):
            # If hierarchy parsing fails, fall back to including the dataset
            return True
    
    async def _get_cached_report(self, county: str, db: AsyncSession) -> Optional[CityReport]:
        """Check for recent cached report"""
        result = await db.execute(
            select(GeneratedReport)
            .where(GeneratedReport.county == county)
            .order_by(GeneratedReport.generated_at.desc())
            .limit(1)
        )
        cached = result.scalar_one_or_none()
        
        if cached:
            # Check if cache is fresh (within 1 hour for testing)
            from datetime import datetime, timedelta
            if cached.generated_at > datetime.utcnow() - timedelta(hours=1):
                report_data = json.loads(cached.report_json)
                # Convert ISO string back to datetime if needed
                if 'generated_at' in report_data and isinstance(report_data['generated_at'], str):
                    report_data['generated_at'] = datetime.fromisoformat(report_data['generated_at'].replace('Z', '+00:00'))
                return CityReport(**report_data)
        
        return None
    
    async def _aggregate_county_data(self, county: str, db: AsyncSession) -> Dict[str, Any]:
        """Advanced data aggregation with weighted anomaly scores and trend analysis"""
        
        # Get all data points for the county
        result = await db.execute(
            select(DataPoint)
            .where(DataPoint.county == county)
        )
        data_points = result.scalars().all()
        
        if not data_points:
            return {}
        
        # Get metric weights
        weights = await self._get_metric_weights(db)
        
        # Analyze data by metric with time series
        metric_analysis = {}
        
        for point in data_points:
            metric = point.metric_name
            year = point.year
            value = point.metric_value
            
            if metric not in metric_analysis:
                metric_analysis[metric] = {
                    'values': [],
                    'years': [],
                    'total': 0,
                    'count': 0,
                    'weight': weights.get(metric, 0.5)  # Default weight
                }
            
            metric_analysis[metric]['values'].append(value)
            metric_analysis[metric]['years'].append(year)
            metric_analysis[metric]['total'] += value
            metric_analysis[metric]['count'] += 1
        
        # Calculate sophisticated metrics
        aggregated = {}
        
        for metric, data in metric_analysis.items():
            if data['count'] == 0:
                continue
                
            # Basic aggregation
            total = data['total']
            avg_per_year = total / len(set(data['years'])) if data['years'] else 0
            
            # Trend analysis
            trend = self._calculate_trend(data['values'], data['years'])
            
            # Anomaly score (simplified - in real implementation would compare to state averages)
            anomaly_score = self._calculate_anomaly_score(total, data['count'])
            
            # Weighted importance score
            weighted_score = anomaly_score * data['weight']
            
            aggregated[metric] = {
                'total': total,
                'average_per_year': avg_per_year,
                'trend': trend,
                'anomaly_score': anomaly_score,
                'weighted_score': weighted_score,
                'weight': data['weight'],
                'data_points': data['count']
            }
        
        # Sort by weighted score to identify top issues
        sorted_metrics = sorted(aggregated.items(), key=lambda x: x[1]['weighted_score'], reverse=True)
        
        # Return top metrics with context
        return {
            'top_metrics': dict(sorted_metrics[:10]),  # Top 10 most concerning metrics
            'county': county,
            'total_data_points': len(data_points),
            'analysis_summary': self._generate_analysis_summary(aggregated)
        }
    
    async def _get_data_sources(self, county: str, db: AsyncSession) -> List[str]:
        """Get list of data sources for county"""
        # This would query the datasets table
        # For now, return placeholder
        return ["arrest_data_1980_2024"]
    
    async def _get_last_update(self, county: str, db: AsyncSession) -> Optional[str]:
        """Get last data update timestamp"""
        # This would query the most recent data point
        # For now, return placeholder
        return "2024-01-01"
    
    def _format_problem(self, problem_data: Dict[str, Any], county: str) -> Problem:
        """Format Claude's problem data into Problem model"""
        problem_id = f"{county.lower().replace(' ', '_')}_{hash(problem_data['title']) % 1000:03d}"
        
        return Problem(
            id=problem_id,
            title=problem_data["title"],
            severity=problem_data.get("severity", "medium"),
            description=problem_data["description"],
            metrics=problem_data.get("metrics", {}),
            solution=problem_data["solution"]
        )
    
    async def _cache_report(self, report: CityReport, db: AsyncSession):
        """Cache the generated report"""
        # Create hash of data for cache invalidation
        # Convert datetime objects to strings for JSON serialization
        report_data = report.model_dump()
        if 'generated_at' in report_data:
            report_data['generated_at'] = report_data['generated_at'].isoformat()
        
        data_hash = hashlib.md5(
            json.dumps(report_data, sort_keys=True).encode()
        ).hexdigest()
        
        cached_report = GeneratedReport(
            county=report.county,
            report_json=json.dumps(report_data),
            data_snapshot_hash=data_hash
        )
        
        db.add(cached_report)
        await db.commit()
    
    async def _generate_intelligent_analysis(self, location: str, level: str, relevant_datasets: List[Any], db: AsyncSession) -> Dict[str, Any]:
        """Generate intelligent analysis using hierarchical geographic aggregation"""
        
        print(f"🧠 Starting intelligent analysis for {location} at {level} level")
        
        try:
            # Use geographic aggregator for hierarchical analysis
            print("🗺️ STAGE 1: Geographic hierarchical analysis...")
            geographic_analysis = await self.geographic_aggregator.get_hierarchical_analysis(
                location, level, db
            )
            print(f"✅ Geographic analysis: {len(geographic_analysis.get('target_data', {}))} metrics")
            
            # STAGE 2: LLM-Powered Strategy Generation
            print("📋 STAGE 2: Generating aggregation strategy...")
            strategy = await self.intelligent_aggregator.analyze_and_plan(
                location, db, "identify_community_problems"
            )
            print(f"✅ Strategy generated: {strategy}")
            
            # STAGE 3: Execute strategy with geographic context
            print("⚙️ STAGE 3: Executing strategy with geographic context...")
            aggregated_data = await self.intelligent_aggregator.execute_strategy(
                location, strategy, db
            )
            print(f"✅ Aggregated data: {len(aggregated_data)} metrics")
            
            # STAGE 4: Combine geographic and intelligent analysis
            print("🔄 STAGE 4: Combining analyses...")
            combined_analysis = {
                'target_data': geographic_analysis.get('target_data', {}),
                'geographic_context': geographic_analysis.get('geographic_context', {}),
                'regional_trends': geographic_analysis.get('regional_trends', {}),
                'balanced_analysis': geographic_analysis.get('balanced_analysis', {}),
                'intelligent_aggregation': aggregated_data,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'geographic_level': level,
                'relevant_datasets': [d.name for d in relevant_datasets]
            }
            
            print(f"🎉 Intelligent analysis completed for {location}")
            return combined_analysis
            
        except Exception as e:
            print(f"❌ Intelligent analysis failed: {e}")
            # Fallback to basic aggregation
            return await self._fallback_aggregation(location, db)
    
    async def _get_metric_weights(self, db: AsyncSession) -> Dict[str, float]:
        """Get importance weights for metrics"""
        from app.database import DataWeight
        
        result = await db.execute(select(DataWeight))
        weights = result.scalars().all()
        
        return {weight.metric_name: weight.weight for weight in weights}
    
    def _calculate_trend(self, values: List[float], years: List[int]) -> str:
        """Calculate trend direction (increasing, decreasing, stable)"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        if len(set(years)) < 2:
            return "no_time_variation"
        
        # Group by year and calculate averages
        year_totals = {}
        for year, value in zip(years, values):
            if year not in year_totals:
                year_totals[year] = []
            year_totals[year].append(value)
        
        # Calculate average per year
        year_averages = {year: sum(vals)/len(vals) for year, vals in year_totals.items()}
        
        if len(year_averages) < 2:
            return "insufficient_data"
        
        # Sort by year
        sorted_years = sorted(year_averages.keys())
        first_avg = year_averages[sorted_years[0]]
        last_avg = year_averages[sorted_years[-1]]
        
        change_percent = ((last_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        
        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_anomaly_score(self, total: float, count: int) -> float:
        """Calculate anomaly score (0-1, higher = more concerning)"""
        if count == 0:
            return 0
        
        # Simple anomaly detection based on magnitude
        # In a real implementation, this would compare to state/county averages
        avg_per_point = total / count
        
        # Normalize to 0-1 scale (this is simplified)
        # Higher values indicate more concerning patterns
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
    
    def _generate_analysis_summary(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of the analysis"""
        if not aggregated:
            return {}
        
        # Find most concerning metrics
        top_concerns = sorted(aggregated.items(), key=lambda x: x[1]['weighted_score'], reverse=True)[:3]
        
        # Calculate overall county health score
        total_weighted_score = sum(data['weighted_score'] for data in aggregated.values())
        avg_weighted_score = total_weighted_score / len(aggregated) if aggregated else 0
        
        # Determine overall risk level
        if avg_weighted_score > 0.7:
            risk_level = "high"
        elif avg_weighted_score > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            'risk_level': risk_level,
            'overall_score': avg_weighted_score,
            'top_concerns': [metric for metric, _ in top_concerns],
            'total_metrics_analyzed': len(aggregated),
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    async def _fallback_aggregation(self, location: str, db: AsyncSession) -> Dict[str, Any]:
        """Fallback to basic aggregation if intelligent analysis fails"""
        print("🔄 Using fallback aggregation...")
        
        # Get all data points for the location using geographic hierarchy
        data_points = []
        
        # Get all datasets and check their geographic hierarchy
        from app.database import Dataset
        datasets_result = await db.execute(select(Dataset))
        datasets = datasets_result.scalars().all()
        
        for dataset in datasets:
            try:
                import json
                hierarchy = json.loads(dataset.geographic_hierarchy_json)
                
                # Check if this dataset is relevant for the location
                location_lower = location.lower()
                is_relevant = False
                
                # Check each level of the hierarchy
                for level, value in hierarchy.items():
                    if value and location_lower in value.lower():
                        is_relevant = True
                        break
                
                if is_relevant:
                    # Get data points from this dataset
                    if location.lower() == "san francisco":
                        # For city-level, get all neighborhoods in that city
                        # Query by metadata_json containing the city name
                        result = await db.execute(
                            select(DataPoint)
                            .where(DataPoint.dataset_id == dataset.id)
                            .where(DataPoint.metadata_json.like(f'%{location}%'))
                        )
                    else:
                        # For specific neighborhoods, query by exact name
                        result = await db.execute(
                            select(DataPoint)
                            .where(DataPoint.dataset_id == dataset.id)
                            .where(DataPoint.county == location)
                        )
                    
                    dataset_points = result.scalars().all()
                    data_points.extend(dataset_points)
                    print(f"📊 Found {len(dataset_points)} data points from dataset: {dataset.name}")
                    
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"⚠️ Error parsing hierarchy for dataset {dataset.name}: {e}")
                continue
        
        if not data_points:
            print(f"❌ No data points found for {location}")
            return {}
        
        print(f"📊 Total found {len(data_points)} data points for {location}")
        
        # Basic aggregation
        aggregated = {}
        for point in data_points:
            metric = point.metric_name
            if metric not in aggregated:
                aggregated[metric] = 0
            aggregated[metric] += point.metric_value
        
        return {
            'aggregated_data': aggregated,
            'llm_insights': ["Basic aggregation completed"],
            'confidence_level': 'low',
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'fallback_mode': True
        }
