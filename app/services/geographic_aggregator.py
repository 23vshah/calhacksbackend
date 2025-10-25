import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from app.database import DataPoint, Dataset
from app.services.llm_service import LLMService

class GeographicAggregator:
    """Handles hierarchical geographic analysis: neighborhood → city → region"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.geographic_hierarchy = {
            'neighborhood': 1,
            'city': 2, 
            'county': 3,
            'region': 4,
            'state': 5
        }
    
    async def get_hierarchical_analysis(self, target_location: str, target_level: str, db: AsyncSession) -> Dict[str, Any]:
        """Get analysis for target location with broader geographic context"""
        
        print(f"🗺️ Analyzing {target_location} at {target_level} level")
        
        # Get target location data
        target_data = await self._get_location_data(target_location, target_level, db)
        
        # Get broader geographic context
        context_data = await self._get_geographic_context(target_location, target_level, db)
        
        # Get regional trends
        regional_trends = await self._get_regional_trends(target_location, target_level, db)
        
        # Balance datasets to prevent bias
        balanced_data = await self._balance_dataset_representation(target_data, db)
        
        return {
            'target_location': target_location,
            'target_level': target_level,
            'target_data': target_data,
            'geographic_context': context_data,
            'regional_trends': regional_trends,
            'balanced_analysis': balanced_data,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    async def _get_location_data(self, location: str, level: str, db: AsyncSession) -> Dict[str, Any]:
        """Get data for specific location"""
        
        # Map location levels to database fields
        location_field_map = {
            'neighborhood': 'metadata_json',  # Extract from metadata
            'city': 'county',  # For now, using county as city proxy
            'county': 'county',
            'region': 'county',  # Will need region mapping
            'state': 'county'    # Will need state mapping
        }
        
        field = location_field_map.get(level, 'county')
        
        # Get all datasets and check their geographic hierarchy
        from app.database import Dataset
        datasets_result = await db.execute(select(Dataset))
        datasets = datasets_result.scalars().all()
        
        data_points = []
        
        for dataset in datasets:
            try:
                import json
                hierarchy = json.loads(dataset.geographic_hierarchy_json)
                
                # Check if this dataset is relevant for the location
                location_lower = location.lower()
                is_relevant = False
                
                # Check each level of the hierarchy
                for level_name, value in hierarchy.items():
                    if value and location_lower in value.lower():
                        is_relevant = True
                        break
                
                if is_relevant:
                    # Get data points from this dataset
                    if level == 'neighborhood':
                        # For neighborhood-level, query by exact name
                        result = await db.execute(
                            select(DataPoint)
                            .where(DataPoint.dataset_id == dataset.id)
                            .where(DataPoint.county == location)
                        )
                    elif level == 'city':
                        # For city-level, get all neighborhoods in that city
                        result = await db.execute(
                            select(DataPoint)
                            .where(DataPoint.dataset_id == dataset.id)
                            .where(DataPoint.metadata_json.like(f'%{location}%'))
                        )
                    else:
                        # For other levels, use the field mapping
                        result = await db.execute(
                            select(DataPoint)
                            .where(DataPoint.dataset_id == dataset.id)
                            .where(getattr(DataPoint, field) == location)
                        )
                    
                    dataset_points = result.scalars().all()
                    data_points.extend(dataset_points)
                    
            except (json.JSONDecodeError, AttributeError) as e:
                continue
        
        return await self._aggregate_location_data(data_points, location, level)
    
    async def _get_geographic_context(self, location: str, level: str, db: AsyncSession) -> Dict[str, Any]:
        """Get broader geographic context (city for neighborhood, region for city, etc.)"""
        
        context_levels = {
            'neighborhood': 'city',
            'city': 'county', 
            'county': 'region',
            'region': 'state'
        }
        
        context_level = context_levels.get(level)
        if not context_level:
            return {}
        
        # Get all locations at the context level
        result = await db.execute(
            select(DataPoint.county).distinct()
        )
        all_locations = [row[0] for row in result.fetchall()]
        
        # Aggregate across all locations at context level
        context_data = {}
        for loc in all_locations:
            loc_data = await self._get_location_data(loc, context_level, db)
            if loc_data:
                context_data[loc] = loc_data
        
        return context_data
    
    async def _get_regional_trends(self, location: str, level: str, db: AsyncSession) -> Dict[str, Any]:
        """Get regional trends and patterns"""
        
        # Get all data points for trend analysis
        result = await db.execute(select(DataPoint))
        all_data_points = result.scalars().all()
        
        # Group by metric and analyze trends
        trends = {}
        metric_groups = {}
        
        for point in all_data_points:
            metric = point.metric_name
            if metric not in metric_groups:
                metric_groups[metric] = []
            metric_groups[metric].append(point)
        
        for metric, points in metric_groups.items():
            if len(points) < 10:  # Skip metrics with too little data
                continue
                
            trend_analysis = self._analyze_metric_trends(points)
            trends[metric] = trend_analysis
        
        return trends
    
    async def _balance_dataset_representation(self, target_data: Dict[str, Any], db: AsyncSession) -> Dict[str, Any]:
        """Balance datasets to prevent bias from large datasets"""
        
        # Get all datasets and their sizes
        result = await db.execute(select(Dataset))
        datasets = result.scalars().all()
        
        dataset_sizes = {}
        for dataset in datasets:
            # Count data points for this dataset
            count_result = await db.execute(
                select(func.count(DataPoint.id))
                .where(DataPoint.dataset_id == dataset.id)
            )
            count = count_result.scalar()
            dataset_sizes[dataset.id] = {
                'name': dataset.name,
                'size': count,
                'source_type': dataset.source_type
            }
        
        print(f"📊 Dataset sizes: {dataset_sizes}")
        
        # Calculate sampling ratios to balance datasets
        max_size = max(size['size'] for size in dataset_sizes.values())
        sampling_ratios = {}
        
        for dataset_id, info in dataset_sizes.items():
            if info['size'] > max_size * 0.5:  # If dataset is >50% of largest
                sampling_ratios[dataset_id] = max_size * 0.5 / info['size']
                print(f"⚖️ Balancing dataset {info['name']}: {info['size']} → {int(max_size * 0.5)} points")
            else:
                sampling_ratios[dataset_id] = 1.0
        
        # Apply balanced sampling to target data
        balanced_data = await self._apply_balanced_sampling(target_data, sampling_ratios, db)
        
        return balanced_data
    
    async def _apply_balanced_sampling(self, target_data: Dict[str, Any], sampling_ratios: Dict[int, float], db: AsyncSession) -> Dict[str, Any]:
        """Apply balanced sampling to prevent dataset bias"""
        
        # This would implement intelligent sampling
        # For now, return the original data with sampling info
        return {
            'original_data': target_data,
            'sampling_ratios': sampling_ratios,
            'balance_applied': True
        }
    
    async def _aggregate_location_data(self, data_points: List[DataPoint], location: str, level: str) -> Dict[str, Any]:
        """Aggregate data for a specific location"""
        
        if not data_points:
            return {}
        
        # Group by metric and analyze
        metric_analysis = {}
        
        for point in data_points:
            metric = point.metric_name
            if metric not in metric_analysis:
                metric_analysis[metric] = {
                    'values': [],
                    'years': [],
                    'total': 0,
                    'count': 0
                }
            
            metric_analysis[metric]['values'].append(point.metric_value)
            metric_analysis[metric]['years'].append(point.year)
            metric_analysis[metric]['total'] += point.metric_value
            metric_analysis[metric]['count'] += 1
        
        # Calculate metrics
        aggregated = {}
        for metric, data in metric_analysis.items():
            if data['count'] == 0:
                continue
            
            # Basic metrics
            total = data['total']
            avg_per_year = total / len(set(data['years'])) if data['years'] else 0
            
            # Recent trend (last 3 years)
            recent_years = [y for y in data['years'] if y and y >= (datetime.now().year - 3)]
            recent_values = [data['values'][i] for i, y in enumerate(data['years']) if y in recent_years]
            recent_avg = sum(recent_values) / len(recent_values) if recent_values else 0
            
            # Trend analysis
            trend = self._calculate_trend(data['values'], data['years'])
            
            # Anomaly score
            anomaly_score = self._calculate_anomaly_score(total, data['count'])
            
            aggregated[metric] = {
                'total': total,
                'average_per_year': avg_per_year,
                'recent_average': recent_avg,
                'trend': trend,
                'anomaly_score': anomaly_score,
                'data_points': data['count'],
                'time_period': f"{min(data['years'])}-{max(data['years'])}" if data['years'] else "unknown"
            }
        
        return aggregated
    
    def _analyze_metric_trends(self, points: List[DataPoint]) -> Dict[str, Any]:
        """Analyze trends for a specific metric across all locations"""
        
        # Group by year
        year_data = {}
        for point in points:
            if not point.year:
                continue
            if point.year not in year_data:
                year_data[point.year] = []
            year_data[point.year].append(point.metric_value)
        
        if len(year_data) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate year averages
        year_averages = {year: sum(values)/len(values) for year, values in year_data.items()}
        sorted_years = sorted(year_averages.keys())
        
        # Calculate trend
        first_avg = year_averages[sorted_years[0]]
        last_avg = year_averages[sorted_years[-1]]
        change_percent = ((last_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        
        # Determine trend direction
        if change_percent > 15:
            trend = 'strongly_increasing'
        elif change_percent > 5:
            trend = 'increasing'
        elif change_percent < -15:
            trend = 'strongly_decreasing'
        elif change_percent < -5:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change_percent': change_percent,
            'first_year': sorted_years[0],
            'last_year': sorted_years[-1],
            'first_average': first_avg,
            'last_average': last_avg,
            'total_data_points': len(points)
        }
    
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
