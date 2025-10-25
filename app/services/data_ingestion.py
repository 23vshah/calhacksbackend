import pandas as pd
import json
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import Dataset, DataPoint
from app.services.llm_service import LLMService
from app.models import NormalizationProfile, GeographicLevel

class DataIngestionService:
    def __init__(self):
        self.llm_service = LLMService()
    
    async def process_csv(self, file_path: str, dataset_name: str, source_type: str, db: AsyncSession) -> Dict[str, Any]:
        """Process CSV file with Claude-powered normalization"""
        
        # Read CSV and get random sample for Claude analysis
        df = pd.read_csv(file_path)
        sample_size = min(100, len(df))
        # Random sampling to avoid bias from single type of rows
        sample_data = df.sample(n=sample_size, random_state=42).to_csv(index=False)
        
        # Get normalization profile from LLM
        profile = await self.llm_service.analyze_schema(sample_data)
        
        # Create dataset record
        dataset = Dataset(
            name=dataset_name,
            source_type=source_type,
            geographic_level=profile.geographic_level,
            geographic_hierarchy_json=profile.geographic_hierarchy.model_dump_json(),
            metadata_json=profile.model_dump_json()
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        
        # Process all rows using the profile - ULTRA-OPTIMIZED VERSION
        print(f"🚀 Processing {len(df)} rows with {len(profile.metrics)} metrics...")
        rows_processed = 0
        batch_size = 10000  # Even larger batch size
        commit_frequency = 5  # Commit every 5 batches to reduce DB overhead
        length = len(df)
        
        all_data_points = []  # Collect all data points before bulk insert
        
        for i in range(0, length, batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Process batch more efficiently using vectorized operations where possible
            for idx, row in batch.iterrows():
                # Extract geographic and time info once per row
                county = str(row[profile.geographic_column]).strip()
                
                # Handle time column - could be year (int) or date (string)
                year = None
                if profile.time_column in row:
                    time_value = row[profile.time_column]
                    if pd.notna(time_value):
                        try:
                            # Try to convert to int first (for year columns)
                            year = int(time_value)
                        except (ValueError, TypeError):
                            # If it's a date string, extract year
                            try:
                                from datetime import datetime
                                if isinstance(time_value, str):
                                    # Handle different date formats
                                    if ' ' in time_value:  # datetime format
                                        year = datetime.strptime(time_value.split(' ')[0], '%Y-%m-%d').year
                                    else:  # date format
                                        year = datetime.strptime(time_value, '%Y-%m-%d').year
                                else:
                                    year = None
                            except (ValueError, TypeError):
                                year = None
                
                # Pre-compute metadata once per row
                metadata = {}
                for dim in profile.dimensions:
                    if isinstance(dim, dict) and 'column' in dim:
                        column_name = dim['column']
                        if column_name in row:
                            metadata[column_name.lower()] = str(row[column_name])
                    elif isinstance(dim, str) and dim in row:
                        metadata[dim.lower()] = str(row[dim])
                
                metadata_json = json.dumps(metadata) if metadata else None
                
                # Process all metrics for this row
                for metric in profile.metrics:
                    if metric["column"] in row:
                        value = row[metric["column"]]
                        # Fast numeric check
                        if pd.notna(value):
                            try:
                                numeric_value = float(value)
                                if not pd.isna(numeric_value):  # Skip NaN values
                                    data_point = DataPoint(
                                        dataset_id=dataset.id,
                                        county=county,
                                        year=year,
                                        category=profile.data_type,
                                        metric_name=metric["normalized_name"],
                                        metric_value=numeric_value,
                                        metadata_json=metadata_json
                                    )
                                    all_data_points.append(data_point)
                            except (ValueError, TypeError):
                                continue
            
            # Progress update every 10 batches only
            if (i // batch_size) % 10 == 0:
                progress = (i / length) * 100
                print(f"📈 Progress: {progress:.1f}%")
            
            # Bulk insert every few batches to reduce DB overhead
            if len(all_data_points) >= batch_size * commit_frequency:
                db.add_all(all_data_points)
                await db.commit()
                rows_processed += len(all_data_points)
                all_data_points = []  # Clear the list
        
        # Insert remaining data points
        if all_data_points:
            db.add_all(all_data_points)
            await db.commit()
            rows_processed += len(all_data_points)
        
        # Calculate weights for new metrics
        unique_metrics = list(set([metric["normalized_name"] for metric in profile.metrics]))
        await self._update_weights(unique_metrics, db)
        
        return {
            "success": True,
            "dataset_id": dataset.id,
            "rows_processed": rows_processed,
            "message": f"Successfully processed {rows_processed} data points"
        }
    
    async def _update_weights(self, metrics: List[str], db: AsyncSession):
        """Update weights for new metrics using LLM"""
        try:
            weights = await self.llm_service.calculate_weights(metrics)
            
            # Store weights in database
            from app.database import DataWeight
            
            for weight in weights:
                # Check if weight already exists
                from sqlalchemy import select
                result = await db.execute(
                    select(DataWeight).where(DataWeight.metric_name == weight.metric_name)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing weight
                    existing.weight = weight.weight
                    existing.reasoning = weight.reasoning
                    existing.last_updated = datetime.utcnow()
                else:
                    # Create new weight
                    new_weight = DataWeight(
                        metric_name=weight.metric_name,
                        category="crime",  # Default category
                        weight=weight.weight,
                        reasoning=weight.reasoning
                    )
                    db.add(new_weight)
            
            await db.commit()
            print(f"✅ Updated weights for {len(weights)} metrics")
            
        except Exception as e:
            print(f"Weight calculation failed: {e}")
            # Continue without weights for now

