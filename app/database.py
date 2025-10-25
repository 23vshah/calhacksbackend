from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import os
from datetime import datetime
from app.models import GeographicLevel

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./theages.db")

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Create async session
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Base class for models
Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    source_type = Column(String, nullable=False)  # arrest, demographics, crime, 311, etc.
    geographic_level = Column(Enum(GeographicLevel), nullable=False)
    geographic_hierarchy_json = Column(Text)  # Geographic hierarchy as JSON
    upload_date = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(Text)  # Normalization profile from Claude
    
    # Relationship
    data_points = relationship("DataPoint", back_populates="dataset")

class DataPoint(Base):
    __tablename__ = "data_points"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    county = Column(String, nullable=False, index=True)
    year = Column(Integer, index=True)
    category = Column(String, nullable=False)  # crime, demographics, economic, etc.
    metric_name = Column(String, nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metadata_json = Column(Text)  # Raw dimensions like gender, race, age_group
    
    # Relationship
    dataset = relationship("Dataset", back_populates="data_points")

class DataWeight(Base):
    __tablename__ = "data_weights"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String, nullable=False, unique=True, index=True)
    category = Column(String, nullable=False)
    weight = Column(Float, nullable=False)  # 0-1 importance score
    reasoning = Column(Text)
    last_updated = Column(DateTime, default=datetime.utcnow)

class GeneratedReport(Base):
    __tablename__ = "generated_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    county = Column(String, nullable=False, index=True)
    generated_at = Column(DateTime, default=datetime.utcnow)
    report_json = Column(Text, nullable=False)
    data_snapshot_hash = Column(String, index=True)

# Database initialization
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Dependency to get database session
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

