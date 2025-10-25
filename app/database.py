from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, ForeignKey, Enum, Index
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
    
    # Relationships
    data_points = relationship("DataPoint", back_populates="dataset")
    quality_scores = relationship("DataQualityScore", back_populates="dataset")

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

class DataCorrelation(Base):
    __tablename__ = "data_correlations"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_1 = Column(String, nullable=False, index=True)
    metric_2 = Column(String, nullable=False, index=True)
    correlation_strength = Column(Float, nullable=False)  # -1 to 1
    correlation_type = Column(String, nullable=False)  # positive, negative, none
    confidence_score = Column(Float, nullable=False)  # 0-1
    sample_size = Column(Integer, nullable=False)
    geographic_scope = Column(String, nullable=False)  # county, city, state, national
    detected_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(Text)  # Additional analysis details

class DataQualityScore(Base):
    __tablename__ = "data_quality_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    metric_name = Column(String, nullable=False, index=True)
    completeness_score = Column(Float, nullable=False)  # 0-1
    accuracy_score = Column(Float, nullable=False)  # 0-1
    consistency_score = Column(Float, nullable=False)  # 0-1
    timeliness_score = Column(Float, nullable=False)  # 0-1
    overall_score = Column(Float, nullable=False)  # 0-1
    validation_notes = Column(Text)
    last_validated = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    dataset = relationship("Dataset", back_populates="quality_scores")

class DataVisualization(Base):
    __tablename__ = "data_visualizations"
    
    id = Column(Integer, primary_key=True, index=True)
    visualization_type = Column(String, nullable=False)  # chart, map, table, etc.
    title = Column(String, nullable=False)
    description = Column(Text)
    data_query_json = Column(Text, nullable=False)  # Query parameters
    chart_config_json = Column(Text)  # Chart.js or similar config
    created_at = Column(DateTime, default=datetime.utcnow)
    is_public = Column(String, default="true")  # true/false as string for SQLite

class CityGoal(Base):
    __tablename__ = "city_goals"
    
    id = Column(Integer, primary_key=True, index=True)
    city_name = Column(String, nullable=False, index=True)
    goal_title = Column(String, nullable=False)
    goal_description = Column(Text, nullable=False)
    target_metric = Column(String, nullable=False)  # e.g., "Reduce homelessness by 20%"
    target_value = Column(Float)
    target_unit = Column(String)  # percentage, count, etc.
    priority_level = Column(String, nullable=False)  # high, medium, low
    deadline = Column(DateTime)
    status = Column(String, default="active")  # active, completed, paused
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    embedding_vector = Column(Text)  # JSON string of vector embeddings
    metadata_json = Column(Text)  # Additional goal context

class PolicyDocument(Base):
    __tablename__ = "policy_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, nullable=False)  # HUD, Urban Institute, OECD, etc.
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    document_type = Column(String, nullable=False)  # legislation, research, best_practice
    geographic_scope = Column(String, nullable=False)  # national, state, local
    topic_tags = Column(Text)  # JSON array of topics
    embedding_vector = Column(Text)  # JSON string of vector embeddings
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(String, default="true")

class GoalRecommendation(Base):
    __tablename__ = "goal_recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    city_goal_id = Column(Integer, ForeignKey("city_goals.id"))
    problem_id = Column(String, nullable=False)  # Reference to problem from report
    policy_document_id = Column(Integer, ForeignKey("policy_documents.id"))
    similarity_score = Column(Float, nullable=False)  # 0-1 cosine similarity
    recommendation_text = Column(Text, nullable=False)
    implementation_steps = Column(Text)  # JSON array of steps
    estimated_impact = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    city_goal = relationship("CityGoal")
    policy_document = relationship("PolicyDocument")

# Agent System Tables
class CommunityIssue(Base):
    __tablename__ = "community_issues"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(String, nullable=False)  # high, medium, low
    source = Column(String, nullable=False)  # reddit, 311, news
    source_id = Column(String)  # Original ID from source
    location_id = Column(Integer, ForeignKey("locations.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(Text)  # Additional source data
    embedding_vector = Column(Text)  # For similarity search
    
    # Relationships
    location = relationship("Location")
    relationships = relationship("IssueRelationship", foreign_keys="IssueRelationship.issue_id_1")

class Location(Base):
    __tablename__ = "locations"
    
    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    address = Column(String)
    neighborhood = Column(String, index=True)
    city = Column(String, nullable=False, index=True)
    county = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    issues = relationship("CommunityIssue", back_populates="location")

class IssueRelationship(Base):
    __tablename__ = "issue_relationships"
    
    id = Column(Integer, primary_key=True, index=True)
    issue_id_1 = Column(Integer, ForeignKey("community_issues.id"))
    issue_id_2 = Column(Integer, ForeignKey("community_issues.id"))
    relationship_type = Column(String, nullable=False)  # similar, related, geographic, temporal
    strength = Column(Float, nullable=False)  # 0-1 similarity score
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    issue_1 = relationship("CommunityIssue", foreign_keys=[issue_id_1])
    issue_2 = relationship("CommunityIssue", foreign_keys=[issue_id_2])

class IssueCluster(Base):
    __tablename__ = "issue_clusters"
    
    id = Column(Integer, primary_key=True, index=True)
    cluster_name = Column(String, nullable=False)
    representative_issue_id = Column(Integer, ForeignKey("community_issues.id"))
    pattern_description = Column(Text)
    issue_count = Column(Integer, default=0)
    severity_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    representative_issue = relationship("CommunityIssue")

class AgentMemory(Base):
    __tablename__ = "agent_memory"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, nullable=False, index=True)
    memory_type = Column(String, nullable=False)  # successful_search, failed_pattern, insight
    data_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Index for fast lookups
    __table_args__ = (Index('ix_agent_memory_agent_type', 'agent_id', 'memory_type'),)

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

