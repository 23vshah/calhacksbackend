from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

class GeographicLevel(str, Enum):
    """Geographic levels for dataset classification"""
    STATE = "state"
    REGION = "region"
    COUNTY = "county"
    CITY = "city"
    NEIGHBORHOOD = "neighborhood"

# Request/Response Models

class DatasetCreate(BaseModel):
    name: str
    source_type: str

class DatasetResponse(BaseModel):
    id: int
    name: str
    source_type: str
    upload_date: datetime
    metadata_json: Optional[str] = None

class DataIngestionResponse(BaseModel):
    success: bool
    dataset_id: int
    rows_processed: int
    message: str

class ReportRequest(BaseModel):
    county: str

class ProblemSolution(BaseModel):
    title: str
    description: str
    estimated_cost: Optional[str] = None
    expected_impact: Optional[str] = None

class Problem(BaseModel):
    id: str
    title: str
    severity: str
    description: str
    metrics: Dict[str, Any]
    solution: ProblemSolution

class ReportSummary(BaseModel):
    population: Optional[int] = None
    data_sources: List[str]
    last_data_update: Optional[str] = None

class CityReport(BaseModel):
    county: str
    generated_at: datetime
    cached: bool
    summary: ReportSummary
    problems: List[Problem]

class SolutionDetailsRequest(BaseModel):
    problem_id: str

class SolutionDetailsResponse(BaseModel):
    problem_id: str
    title: str
    description: str
    detailed_analysis: str
    supporting_data: Dict[str, Any]
    solution: ProblemSolution
    implementation_steps: List[str]

class DownloadReportRequest(BaseModel):
    county: str
    selected_problems: List[str]

class DownloadReportResponse(BaseModel):
    report_url: str
    report_id: str
    expires_at: datetime

# Internal Models for Data Processing

class GeographicHierarchy(BaseModel):
    neighborhood: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    region: Optional[str] = None

class NormalizationProfile(BaseModel):
    data_type: str
    geographic_level: str
    geographic_hierarchy: GeographicHierarchy
    time_granularity: str
    metrics: List[Dict[str, str]]
    dimensions: List[Union[str, Dict[str, Any]]]  # Can handle both strings and objects
    geographic_column: str
    time_column: str

class MetricWeight(BaseModel):
    metric_name: str
    weight: float
    reasoning: str

