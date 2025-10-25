from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.report_generator import ReportGeneratorService
from app.models import CityReport, SolutionDetailsRequest, SolutionDetailsResponse, DownloadReportRequest, DownloadReportResponse
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/generate-report", response_model=CityReport)
async def generate_report(
    location: str = Query(..., description="Location name to generate report for (e.g., 'Oakland', 'Alameda County')"),
    level: str = Query("county", description="Geographic level: county, city, neighborhood"),
    db: AsyncSession = Depends(get_db)
):
    """Generate or retrieve cached city report"""
    
    try:
        report_service = ReportGeneratorService()
        report = await report_service.generate_report(location, level, db)
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.get("/solution-details", response_model=SolutionDetailsResponse)
async def get_solution_details(
    problem_id: str = Query(..., description="Problem ID to get details for"),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed information about a specific problem and solution"""
    
    # For now, return mock data
    # In a real implementation, this would query the database for the specific problem
    return SolutionDetailsResponse(
        problem_id=problem_id,
        title="Elevated Youth Arrest Rates",
        description="18-29 age group shows 150% higher arrest rates than state average, indicating potential systemic issues with youth engagement and opportunity.",
        detailed_analysis="Analysis shows that youth arrest rates in this county are significantly above state and national averages. The data indicates a correlation between economic factors and arrest rates, suggesting that lack of opportunity may be driving criminal behavior among young adults.",
        supporting_data={
            "violent_arrests": 949,
            "property_arrests": 1593,
            "state_comparison": "+45%",
            "national_comparison": "+38%",
            "trend": "increasing"
        },
        solution={
            "title": "Youth Intervention Programs",
            "description": "Implement community-based diversion programs and mentorship initiatives targeting at-risk youth.",
            "estimated_cost": "$2.5M annually",
            "expected_impact": "20-30% reduction in youth recidivism within 2 years"
        },
        implementation_steps=[
            "Establish community advisory board",
            "Partner with local non-profits",
            "Develop mentorship matching system",
            "Create job training programs",
            "Implement early intervention protocols"
        ]
    )

@router.get("/generate-neighborhood-report")
async def generate_neighborhood_report(
    neighborhood: str = Query(..., description="Neighborhood name to analyze"),
    city: str = Query(..., description="City containing the neighborhood"),
    db: AsyncSession = Depends(get_db)
):
    """Generate report for a specific neighborhood with city and regional context"""
    
    try:
        from app.services.geographic_aggregator import GeographicAggregator
        geo_aggregator = GeographicAggregator()
        
        # Get hierarchical analysis
        analysis = await geo_aggregator.get_hierarchical_analysis(neighborhood, 'neighborhood', db)
        
        # Generate report using LLM
        from app.services.llm_service import LLMService
        llm_service = LLMService()
        
        # Format data for LLM
        llm_data = {
            'neighborhood': analysis.get('target_data', {}),
            'city_context': analysis.get('geographic_context', {}),
            'regional_trends': analysis.get('regional_trends', {}),
            'balanced_analysis': analysis.get('balanced_analysis', {})
        }
        
        problems = await llm_service.synthesize_problems(llm_data, f"{neighborhood}, {city}")
        
        return {
            'neighborhood': neighborhood,
            'city': city,
            'analysis': analysis,
            'problems': problems,
            'generated_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neighborhood analysis failed: {str(e)}")

@router.post("/download-report", response_model=DownloadReportResponse)
async def download_report(
    request: DownloadReportRequest,
    db: AsyncSession = Depends(get_db)
):
    """Export selected insights as downloadable report"""
    
    # For now, return mock data
    # In a real implementation, this would generate a PDF or JSON file
    report_id = f"report_{request.county.replace(' ', '_').lower()}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    return DownloadReportResponse(
        report_url=f"/api/reports/download/{report_id}",
        report_id=report_id,
        expires_at=datetime.utcnow() + timedelta(hours=24)
    )

