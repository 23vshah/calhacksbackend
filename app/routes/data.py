from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.data_ingestion import DataIngestionService
from app.models import DatasetResponse, DataIngestionResponse
from sqlalchemy import select
from app.database import Dataset
import tempfile
import os

router = APIRouter()

@router.post("/ingest-data", response_model=DataIngestionResponse)
async def ingest_data(
    file: UploadFile = File(...),
    dataset_name: str = None,
    source_type: str = None,
    db: AsyncSession = Depends(get_db)
):
    """Upload and process a dataset file"""
    
    if not dataset_name:
        dataset_name = file.filename or "uploaded_dataset"
    
    if not source_type:
        source_type = "unknown"
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Process the file
        ingestion_service = DataIngestionService()
        result = await ingestion_service.process_csv(
            tmp_file_path, 
            dataset_name, 
            source_type, 
            db
        )
        
        return DataIngestionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data ingestion failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@router.get("/datasets", response_model=list[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    """List all ingested datasets"""
    
    result = await db.execute(select(Dataset))
    datasets = result.scalars().all()
    
    return [DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        source_type=dataset.source_type,
        upload_date=dataset.upload_date,
        metadata_json=dataset.metadata_json
    ) for dataset in datasets]

@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a dataset and all related data points"""
    
    # Find dataset
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Delete related data points first
    from app.database import DataPoint
    from sqlalchemy import delete
    await db.execute(
        delete(DataPoint).where(DataPoint.dataset_id == dataset_id)
    )
    
    # Delete dataset
    from sqlalchemy import delete
    await db.execute(
        delete(Dataset).where(Dataset.id == dataset_id)
    )
    await db.commit()
    
    return {"message": f"Dataset {dataset_id} deleted successfully"}

