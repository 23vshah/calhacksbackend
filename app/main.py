from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from app.database import init_db
from app.routes import data, reports

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown
    pass

app = FastAPI(
    title="Theages Backend API",
    description="AI-driven city data analysis and report generation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/api", tags=["data"])
app.include_router(reports.router, prefix="/api", tags=["reports"])

@app.get("/")
async def root():
    return {"message": "Theages Backend API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/llm-info")
async def llm_info():
    """Get information about the current LLM provider"""
    from app.services.llm_service import LLMService
    try:
        llm_service = LLMService()
        return llm_service.get_provider_info()
    except Exception as e:
        return {"error": str(e)}

