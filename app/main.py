from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from app.database import init_db
from app.routes import data, reports, goals, agents, knowledge_graph

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
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        frontend_url,
        "http://localhost:3000",  # React default
        "http://localhost:5173",  # Vite default
        "http://localhost:8080",  # Alternative dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/api", tags=["data"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(goals.router, prefix="/api", tags=["goals"])
app.include_router(agents.router, prefix="/api", tags=["agents"])
app.include_router(knowledge_graph.router, prefix="/api", tags=["knowledge-graph"])

@app.get("/")
async def root():
    return {
        "message": "Theages Backend API", 
        "status": "running",
        "cors_origins": [
            "http://localhost:3000",
            "http://localhost:5173", 
            "http://localhost:8080"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "cors_enabled": True}

@app.get("/llm-info")
async def llm_info():
    """Get information about the current LLM provider"""
    from app.services.llm_service import LLMService
    try:
        llm_service = LLMService()
        return llm_service.get_provider_info()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/test-cors")
async def test_cors():
    """Test endpoint to verify CORS is working"""
    return {
        "message": "CORS test successful",
        "timestamp": "2024-01-01T00:00:00Z",
        "frontend_should_see_this": True
    }

@app.get("/api/routes")
async def list_routes():
    """List all available API routes for debugging"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'unnamed')
            })
    return {
        "available_routes": routes,
        "total_routes": len(routes)
    }

