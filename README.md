# Westgate Backend API

An AI-driven city data analysis and intelligence platform that processes urban data from multiple sources to generate actionable insights and reports.

## Overview

Westgate Backend is a FastAPI-based service that aggregates and analyzes city data from various sources including SF311 requests, Reddit discussions, crime statistics, and neighborhood indicators. It uses intelligent agents and LLM services to process data and generate comprehensive reports for urban planning and civic engagement.

## Architecture

```
backend/
├── app/
│   ├── routes/           # API endpoints
│   │   ├── agents.py     # Agent orchestration & triggers
│   │   ├── data.py       # Data ingestion & management
│   │   ├── reports.py    # Report generation
│   │   ├── goals.py      # Goal tracking & management
│   │   └── knowledge_graph.py  # Knowledge graph operations
│   ├── services/         # Core business logic
│   │   ├── agents/       # Specialized AI agents
│   │   │   ├── sf311_agent.py      # SF311 data analysis
│   │   │   ├── reddit_agent.py     # Social media insights
│   │   │   └── knowledge_graph_agent.py  # Graph operations
│   │   ├── agent_framework.py     # Agent orchestration
│   │   ├── llm_service.py         # LLM integration
│   │   ├── report_generator.py    # Report creation
│   │   └── data_ingestion.py      # Data processing
│   ├── models.py         # Database models
│   └── database.py      # Database configuration
├── parsers/              # Data source parsers
│   ├── 311parser/        # SF311 API integration
│   └── redditParser/     # Reddit data extraction
└── requirements.txt      # Python dependencies
```

## Key Features

### 🤖 Intelligent Agents
- **SF311 Agent**: Analyzes city service requests for patterns and insights
- **Reddit Agent**: Extracts community sentiment and local issues
- **Knowledge Graph Agent**: Builds relationships between data points

### 📊 Intelligent Data Processing
- **Multi-source Ingestion**: Seamlessly processes CSV files, APIs, social media feeds, and real-time data streams
- **Smart Aggregation**: AI-powered data normalization across different formats, schemas, and geographic levels
- **Universal Architecture**: Designed to handle any city, county, or state data with automatic schema detection and mapping
- **Geographic Intelligence**: Advanced spatial analysis with neighborhood, district, and regional mapping
- **Data Harmonization**: Intelligent merging of disparate data sources into unified, queryable datasets

### 📈 Report Generation
- Automated city reports with AI-generated insights
- PDF export functionality
- Customizable report templates

### 🎯 Goal Management
- Track civic goals and initiatives
- Progress monitoring and analytics
- Integration with report generation

## Tech Stack

- **Framework**: FastAPI with async/await
- **Database**: SQLite with SQLAlchemy ORM
- **AI/ML**: OpenAI GPT, Anthropic Claude, Sentence Transformers
- **Data Processing**: Pandas, NumPy, FAISS for vector search
- **APIs**: SF311, Reddit, custom data sources

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Add your API keys (OpenAI, Anthropic, etc.)
   ```

3. **Run the server**:
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Access the API**:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## API Endpoints

- `/api/agents/*` - Agent operations and data triggers
- `/api/reports/*` - Report generation and management
- `/api/data/*` - Data ingestion and querying
- `/api/goals/*` - Goal tracking and analytics
- `/api/knowledge-graph/*` - Knowledge graph operations

## Data Sources & Architecture

### Supported Data Types
- **SF311**: San Francisco service requests and complaints
- **Reddit**: Community discussions and local issues  
- **Crime Data**: Historical crime statistics and trends
- **Neighborhood Indicators**: Demographics and quality metrics
- **Permit Data**: Building and development permits
- **Custom Datasets**: Any CSV or API data from cities, counties, or states

### Intelligent Data Pipeline
The platform features a sophisticated data ingestion architecture that:
- **Auto-detects** data schemas and formats across different sources
- **Normalizes** geographic references (addresses, coordinates, neighborhoods)
- **Harmonizes** temporal data (dates, time zones, reporting periods)
- **Aggregates** data at multiple geographic levels (block, neighborhood, district, city, county)
- **Validates** data quality and completeness automatically
- **Indexes** data for fast querying and analysis

This universal architecture enables rapid integration of new data sources from any municipality, transforming raw urban data into actionable intelligence for better city planning and civic engagement.