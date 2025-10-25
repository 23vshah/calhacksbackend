# Theages Backend API

AI-driven city data analysis and report generation system.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

3. Run the application:
```bash
python run.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Data Ingestion
- `POST /api/ingest-data` - Upload and process CSV datasets
- `GET /api/datasets` - List all ingested datasets
- `DELETE /api/datasets/{id}` - Delete a dataset

### Report Generation
- `GET /api/generate-report?county={name}` - Generate city report
- `GET /api/solution-details?id={problem_id}` - Get detailed solution info
- `POST /api/download-report` - Export report

## Testing with Arrest Data

1. Upload the `OnlineArrestData1980-2024.csv` file via the `/api/ingest-data` endpoint
2. Generate a report for "Alameda County" via `/api/generate-report?county=Alameda County`

## Architecture

- **FastAPI** - Web framework
- **SQLAlchemy** - Database ORM with SQLite
- **Claude API** - AI-powered data analysis and report synthesis
- **Pandas** - Data processing

