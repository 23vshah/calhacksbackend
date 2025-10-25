<!-- 03bc17d4-2240-48cf-b1f7-2cef38c98bfd 61db3ae6-6829-40ef-93b5-7fbefa819cf7 -->
# Theages Backend System - Checkpoint 1

## Architecture Overview

The system will have 4 main components:

1. **Data Ingestion API** - Accept and process diverse datasets
2. **LLM-Powered Data Normalization** - Use Claude to understand and normalize heterogeneous data
3. **Database Layer** - SQLite with efficient querying for report generation
4. **Report Generation API** - Synthesize insights and generate city reports

## Data Flow

```
Raw Dataset (CSV/JSON) 
  → /ingest-data API 
  → Claude analyzes schema and content 
  → Normalized data stored in DB with metadata 
  → /generate-report queries normalized data 
  → Claude synthesizes problems + solutions 
  → Return structured report JSON
```

## Database Schema Design

### Core Tables

**1. `datasets`** - Track all ingested datasets

- id, name, source_type (arrest, demographics, crime, 311, etc.), upload_date, metadata_json

**2. `data_points`** - Normalized data storage (EAV-style for flexibility)

- id, dataset_id, county, year, category, metric_name, metric_value, raw_data_json

**3. `data_weights`** - LLM-generated importance weights

- id, metric_name, category, weight (0-1), reasoning, last_updated

**4. `generated_reports`** - Cached reports

- id, county, generated_at, report_json, data_snapshot_hash

### Why This Schema?

- **Flexibility**: EAV pattern lets us store arrest data, demographics, 311 complaints without schema changes
- **County-based**: All data normalized to county level for consistent aggregation
- **Metadata preservation**: Keep raw_data_json for traceability
- **Weight system**: Claude determines which metrics matter most for problem detection

## Key Implementation Files

### 1. **app/main.py** - FastAPI application entry point

- CORS middleware for frontend
- Route registration
- Startup event to initialize DB

### 2. **app/database.py** - SQLAlchemy setup

- SQLite connection with async support
- Table models matching schema above
- Database initialization function

### 3. **app/models.py** - Pydantic models

- Request/response schemas
- Data validation models

### 4. **app/services/claude_service.py** - Claude API integration

- Wrapper for Anthropic API calls
- Schema analysis prompt templates
- Report synthesis prompts
- Weight calculation prompts

### 5. **app/services/data_ingestion.py** - Data processing logic

- CSV/JSON parser
- Claude-powered schema detection: "Analyze this dataset and tell me: what does each column measure? What geographic level? What time period?"
- Data normalization into `data_points` table
- Weight calculation and storage

### 6. **app/services/report_generator.py** - Report creation

- Query aggregated data by county
- Apply weights to identify top issues
- Claude synthesis for problems + solutions
- Caching logic (check hash, return cached if fresh)

### 7. **app/routes/data.py** - Data ingestion endpoints

```
POST /api/ingest-data
 - Accept file upload (CSV/JSON)
 - Validate and process
 - Return ingestion status

GET /api/datasets
 - List all ingested datasets

DELETE /api/datasets/{id}
 - Remove dataset and related data points
```

### 8. **app/routes/reports.py** - Report generation endpoints

```
GET /api/generate-report?county={name}
 - Generate or retrieve cached report
 - Return structured JSON with problems/solutions

GET /api/solution-details?id={problem_id}
 - Detailed info about specific problem

POST /api/download-report
 - Export selected insights as JSON/PDF
```

### 9. **app/utils/normalization.py** - Helper functions

- Geographic standardization (county names)
- Date/time normalization
- Unit conversions

### 10. **.env.example** - Configuration template

```
ANTHROPIC_API_KEY=your_key_here
DATABASE_URL=sqlite:///./theages.db
FRONTEND_URL=http://localhost:3000
```

## LLM-Powered Normalization Strategy

### Step 1: Schema Analysis (ONE-TIME per dataset upload)

**Claude analyzes ONLY a sample (first 10-50 rows) to create normalization profile**

**Prompt to Claude:**

```
You are a data analyst. Analyze this dataset sample:
[First 10-50 rows of CSV/JSON]

Respond in JSON format:
{
  "data_type": "crime" | "demographics" | "economic" | "housing" | "311" | "other",
  "geographic_level": "county" | "city" | "zip" | "address",
  "time_granularity": "yearly" | "monthly" | "daily" | "none",
  "metrics": [
    {"column": "VIOLENT", "normalized_name": "violent_arrests", "description": "Violent crime arrests", "unit": "count"},
    {"column": "PROPERTY", "normalized_name": "property_arrests", "description": "Property crime arrests", "unit": "count"},
    ...
  ],
  "dimensions": ["GENDER", "RACE", "AGE_GROUP"],  // Columns for grouping/filtering
  "geographic_column": "COUNTY",
  "time_column": "YEAR"
}
```

**This normalization profile is then applied to ALL 100,000+ rows without additional LLM calls.**

**Processing flow:**

1. Upload CSV (100k rows)
2. Claude analyzes sample → returns profile JSON
3. Profile stored in `datasets` table as `metadata_json`
4. Python loop processes ALL rows using the profile (fast, deterministic)
5. Each row creates multiple `data_points` entries (one per metric)

**Example for arrest data:**

```python
# For each row in 100k rows:
row = {"YEAR": 1980, "COUNTY": "Alameda County", "VIOLENT": 505, "PROPERTY": 1351, ...}

# Using profile, create data_points:
INSERT INTO data_points (dataset_id, county, year, metric_name, metric_value, metadata)
VALUES 
  (1, "Alameda County", 1980, "violent_arrests", 505, '{"gender":"Male","race":"Black","age":"Under 18"}'),
  (1, "Alameda County", 1980, "property_arrests", 1351, '{"gender":"Male","race":"Black","age":"Under 18"}'),
  ...
```

### Step 2: Metric Weighting (ONE-TIME after ingestion, or when new datasets added)

**Prompt to Claude:**

```
You are an urban planning expert. Given these metrics across various datasets:
- Arrest counts (violent, property, drug)
- Population demographics (age, race)
- 311 complaint types
- Housing vacancy rates
- Unemployment rates

Assign importance weights (0-1) for detecting community problems:
- Economic decline
- Public safety issues
- Housing crisis
- Infrastructure gaps

Return JSON: {"metric_name": weight, ...} with reasoning.
```

Store these weights in `data_weights` table.

### Step 3: Problem Detection (during report generation)

**Algorithm:**

1. Query all data_points for county
2. Calculate weighted anomaly scores (compare to state average, historical trends)
3. Identify top 3 outliers (high crime + low income + high vacancy = economic distress)
4. Send to Claude for synthesis

**Prompt to Claude:**

```
Based on this county data:
- Violent arrests: 2,149 (150% of state avg)
- Unemployment: 12% (state avg: 7%)
- Housing vacancy: 18% (state avg: 10%)

Generate:
1. Problem title (concise)
2. Problem description (2 sentences)
3. Actionable solution (policy recommendation)
4. Expected impact

Format as JSON.
```

## API Response Examples

### `/api/generate-report?county=Alameda County`

```json
{
  "county": "Alameda County",
  "generated_at": "2025-10-24T10:30:00Z",
  "cached": false,
  "summary": {
    "population": 1500000,
    "data_sources": ["arrest_data_1980_2024", "census_2020"],
    "last_data_update": "2024-01-01"
  },
  "problems": [
    {
      "id": "alameda_001",
      "title": "Elevated Arrest Rates Among Young Adults",
      "severity": "high",
      "description": "18-29 age group shows 150% higher arrest rates than state average...",
      "metrics": {
        "violent_arrests": 949,
        "property_arrests": 1593,
        "state_comparison": "+45%"
      },
      "solution": {
        "title": "Youth Intervention Programs",
        "description": "Implement community-based diversion programs...",
        "estimated_cost": "$2.5M annually",
        "expected_impact": "20-30% reduction in recidivism"
      }
    }
  ]
}
```

## Checkpoint 1 Implementation Steps

1. Set up FastAPI project structure with all files listed above
2. Configure SQLAlchemy with SQLite and create database models
3. Implement Claude service with schema analysis capability
4. Build `/api/ingest-data` endpoint that:

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Accepts CSV upload
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Uses Claude to analyze schema
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Normalizes arrest data into `data_points` table

5. Build `/api/generate-report` endpoint that:

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Queries data by county
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Sends to Claude for problem synthesis
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Returns structured report JSON

6. Add CORS middleware for frontend integration
7. Test with the arrest dataset you have

## Future Extensibility

- **New datasets**: Same `/ingest-data` endpoint, Claude adapts to new schemas
- **Multiple data types**: EAV schema handles any metric
- **Advanced analytics**: Add `data_correlations` table for ML-detected patterns
- **Real-time updates**: Webhook endpoint for automated data refreshes

## Dependencies to Add

```
anthropic==0.21.3  # Claude API
aiofiles==23.2.1   # Async file handling
```

(Most other deps already in requirements.txt)

### To-dos

- [ ] Create FastAPI project structure with app/, routes/, services/, models.py, database.py
- [ ] Implement SQLAlchemy models for datasets, data_points, data_weights, generated_reports tables
- [ ] Build Claude API service with schema analysis and report synthesis prompts
- [ ] Implement /api/ingest-data endpoint with CSV upload, Claude schema analysis, and normalization
- [ ] Implement /api/generate-report endpoint with data aggregation, Claude synthesis, and caching
- [ ] Test complete flow: ingest arrest CSV, generate report for Alameda County