import asyncio
from sqlalchemy import text
from app.database import AsyncSessionLocal

async def check_database():
    async with AsyncSessionLocal() as db:
        # Check issues
        result = await db.execute(text("SELECT COUNT(*) FROM community_issues"))
        issues_count = result.scalar()
        print(f"Issues: {issues_count}")
        
        # Check relationships
        result = await db.execute(text("SELECT COUNT(*) FROM issue_relationships"))
        relationships_count = result.scalar()
        print(f"Relationships: {relationships_count}")
        
        # Check clusters
        result = await db.execute(text("SELECT COUNT(*) FROM issue_clusters"))
        clusters_count = result.scalar()
        print(f"Clusters: {clusters_count}")
        
        # Check locations
        result = await db.execute(text("SELECT COUNT(*) FROM locations"))
        locations_count = result.scalar()
        print(f"Locations: {locations_count}")

if __name__ == "__main__":
    asyncio.run(check_database())
