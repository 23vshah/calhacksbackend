#!/usr/bin/env python3
"""
Script to clear all data from the database
"""
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import delete
from app.database import Base, Dataset, DataPoint, DataWeight, GeneratedReport

async def clear_database():
    """Clear all data from the database"""
    
    # Create database connection
    DATABASE_URL = "sqlite+aiosqlite:///./theages.db"
    engine = create_async_engine(DATABASE_URL, echo=True)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with AsyncSessionLocal() as session:
        try:
            print("🗑️  Clearing database...")
            
            # Delete all data points
            await session.execute(delete(DataPoint))
            print("✅ Cleared data_points table")
            
            # Delete all datasets
            await session.execute(delete(Dataset))
            print("✅ Cleared datasets table")
            
            # Delete all data weights
            await session.execute(delete(DataWeight))
            print("✅ Cleared data_weights table")
            
            # Delete all generated reports
            await session.execute(delete(GeneratedReport))
            print("✅ Cleared generated_reports table")
            
            # Commit the changes
            await session.commit()
            print("🎉 Database cleared successfully!")
            
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
            await session.rollback()
        finally:
            await session.close()

if __name__ == "__main__":
    asyncio.run(clear_database())
