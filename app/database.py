from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime


# SQLite = one .db file on your disk
# No server. No installation. Built into Python.
# aiosqlite = async version so FastAPI does not block on DB calls
DATABASE_URL = "sqlite+aiosqlite:///./pipeline_monitor.db"

engine = create_async_engine(DATABASE_URL, echo=False)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


class BenchmarkRun(Base):
    # One row = one RAG query run
    # Every time your system answers a question, a row is inserted here
    # This is how you track and compare strategies over time
    __tablename__ = "benchmark_runs"

    id = Column(Integer, primary_key=True)
    question_id = Column(String)        # e.g. "q001" from golden dataset
    strategy = Column(String)           # naive / hybrid / hyde / reranked
    answer = Column(String)             # what the LLM said
    confidence = Column(Float)          # 0.0 to 1.0
    is_answerable = Column(String)      # "true" or "false"
    retrieve_latency = Column(Float)    # seconds spent retrieving chunks
    generate_latency = Column(Float)    # seconds spent generating answer
    total_latency = Column(Float)       # total seconds end to end
    timestamp = Column(DateTime, default=datetime.utcnow)


async def init_db():
    # Creates pipeline_monitor.db file and all tables
    # Safe to call multiple times - only creates if not exists
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    # FastAPI dependency - inject into routes to get DB session
    async with AsyncSessionLocal() as session:
        yield session