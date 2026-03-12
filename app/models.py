from pydantic import BaseModel, field_validator
from typing import Optional, List
from enum import Enum


# Enums = fixed set of allowed values
# Prevents typos like "hybrd" instead of "hybrid"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RAGStrategy(str, Enum):
    NAIVE = "naive"
    HYBRID = "hybrid"
    HYDE = "hyde"
    RERANKED = "reranked"


# THE most important model in the project.
# This is what the LLM MUST return every single time.
# Instructor enforces this shape - no free text allowed.
# If LLM returns plain text, Instructor retries until
# it gets a proper RAGResponse object.

class RAGResponse(BaseModel):
    answer: str
    confidence: float       # 0.0 to 1.0
    is_answerable: bool     # False = abstain (Tier 3 questions)
    reasoning: str          # why it answered or why it refused

    @field_validator("confidence")
    def confidence_must_be_valid(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return round(v, 3)


class RetrievedChunk(BaseModel):
    content: str
    score: float
    metadata: dict
    retrieval_method: str = "vector"    # vector / bm25 / reranked


class RAGResult(BaseModel):
    # Complete result of one RAG pipeline run
    # Returned by the API and stored in the database
    strategy: str
    query: str
    answer: str
    confidence: float
    is_answerable: bool
    reasoning: str
    retrieved_chunks: List[dict]
    latency: dict                       # retrieve, generate, total


# API request shapes

class QueryRequest(BaseModel):
    question: str
    strategy: str = "naive"
    top_k: Optional[int] = 5


class BenchmarkRequest(BaseModel):
    question: str
    run_all_strategies: bool = True


# Human in the loop
# When agent cannot fix something it escalates to human
# Human responds via API with this model

class HumanApproval(BaseModel):
    anomaly_id: int
    approved: bool
    notes: Optional[str] = None


# Evaluation models

class EvalScore(BaseModel):
    question_id: str
    strategy: str
    retrieval_recall: Optional[float] = None
    answer_confidence: float = 0.0
    is_answerable: bool = True
    total_latency: float = 0.0
    hallucination_detected: bool = False


class BenchmarkReport(BaseModel):
    total_questions: int
    strategies_tested: List[str]
    scores_by_strategy: dict
    recommended_strategy: str
    summary: str