"""
app/models.py  — UPDATED VERSION
==================================
Added TokenUsage and updated RAGResult to capture
token usage from Groq responses.

Changes from original:
  + TokenUsage         new model for token tracking
  + RAGResult          added token_usage, request_id, cost_usd fields
  Everything else unchanged.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict


# =============================================================================
# GROQ PRICING (as of 2025)
# These are approximate — check https://console.groq.com/settings/billing
# =============================================================================

GROQ_PRICES = {
    # model_name: (input_price_per_1k, output_price_per_1k) in USD
    "llama-3.3-70b-versatile": (0.00059, 0.00079),
    "llama-3.1-8b-instant":    (0.00005, 0.00008),
    "groq/llama-3.3-70b-versatile": (0.00059, 0.00079),
    "groq/llama-3.1-8b-instant":    (0.00005, 0.00008),
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for a Groq API call."""
    prices = GROQ_PRICES.get(model, (0.00059, 0.00079))
    input_cost  = (prompt_tokens / 1000) * prices[0]
    output_cost = (completion_tokens / 1000) * prices[1]
    return round(input_cost + output_cost, 8)


# =============================================================================
# TOKEN USAGE MODEL
# =============================================================================

class TokenUsage(BaseModel):
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    total_tokens:      int   = 0
    cost_usd:          float = 0.0
    model:             str   = ""


# =============================================================================
# EXISTING MODELS — UNCHANGED
# =============================================================================

class RAGResponse(BaseModel):
    """Structured output from LLM generation step."""
    answer:        str   = Field(description="The answer to the question based on context")
    confidence:    float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    is_answerable: bool  = Field(description="Whether the question can be answered from context")
    reasoning:     str   = Field(description="Brief reasoning for the answer and confidence")


class RAGResult(BaseModel):
    """Full result from a RAG pipeline run."""
    strategy:         str
    query:            str
    answer:           str
    confidence:       float
    is_answerable:    bool
    reasoning:        str
    retrieved_chunks: List[Dict] = []
    latency:          Dict[str, float] = {}

    # NEW — observability fields
    token_usage:  Optional[TokenUsage] = None
    request_id:   Optional[str]        = None
    cost_usd:     Optional[float]      = None


class QueryRequest(BaseModel):
    question: str
    strategy: str = "naive"
    top_k:    int = 5


class BenchmarkRequest(BaseModel):
    question: str
    top_k:    int = 5


class StrategyComparison(BaseModel):
    strategy:         str
    answer:           str
    confidence:       float
    is_answerable:    bool
    latency:          Dict[str, float]
    top_chunk_score:  float = 0.0
    retrieval_method: str   = ""
    token_usage:      Optional[TokenUsage] = None
    cost_usd:         Optional[float]      = None


class BenchmarkResult(BaseModel):
    query:             str
    strategies:        List[StrategyComparison]
    best_strategy:     str
    fastest_strategy:  str
    total_time:        float
    summary:           str
    total_cost_usd:    Optional[float] = None
    request_id:        Optional[str]   = None