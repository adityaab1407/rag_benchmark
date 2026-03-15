"""
app/evaluation/ragas_eval.py
=============================
RAGAS evaluation — industry standard RAG metrics.

INSTALL FIRST:
  pip install ragas datasets langchain-groq langchain-community
  (needs Microsoft C++ Build Tools on Windows for scikit-network)

RAGAS METRICS:
  answer_relevancy      embedding similarity of answer to question
                        no LLM needed, fast
                        score 0-1, higher = more relevant

  faithfulness          are all claims in the answer supported by
                        the retrieved context?
                        uses LLM to check each claim
                        score 0-1, higher = more faithful

  context_precision     of the chunks retrieved, what fraction
                        were actually useful for the answer?
                        score 0-1, higher = more precise retrieval

  context_recall        did retrieval cover all information needed
                        to answer the question fully?
                        requires ground_truth — tier 1+2 only
                        score 0-1, higher = more complete retrieval

REQUIRES:
  chunk_texts column in benchmark CSV
    → updated run_benchmark.py saves this
    → if missing, backfill_chunks_via_retrieval() re-runs retrieval

  ground_truth_answer in golden_dataset.json
    → already present for tier 1+2 questions

HOW TO RUN:
  python -m app.evaluation.ragas_eval

OUTPUT:
  results/ragas_results.json
  Printed summary table
  Auto-loaded by Streamlit Benchmark tab
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from typing import Optional
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check RAGAS available
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings
    RAGAS_AVAILABLE = True
except ImportError as e:
    logger.warning("RAGAS not installed: {}", e)
    logger.warning("Run: pip install ragas datasets langchain-groq langchain-community")
    RAGAS_AVAILABLE = False

from app.config import settings


# =============================================================================
# DATA LOADING
# =============================================================================

def load_benchmark_csv(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "ERROR" not in str(row.get("answer", "")):
                rows.append(row)
    return rows


def load_golden(path: str = "data/golden_dataset.json") -> dict:
    data      = json.loads(Path(path).read_text())
    questions = data.get("questions", data) if isinstance(data, dict) else data
    return {q["id"]: q for q in questions}


def get_chunk_texts(row: dict) -> list:
    """Parse chunk_texts column — triple-pipe separated."""
    raw = row.get("chunk_texts", "")
    if not raw:
        return []
    return [c.strip() for c in raw.split("|||") if c.strip()]


def backfill_chunks_via_retrieval(rows: list) -> list:
    """
    If chunk_texts column is missing or empty, re-run retrieval only
    (no Groq) to get chunk content. Used for old-format CSVs.
    No tokens burned — only embedding + Qdrant search.
    """
    logger.info("chunk_texts missing — backfilling via retrieval (no LLM)...")

    from app.ingestion.embedder import Embedder
    from app.retrieval.vector_store import VectorStore
    from app.retrieval.bm25_store import BM25Store

    embedder     = Embedder()
    vector_store = VectorStore()
    bm25_store   = BM25Store()
    bm25_store.load_index()

    chunk_cache = {}

    for row in rows:
        qid      = row["question_id"]
        strategy = row["strategy"]
        question = row["question"]
        key      = f"{qid}_{strategy}"

        if key in chunk_cache:
            row["_chunks"] = chunk_cache[key]
            continue

        try:
            embedding = embedder.embed_single(question)
            results   = vector_store.search(
                query_embedding=embedding,
                top_k=5,
                strategy_filter="fixed_size" if strategy == "naive" else None
            )
            texts            = [r.get("content", "")[:500] for r in results[:3]]
            chunk_cache[key] = texts
            row["_chunks"]   = texts
            logger.info("  Backfilled {}/{} — {} chunks", qid, strategy, len(texts))

        except Exception as e:
            logger.warning("  Backfill failed {}/{}: {}", qid, strategy, e)
            row["_chunks"] = []

    return rows


# =============================================================================
# DATASET BUILDER
# =============================================================================

def build_ragas_dataset(
    rows: list,
    golden: dict,
    strategy: str,
    tiers: list = [1, 2],
) -> Optional[object]:
    """Build RAGAS Dataset for one strategy."""

    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []

    eligible = [
        r for r in rows
        if r["strategy"] == strategy
        and int(r.get("tier", 0)) in tiers
        and str(r.get("is_answerable", "false")).lower() == "true"
    ]

    for row in eligible:
        qid = row["question_id"]
        g   = golden.get(qid, {})

        chunks = get_chunk_texts(row)
        if not chunks:
            chunks = row.get("_chunks", [])
        if not chunks:
            logger.warning("  No chunks for {}/{} — skipping", qid, strategy)
            continue

        gt = g.get("ground_truth_answer", "") or ""

        questions.append(row["question"])
        answers.append(row["answer"])
        contexts.append(chunks)
        ground_truths.append(gt)

    if not questions:
        return None

    logger.info("  {} rows ready for RAGAS", len(questions))

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })


# =============================================================================
# MAIN EVALUATOR
# =============================================================================

def run_ragas_eval(
    benchmark_csv: str,
    golden_path: str = "data/golden_dataset.json",
    tiers: list = [1, 2],
    delay: float = 15.0,
) -> dict:

    if not RAGAS_AVAILABLE:
        logger.error("RAGAS not installed.")
        logger.error("pip install ragas datasets langchain-groq langchain-community")
        return {}

    logger.info("=" * 60)
    logger.info("RAGAS EVALUATION — llama-3.1-8b-instant")
    logger.info("=" * 60)

    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=settings.groq_api_key,
        temperature=0,
    )
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    ragas_llm        = LangchainLLMWrapper(groq_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    rows   = load_benchmark_csv(benchmark_csv)
    golden = load_golden(golden_path)
    logger.info("Loaded {} clean rows", len(rows))

    has_chunks = any(row.get("chunk_texts") for row in rows)
    if not has_chunks:
        rows = backfill_chunks_via_retrieval(rows)

    has_gt = any(
        golden.get(qid, {}).get("ground_truth_answer")
        for qid in set(r["question_id"] for r in rows)
    )

    metrics_to_run = [answer_relevancy, faithfulness, context_precision]
    if has_gt:
        metrics_to_run.append(context_recall)
        logger.info("context_recall enabled — ground truths found")
    else:
        logger.info("context_recall skipped — no ground truths")

    for m in metrics_to_run:
        m.llm        = ragas_llm
        m.embeddings = ragas_embeddings

    results = {}

    for strategy in ["naive", "hybrid", "hyde", "reranked"]:
        logger.info("")
        logger.info("Evaluating: {}", strategy)

        dataset = build_ragas_dataset(rows, golden, strategy, tiers)
        if dataset is None:
            logger.warning("  No valid data for {} — skipping", strategy)
            continue

        try:
            score = evaluate(dataset=dataset, metrics=metrics_to_run)

            results[strategy] = {
                "answer_relevancy":  round(float(score.get("answer_relevancy", 0) or 0), 3),
                "faithfulness":      round(float(score.get("faithfulness", 0) or 0), 3),
                "context_precision": round(float(score.get("context_precision", 0) or 0), 3),
                "context_recall":    round(float(score.get("context_recall", 0) or 0), 3) if has_gt else None,
                "n": len(dataset),
            }

            logger.info(
                "  ans_rel={} faith={} ctx_prec={} ctx_rec={}",
                results[strategy]["answer_relevancy"],
                results[strategy]["faithfulness"],
                results[strategy]["context_precision"],
                results[strategy]["context_recall"],
            )

        except Exception as e:
            logger.error("  RAGAS failed for {}: {}", strategy, str(e)[:200])
            results[strategy] = {"error": str(e)[:200], "n": 0}

        if strategy != "reranked":
            logger.info("  Waiting {}s before next strategy...", delay)
            time.sleep(delay)

    Path("results").mkdir(exist_ok=True)
    output_path = "results/ragas_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("RAGAS SUMMARY")
    logger.info("=" * 60)
    logger.info(
        "{:<12} {:>10} {:>12} {:>12} {:>12}",
        "strategy", "ans_rel", "faithfulness", "ctx_prec", "ctx_recall"
    )
    logger.info("-" * 60)
    for s in ["naive", "hybrid", "hyde", "reranked"]:
        if s not in results or "error" in results[s]:
            logger.info("{:<12} ERROR", s)
            continue
        r = results[s]
        logger.info(
            "{:<12} {:>10} {:>12} {:>12} {:>12}",
            s,
            r.get("answer_relevancy") or "—",
            r.get("faithfulness") or "—",
            r.get("context_precision") or "—",
            r.get("context_recall") or "—",
        )
    logger.info("=" * 60)
    logger.info("Saved: {}", output_path)
    return results


if __name__ == "__main__":
    csvs = sorted(Path("results").glob("benchmark_*.csv"))
    if not csvs:
        print("No benchmark CSV in results/ — run benchmark first")
        sys.exit(1)
    latest = str(csvs[-1])
    logger.info("Using benchmark: {}", latest)
    run_ragas_eval(latest)