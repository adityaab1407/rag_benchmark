import sys
import os
import json
import csv
import time
from pathlib import Path
from datetime import datetime
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store
from app.retrieval.strategies.naive import NaiveRAG
from app.retrieval.strategies.hybrid import HybridRAG
from app.retrieval.strategies.hyde import HyDeRAG
from app.retrieval.strategies.reranked import RerankedRAG


def load_golden_dataset():
    path = Path("data/golden_dataset.json")
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data.get("questions", [])


def load_existing_results():
    # Find the latest benchmark CSV and load completed rows
    # Returns (existing_results, completed_set)
    completed = set()
    existing_results = []

    results_dir = Path("results")
    if not results_dir.exists():
        return existing_results, completed

    existing_csvs = sorted(results_dir.glob("benchmark_*.csv"))
    if not existing_csvs:
        return existing_results, completed

    latest = existing_csvs[-1]
    logger.info("Found existing results: {} - will resume from here", latest)

    with open(latest, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only keep rows that succeeded - re-run errored ones
            if "ERROR" not in row["answer"]:
                existing_results.append(row)
                completed.add((row["question_id"], row["strategy"]))

    logger.info(
        "Loaded {} completed rows, will skip these and re-run errors",
        len(completed)
    )
    return existing_results, completed


def run_benchmark():
    logger.info("=" * 60)
    logger.info("RAG BENCHMARK RUNNER")
    logger.info("=" * 60)

    # Load existing results first - resume from where we left off
    existing_results, completed = load_existing_results()

    # Load components once - reused by all strategies
    logger.info("Loading components...")
    embedder = Embedder()
    vector_store = VectorStore()
    bm25_store = BM25Store()
    bm25_store.load_index()

    # Initialise all strategies
    strategies = {
        "naive":    NaiveRAG(embedder, vector_store),
        "hybrid":   HybridRAG(embedder, vector_store, bm25_store),
        "hyde":     HyDeRAG(embedder, vector_store),
        "reranked": RerankedRAG(embedder, vector_store, bm25_store),
    }
    logger.info("All strategies ready")

    # Load golden dataset
    questions = load_golden_dataset()
    logger.info("Loaded {} questions from golden dataset", len(questions))

    # Results storage - start with existing completed results
    results = existing_results.copy()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Count how many we actually need to run
    remaining = sum(
        1 for q in questions
        for s in strategies
        if (q.get("id", "unknown"), s) not in completed
    )
    logger.info(
        "Need to run {} queries ({} already completed)",
        remaining,
        len(completed)
    )

    current = 0

    for q in questions:
        question_id = q.get("id", "unknown")
        question_text = q.get("question", "")
        tier = q.get("tier", 0)
        is_answerable = q.get("is_answerable", True)

        # Check if all strategies for this question are done
        all_done = all(
            (question_id, s) in completed
            for s in strategies
        )
        if all_done:
            logger.info("Skipping q{} - all strategies completed", question_id)
            continue

        logger.info(
            "Question {}: [Tier {}] {}...",
            question_id, tier, question_text[:50]
        )

        strategies_run_this_question = 0

        for strategy_name, strategy in strategies.items():

            # Skip already completed
            if (question_id, strategy_name) in completed:
                logger.info("  Skipping {}/{} - already done", question_id, strategy_name)
                continue

            current += 1
            logger.info(
                "  [{}/{}] Running {} strategy...",
                current, remaining, strategy_name
            )

            try:
                result = strategy.run(question_text, top_k=5)

                row = {
                    "question_id":          question_id,
                    "tier":                 tier,
                    "question":             question_text[:100],
                    "strategy":             strategy_name,
                    "answer":               result.answer[:200],
                    "confidence":           result.confidence,
                    "is_answerable":        result.is_answerable,
                    "expected_answerable":  is_answerable,
                    "abstention_correct":   result.is_answerable == is_answerable,
                    "retrieve_latency":     result.latency["retrieve"],
                    "generate_latency":     result.latency["generate"],
                    "total_latency":        result.latency["total"],
                }
                results.append(row)
                strategies_run_this_question += 1

                logger.info(
                    "    confidence={} answerable={} latency={}s",
                    result.confidence,
                    result.is_answerable,
                    result.latency["total"]
                )

            except Exception as e:
                error_msg = str(e)
                logger.error("    FAILED: {}", error_msg)

                # Check if rate limited - wait and retry once
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    logger.warning("Rate limit hit - waiting 60s before retrying...")
                    time.sleep(60)
                    try:
                        result = strategy.run(question_text, top_k=5)
                        row = {
                            "question_id":          question_id,
                            "tier":                 tier,
                            "question":             question_text[:100],
                            "strategy":             strategy_name,
                            "answer":               result.answer[:200],
                            "confidence":           result.confidence,
                            "is_answerable":        result.is_answerable,
                            "expected_answerable":  is_answerable,
                            "abstention_correct":   result.is_answerable == is_answerable,
                            "retrieve_latency":     result.latency["retrieve"],
                            "generate_latency":     result.latency["generate"],
                            "total_latency":        result.latency["total"],
                        }
                        results.append(row)
                        strategies_run_this_question += 1
                        logger.info("    Retry succeeded")
                        continue
                    except Exception as e2:
                        logger.error("    Retry also failed: {}", e2)

                # Record the failure
                results.append({
                    "question_id":          question_id,
                    "tier":                 tier,
                    "question":             question_text[:100],
                    "strategy":             strategy_name,
                    "answer":               "ERROR: {}".format(error_msg[:100]),
                    "confidence":           0.0,
                    "is_answerable":        False,
                    "expected_answerable":  is_answerable,
                    "abstention_correct":   False,
                    "retrieve_latency":     0.0,
                    "generate_latency":     0.0,
                    "total_latency":        0.0,
                })

        # Save progress after every question
        # If the script crashes we don't lose everything
        if strategies_run_this_question > 0:
            Path("results").mkdir(exist_ok=True)
            checkpoint_path = "results/benchmark_{}.csv".format(timestamp)
            with open(checkpoint_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            logger.info("Progress saved to {}", checkpoint_path)

        # Wait between questions to avoid rate limits
        # 10 seconds = ~360 tokens/min headroom
        if strategies_run_this_question > 0 and current < remaining:
            logger.info("Waiting 10s before next question...")
            time.sleep(10)

    # Final save
    Path("results").mkdir(exist_ok=True)
    final_path = "results/benchmark_{}.csv".format(timestamp)
    with open(final_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logger.info("Final results saved to {}", final_path)

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)

    for strategy_name in strategies.keys():
        strategy_results = [
            r for r in results
            if r["strategy"] == strategy_name
            and "ERROR" not in str(r["answer"])
        ]
        if not strategy_results:
            continue

        avg_confidence = sum(
            float(r["confidence"]) for r in strategy_results
        ) / len(strategy_results)

        avg_latency = sum(
            float(r["total_latency"]) for r in strategy_results
        ) / len(strategy_results)

        abstention_accuracy = sum(
            1 for r in strategy_results
            if str(r["abstention_correct"]).lower() == "true"
        ) / len(strategy_results)

        answered = sum(
            1 for r in strategy_results
            if str(r["is_answerable"]).lower() == "true"
        )

        logger.info(
            "{:10} | avg_conf={:.2f} | abstention={:.0%} | avg_latency={:.2f}s | answered={}/{}",
            strategy_name,
            avg_confidence,
            abstention_accuracy,
            avg_latency,
            answered,
            len(strategy_results)
        )

    logger.info("=" * 60)
    logger.info("Full results: {}", final_path)


if __name__ == "__main__":
    run_benchmark()
