import csv
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from groq import Groq
import instructor
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings


# =============================================================================
# JUDGE EXPLAINED
# =============================================================================
#
# Uses llama-3.1-8b-instant as judge (separate quota from 70b)
# Runs on ALL 30 questions including tier 3
#
# FAITHFULNESS (0.0 - 1.0)
#   Does the answer only contain information grounded in the question?
#   1.0 = fully grounded, no invented facts
#   0.0 = completely fabricated
#
# RELEVANCE (0.0 - 1.0)
#   Does the answer actually address what was asked?
#   1.0 = directly and completely answers the question
#   0.0 = completely off-topic
#
# HALLUCINATION_FREE (0.0 - 1.0)  <- LOWER IS WORSE
#   Did the model invent specific facts, numbers, names?
#   1.0 = no hallucination detected
#   0.0 = severe hallucination
#
# ABSTENTION_CORRECT (0.0 or 1.0)  <- tier 3 specific
#   For unanswerable questions: did the system correctly abstain?
#   1.0 = correctly said "I don't know"
#   0.0 = confidently answered an unanswerable question
#
# Resume logic: if judge CSV already exists, skips completed rows
# Checkpoint: saves after every row, crash safe
#
# =============================================================================


class JudgeScore(BaseModel):
    faithfulness: float = Field(
        ge=0.0, le=1.0,
        description="How faithful is the answer to what the question implies? 1.0=fully grounded"
    )
    relevance: float = Field(
        ge=0.0, le=1.0,
        description="How relevant is the answer to the question? 1.0=directly answers"
    )
    hallucination_free: float = Field(
        ge=0.0, le=1.0,
        description="How free of hallucination? 1.0=no hallucination, 0.0=severe fabrication"
    )
    abstention_correct: float = Field(
        ge=0.0, le=1.0,
        description="For unanswerable questions: 1.0=correctly abstained, 0.0=hallucinated answer. For answerable questions always set 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation of scores in 1-2 sentences max"
    )


def load_benchmark_csv(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "ERROR" not in str(row.get("answer", "")):
                rows.append(row)
    return rows


def load_golden(path: str = "data/golden_dataset.json"):
    data = json.loads(Path(path).read_text())
    questions = data.get("questions", data) if isinstance(data, dict) else data
    return {q["id"]: q for q in questions}


def judge_answer(
    client,
    question: str,
    answer: str,
    is_answerable_expected: bool,
    is_answerable_actual: bool,
    ground_truth: Optional[str],
    must_not_hallucinate: list,
    tier: int,
) -> Optional[JudgeScore]:

    # Build hallucination traps
    traps_str = ""
    if must_not_hallucinate:
        traps_str = (
            "\nHallucination traps — penalise hallucination_free if answer mentions:\n"
            + "\n".join(f"  - {t}" for t in must_not_hallucinate[:5])
        )

    # Ground truth hint
    truth_str = ""
    if ground_truth:
        truth_str = f"\nGround truth answer (for reference):\n{ground_truth[:300]}"

    # Tier 3 specific instructions
    if tier == 3:
        tier_instruction = (
            "\nThis is an ADVERSARIAL question — the correct answer is that the system "
            "should NOT be able to answer it (the information is missing, outdated, or ambiguous). "
            f"The system {'correctly abstained' if not is_answerable_actual else 'incorrectly tried to answer'}. "
            "Set abstention_correct=1.0 if system said it cannot answer, 0.0 if it fabricated an answer."
        )
    else:
        tier_instruction = "\nThis is an answerable question. Set abstention_correct=1.0 always for answerable questions."

    prompt = f"""You are a strict but fair RAG system evaluator.

Question: {question}

RAG System Answer: {answer[:400]}
{truth_str}
{traps_str}
{tier_instruction}

Score on 4 dimensions (0.0 to 1.0):

1. faithfulness: Are claims grounded and reasonable? No invented specifics? 1.0=fully grounded
2. relevance: Does it directly address the question? 1.0=fully answers, 0.5=partial, 0.0=off-topic
3. hallucination_free: Any invented facts, numbers, names not in question? 1.0=clean, 0.0=fabricated
4. abstention_correct: See instructions above.

Be strict — mediocre answers score 0.5-0.6, not 0.8+.
Reasoning must be 1-2 sentences only.
"""

    try:
        score = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            response_model=JudgeScore,
            max_retries=2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict RAG evaluation judge. "
                        "Score answers honestly and return valid JSON only."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
        return score
    except Exception as e:
        logger.error("Judge failed: {}", str(e)[:120])
        return None


def run_judge(
    benchmark_csv: str,
    golden_path: str = "data/golden_dataset.json",
    delay: float = 2.0,
):
    logger.info("=" * 60)
    logger.info("LLM JUDGE — All 30 questions, llama-3.1-8b-instant")
    logger.info("=" * 60)

    client = instructor.from_groq(
        Groq(api_key=settings.groq_api_key),
        mode=instructor.Mode.JSON
    )

    rows = load_benchmark_csv(benchmark_csv)
    golden = load_golden(golden_path)

    eligible = [r for r in rows]
    logger.info(
        "Total clean rows: {} | All eligible for judging: {}",
        len(rows), len(eligible)
    )

    # Resume logic — find existing judge CSV
    existing_judges = sorted(Path("results").glob("judge_*.csv"))
    completed_pairs = set()
    results = []
    output_path = None

    if existing_judges:
        latest_judge = existing_judges[-1]
        output_path = str(latest_judge)
        logger.info("Found existing judge results: {}", output_path)
        with open(output_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                results.append(row)
                if row.get("reasoning") != "JUDGE_FAILED":
                    completed_pairs.add((row["question_id"], row["strategy"]))
        logger.info("Already judged: {} pairs — will skip these", len(completed_pairs))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/judge_{timestamp}.csv"
        logger.info("Starting fresh: {}", output_path)

    # Filter out already completed
    remaining = [
        r for r in eligible
        if (r["question_id"], r["strategy"]) not in completed_pairs
    ]
    logger.info("Remaining to judge: {}/{}", len(remaining), len(eligible))

    if not remaining:
        logger.info("All rows already judged — printing summary only")
    else:
        fieldnames = [
            "question_id", "tier", "strategy", "question", "answer",
            "faithfulness", "relevance", "hallucination_free",
            "abstention_correct", "judge_score", "reasoning"
        ]

        for i, row in enumerate(remaining, 1):
            qid      = row["question_id"]
            strategy = row["strategy"]
            question = row["question"]
            answer   = row["answer"]
            tier     = int(row.get("tier", 1))

            g = golden.get(qid, {})
            ground_truth           = g.get("ground_truth_answer", None)
            must_not_hallucinate   = g.get("expected_retrieval", {}).get("must_not_hallucinate", [])
            is_answerable_expected = g.get("is_answerable", True)
            is_answerable_actual   = str(row.get("is_answerable", "false")).lower() == "true"

            logger.info(
                "[{}/{}] Judging {}/{} (tier {})",
                i, len(remaining), qid, strategy, tier
            )

            score = judge_answer(
                client,
                question=question,
                answer=answer,
                is_answerable_expected=is_answerable_expected,
                is_answerable_actual=is_answerable_actual,
                ground_truth=ground_truth,
                must_not_hallucinate=must_not_hallucinate,
                tier=tier,
            )

            if score:
                overall = round(
                    (score.faithfulness + score.relevance +
                     score.hallucination_free + score.abstention_correct) / 4, 3
                )
                result_row = {
                    "question_id":        qid,
                    "tier":               tier,
                    "strategy":           strategy,
                    "question":           question[:100],
                    "answer":             answer[:200],
                    "faithfulness":       round(score.faithfulness, 3),
                    "relevance":          round(score.relevance, 3),
                    "hallucination_free": round(score.hallucination_free, 3),
                    "abstention_correct": round(score.abstention_correct, 3),
                    "judge_score":        overall,
                    "reasoning":          score.reasoning[:200],
                }
                logger.info(
                    "  faith={:.2f} rel={:.2f} halluc={:.2f} abst={:.2f} overall={:.2f}",
                    score.faithfulness, score.relevance,
                    score.hallucination_free, score.abstention_correct,
                    overall,
                )
            else:
                result_row = {
                    "question_id":        qid,
                    "tier":               tier,
                    "strategy":           strategy,
                    "question":           question[:100],
                    "answer":             answer[:200],
                    "faithfulness":       None,
                    "relevance":          None,
                    "hallucination_free": None,
                    "abstention_correct": None,
                    "judge_score":        None,
                    "reasoning":          "JUDGE_FAILED",
                }
                logger.warning("  Judge failed for {}/{}", qid, strategy)

            results.append(result_row)

            # Checkpoint after every row
            Path("results").mkdir(exist_ok=True)
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

            if i < len(remaining):
                time.sleep(delay)

    # Final summary
    def safe_avg(values):
        vals = [float(v) for v in values if v not in [None, "None", ""]]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    clean_results = [r for r in results if r.get("judge_score") not in [None, "None", ""]]

    logger.info("")
    logger.info("=" * 60)
    logger.info("JUDGE SUMMARY")
    logger.info("=" * 60)

    for strategy in ["naive", "hybrid", "hyde", "reranked"]:
        sub = [r for r in clean_results if r["strategy"] == strategy]
        if not sub:
            continue
        logger.info(
            "{:10} | faith={:.2f} rel={:.2f} halluc={:.2f} abst={:.2f} overall={:.2f} n={}",
            strategy,
            safe_avg([r["faithfulness"] for r in sub]),
            safe_avg([r["relevance"] for r in sub]),
            safe_avg([r["hallucination_free"] for r in sub]),
            safe_avg([r["abstention_correct"] for r in sub]),
            safe_avg([r["judge_score"] for r in sub]),
            len(sub),
        )

    logger.info("")
    logger.info("BY TIER:")
    for tier in [1, 2, 3]:
        tier_sub = [r for r in clean_results if str(r.get("tier")) == str(tier)]
        if not tier_sub:
            continue
        logger.info(
            "  Tier {} | overall={:.2f} halluc_free={:.2f} n={}",
            tier,
            safe_avg([r["judge_score"] for r in tier_sub]),
            safe_avg([r["hallucination_free"] for r in tier_sub]),
            len(tier_sub),
        )

    logger.info("=" * 60)
    logger.info("Results saved: {}", output_path)
    return output_path


if __name__ == "__main__":
    results_dir = Path("results")
    csvs = sorted(results_dir.glob("benchmark_*.csv"))

    if not csvs:
        print("No benchmark CSV found in results/ — run benchmark first")
        exit(1)

    latest = str(csvs[-1])
    logger.info("Using benchmark: {}", latest)

    run_judge(latest)