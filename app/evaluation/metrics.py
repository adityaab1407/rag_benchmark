import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger


# =============================================================================
# METRICS EXPLAINED
# =============================================================================
#
# RECALL@K
#   Did the correct source file appear in the top K retrieved chunks?
#   recall@1 = 0.6 means 60% of time the right file was the top chunk
#   recall@5 = 0.8 means 80% of time the right file was in top 5
#   Uses source_documents from golden dataset to check filenames
#
# MRR (Mean Reciprocal Rank)
#   At what POSITION did the first relevant chunk appear?
#   If correct chunk was rank 1: score = 1/1 = 1.0
#   If correct chunk was rank 2: score = 1/2 = 0.5
#   If correct chunk was rank 5: score = 1/5 = 0.2
#   Average across all questions = MRR
#   Higher is better. 1.0 = always top result
#
# TERM HIT RATE
#   Did any retrieved chunk contain the must_contain_terms?
#   "r=8" and "lora_alpha" must appear somewhere in top 5 chunks
#   More direct than source file matching
#   Tests if right CONTENT was retrieved, not just right file
#
# ABSTENTION ACCURACY
#   For Tier 3 questions: did system correctly say is_answerable=false?
#   For Tier 1/2 questions: did system correctly say is_answerable=true?
#   Perfect score = 1.0
#
# CONFIDENCE CALIBRATION
#   For answerable questions: was confidence >= expected_confidence_min?
#   For unanswerable: was confidence <= expected_confidence_max?
#   Measures if system knows when it knows things
#
# =============================================================================


def load_golden_dataset(path: str = "data/golden_dataset.json") -> Dict:
    data = json.loads(Path(path).read_text())
    return {q["id"]: q for q in data["questions"]}


def load_benchmark_csv(path: str) -> List[Dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "ERROR" not in str(row.get("answer", "")):
                results.append(row)
    return results


def compute_recall_at_k(
    retrieved_chunks: List[Dict],
    expected_sources: List[str],
    k: int
) -> Optional[float]:
    if not expected_sources:
        return None
    top_k = retrieved_chunks[:k]
    for chunk in top_k:
        source = chunk.get("metadata", {}).get("filename", "")
        for expected in expected_sources:
            if expected.lower() in source.lower():
                return 1.0
    return 0.0


def compute_mrr(
    retrieved_chunks: List[Dict],
    expected_sources: List[str]
) -> Optional[float]:
    if not expected_sources:
        return None
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("metadata", {}).get("filename", "")
        for expected in expected_sources:
            if expected.lower() in source.lower():
                return 1.0 / rank
    return 0.0


def compute_term_hit_rate(
    retrieved_chunks: List[Dict],
    must_contain_terms: List[str]
) -> Optional[float]:
    if not must_contain_terms:
        return None
    all_content = " ".join([
        c.get("content", "") for c in retrieved_chunks
    ]).lower()
    hits = sum(
        1 for term in must_contain_terms
        if term.lower() in all_content
    )
    return round(hits / len(must_contain_terms), 3)


def compute_confidence_calibration(
    confidence: float,
    is_answerable: bool,
    golden: Dict
) -> float:
    if is_answerable:
        expected_min = golden.get("expected_confidence_min", 0.7)
        return 1.0 if confidence >= expected_min else 0.0
    else:
        expected_max = golden.get("expected_confidence_max", 0.3)
        return 1.0 if confidence <= expected_max else 0.0


def analyze_sources(
    benchmark_csv_path: str,
    golden_dataset_path: str = "data/golden_dataset.json"
) -> Dict:
    golden = load_golden_dataset(golden_dataset_path)
    results = load_benchmark_csv(benchmark_csv_path)

    # Expected sources from golden dataset
    expected_sources = {}
    for qid, g in golden.items():
        sources = g.get("source_documents", [])
        for s in sources:
            expected_sources[s] = expected_sources.get(s, 0) + 1

    # Category breakdown from golden dataset
    categories = {}
    for qid, g in golden.items():
        cat = g.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    # Corpus breakdown - which doc types are in the golden dataset
    corpus_groups = {
        "huggingface": [],
        "langchain":   [],
        "anthropic":   [],
        "papers":      [],
        "beir":        [],
        "other":       []
    }
    for qid, g in golden.items():
        sources = g.get("source_documents", [])
        placed = False
        for s in sources:
            s_lower = s.lower()
            if any(x in s_lower for x in ["peft", "transformers", "llm_tutorial", "pipeline", "tokeniz"]):
                corpus_groups["huggingface"].append(qid)
                placed = True
            elif any(x in s_lower for x in ["langchain", "lcel", "agent", "chain"]):
                corpus_groups["langchain"].append(qid)
                placed = True
            elif "anthropic" in s_lower:
                corpus_groups["anthropic"].append(qid)
                placed = True
            elif any(x in s_lower for x in ["paper", "pdf", "lora", "rag", "attention", "selfrag"]):
                corpus_groups["papers"].append(qid)
                placed = True
            elif "scifact" in s_lower or "beir" in s_lower:
                corpus_groups["beir"].append(qid)
                placed = True
        if not placed and not sources:
            corpus_groups["other"].append(qid)

    # Per strategy source hits
    strategy_source_hits = {}
    for row in results:
        strategy = row["strategy"]
        qid = row["question_id"]
        if qid not in golden:
            continue
        g = golden[qid]
        expected = g.get("source_documents", [])
        answer = row.get("answer", "").lower()

        if strategy not in strategy_source_hits:
            strategy_source_hits[strategy] = {}

        for src in expected:
            src_stem = src.replace(".md", "").replace(".pdf", "").replace("_", " ").lower()
            hit = src_stem in answer or src.lower() in answer
            if src not in strategy_source_hits[strategy]:
                strategy_source_hits[strategy][src] = {"expected": 0, "hit": 0}
            strategy_source_hits[strategy][src]["expected"] += 1
            if hit:
                strategy_source_hits[strategy][src]["hit"] += 1

    # Top sources from CSV if top_sources column exists
    csv_sources = {}
    for row in results:
        top_src_field = row.get("top_sources", "")
        if top_src_field:
            for src in top_src_field.split("|"):
                src = src.strip()
                if src and src != "unknown":
                    csv_sources[src] = csv_sources.get(src, 0) + 1

    return {
        "expected_sources":      expected_sources,
        "categories":            categories,
        "corpus_groups":         {k: len(v) for k, v in corpus_groups.items()},
        "corpus_group_details":  corpus_groups,
        "strategy_source_hits":  strategy_source_hits,
        "csv_sources":           csv_sources,
    }


def evaluate_benchmark(
    benchmark_csv_path: str,
    golden_dataset_path: str = "data/golden_dataset.json"
) -> Dict:
    logger.info("Loading golden dataset...")
    golden = load_golden_dataset(golden_dataset_path)

    logger.info("Loading benchmark results from {}", benchmark_csv_path)
    results = load_benchmark_csv(benchmark_csv_path)
    logger.info("Loaded {} successful results", len(results))

    by_strategy = {}
    for row in results:
        s = row["strategy"]
        if s not in by_strategy:
            by_strategy[s] = []
        by_strategy[s].append(row)

    strategy_metrics = {}

    for strategy_name, rows in by_strategy.items():
        logger.info("Evaluating strategy: {}", strategy_name)

        term_hit_scores = []
        abstention_scores = []
        calibration_scores = []

        for row in rows:
            qid = row["question_id"]
            if qid not in golden:
                continue

            g = golden[qid]
            must_contain_terms = g.get("expected_retrieval", {}).get(
                "must_contain_terms", []
            )
            is_answerable_expected = g.get("is_answerable", True)
            confidence = float(row.get("confidence", 0.0))
            is_answerable_actual = str(
                row.get("is_answerable", "false")
            ).lower() == "true"

            term_hit = compute_term_hit_rate(
                [{"content": row.get("answer", "")}],
                must_contain_terms
            )
            if term_hit is not None:
                term_hit_scores.append(term_hit)

            abstention_correct = (
                is_answerable_actual == is_answerable_expected
            )
            abstention_scores.append(1.0 if abstention_correct else 0.0)

            cal = compute_confidence_calibration(
                confidence,
                is_answerable_expected,
                g
            )
            calibration_scores.append(cal)

        def safe_avg(scores):
            return round(sum(scores) / len(scores), 3) if scores else 0.0

        strategy_metrics[strategy_name] = {
            "total_questions":        len(rows),
            "term_hit_rate":          safe_avg(term_hit_scores),
            "abstention_accuracy":    safe_avg(abstention_scores),
            "confidence_calibration": safe_avg(calibration_scores),
            "avg_confidence":         round(
                sum(float(r["confidence"]) for r in rows) / len(rows), 3
            ),
            "avg_latency":            round(
                sum(float(r["total_latency"]) for r in rows) / len(rows), 3
            ),
            "answered_count":         sum(
                1 for r in rows
                if str(r.get("is_answerable", "false")).lower() == "true"
            ),
            "abstained_count":        sum(
                1 for r in rows
                if str(r.get("is_answerable", "false")).lower() == "false"
            ),
        }

        logger.info(
            "{}: term_hit={} abstention={} calibration={} avg_conf={} avg_latency={}s",
            strategy_name,
            strategy_metrics[strategy_name]["term_hit_rate"],
            strategy_metrics[strategy_name]["abstention_accuracy"],
            strategy_metrics[strategy_name]["confidence_calibration"],
            strategy_metrics[strategy_name]["avg_confidence"],
            strategy_metrics[strategy_name]["avg_latency"],
        )

    best = {}
    if strategy_metrics:
        best["term_hit_rate"] = max(
            strategy_metrics,
            key=lambda s: strategy_metrics[s]["term_hit_rate"]
        )
        best["abstention_accuracy"] = max(
            strategy_metrics,
            key=lambda s: strategy_metrics[s]["abstention_accuracy"]
        )
        best["confidence_calibration"] = max(
            strategy_metrics,
            key=lambda s: strategy_metrics[s]["confidence_calibration"]
        )
        best["fastest"] = min(
            strategy_metrics,
            key=lambda s: strategy_metrics[s]["avg_latency"]
        )

    return {
        "strategies":          strategy_metrics,
        "best":                best,
        "total_results":       len(results),
        "questions_evaluated": len(set(r["question_id"] for r in results))
    }


if __name__ == "__main__":
    import sys

    results_dir = Path("results")
    csvs = sorted(results_dir.glob("benchmark_*.csv"))

    if not csvs:
        print("No benchmark CSVs found in results/ - run benchmark first")
        sys.exit(1)

    latest = csvs[-1]
    print("Using: {}".format(latest))
    print()

    report = evaluate_benchmark(str(latest))

    print()
    print("=" * 65)
    print("EVALUATION REPORT")
    print("=" * 65)
    print("{:<12} {:>10} {:>12} {:>13} {:>10} {:>10}".format(
        "Strategy", "TermHit", "Abstention", "Calibration", "AvgConf", "AvgLatency"
    ))
    print("-" * 65)

    for strategy, m in report["strategies"].items():
        print("{:<12} {:>10} {:>12} {:>13} {:>10} {:>10}".format(
            strategy,
            "{:.0%}".format(m["term_hit_rate"]),
            "{:.0%}".format(m["abstention_accuracy"]),
            "{:.0%}".format(m["confidence_calibration"]),
            "{:.2f}".format(m["avg_confidence"]),
            "{}s".format(m["avg_latency"])
        ))

    print("=" * 65)
    print()
    print("BEST PER METRIC:")
    for metric, winner in report["best"].items():
        print("  {:25} {}".format(metric, winner))

    print()
    print("Questions evaluated: {}/30".format(report["questions_evaluated"]))
    print()

    # Source analysis
    print("=" * 65)
    print("SOURCE DOCUMENT ANALYSIS")
    print("=" * 65)

    source_report = analyze_sources(str(latest))

    print()
    print("CORPUS BREAKDOWN (questions per source group):")
    print("-" * 40)
    for group, count in source_report["corpus_groups"].items():
        bar = "#" * count
        print("  {:15} {:>3} questions  {}".format(group, count, bar))

    print()
    print("QUESTION CATEGORIES:")
    print("-" * 40)
    for cat, count in sorted(
        source_report["categories"].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print("  {:40} {}".format(cat, count))

    print()
    print("EXPECTED SOURCE FILES (how often each file is the answer):")
    print("-" * 40)
    for src, count in sorted(
        source_report["expected_sources"].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        bar = "#" * count
        print("  {:35} {:>2}x  {}".format(src, count, bar))

    if source_report["csv_sources"]:
        print()
        print("MOST RETRIEVED SOURCE FILES (from benchmark CSV):")
        print("-" * 40)
        for src, count in sorted(
            source_report["csv_sources"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]:
            bar = "#" * count
            print("  {:35} {:>2}x  {}".format(src, count, bar))

    print()
    print("NOTE: Full Recall@K requires chunk metadata in CSV")
    print("      Tomorrow's complete benchmark will enable this")