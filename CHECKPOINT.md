# RAG Benchmark Project — Session Checkpoint

## Project Overview
Multi-strategy RAG comparison engine over HuggingFace / LangChain / Anthropic docs.
Compares 4 RAG strategies: Naive RAG, Hybrid Search, HyDE, Reranked Hybrid.

**Stack:** FastAPI, Qdrant, Redis, SQLite, LiteLLM, Instructor, sentence-transformers, rank-bm25, PyMuPDF, Docker

---

## Key Decisions (Do Not Change Without Noting It Here)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- LLM: `gpt-4o-mini` for dev — swap via LiteLLM config
- Chunk strategies: fixed_size (512 chars / 50 overlap), semantic, parent-child — all stored in ONE Qdrant collection tagged by `strategy` field
- BM25 index: cached in Redis as pickle under key `bm25_index`
- Abstention: via `is_answerable` bool in RAGResponse Pydantic model
- All LLM outputs use Instructor for structured outputs — never raw strings

---

## Session Progress Tracker

### Day 0 — Environment Setup
- [ ] Virtual environment created
- [ ] requirements.txt installed
- [ ] .env file created with API keys
- [ ] Folder structure created
- [ ] docker-compose.yml written
- [ ] Docker running — Qdrant verified at localhost:6333/dashboard
- [ ] Docker running — Redis verified with PONG
- [ ] HuggingFace docs downloaded
- [ ] LangChain docs downloaded
- [ ] Research papers downloaded (5 PDFs)
- [ ] BEIR dataset downloaded
- [ ] Golden dataset started (min 10 questions tonight)

### Day 1 Morning — Foundation Files
- [ ] app/config.py — Settings class reading .env correctly
- [ ] app/database.py — SQLite tables created (BenchmarkRun)
- [ ] app/models.py — RAGResponse, AgentDecision, HumanApproval
- [ ] app/ingestion/loader.py — loads .txt, .md, .html, .pdf
- [ ] TEST: loader reads documents without error

### Day 1 Afternoon — Ingestion Pipeline
- [ ] app/ingestion/chunker.py — fixed_size, semantic, parent_child
- [ ] TEST: chunk overlap visible at chunk boundary
- [ ] app/ingestion/embedder.py — 384-dim normalised vectors
- [ ] TEST: vector magnitude ~1.0
- [ ] app/retrieval/vector_store.py — Qdrant create/store/search
- [ ] TEST: Qdrant collection created in dashboard

### Day 1 Evening — Ingest & Verify
- [ ] scripts/ingest.py — full pipeline runs
- [ ] scripts/verify_setup.py — all checks pass
- [ ] TEST: Qdrant dashboard shows 5000+ points
- [ ] TEST: Redis has bm25_index key
- [ ] data/golden_dataset.json — ALL 30 questions complete

### Day 2 Morning — First RAG Pipeline
- [ ] app/retrieval/bm25_store.py — build, cache, search
- [ ] TEST: Redis bm25_index exists after ingest
- [ ] app/retrieval/strategies/naive.py — retrieve + generate + timing
- [ ] TEST: Tier 1 question answers correctly with confidence > 0.7
- [ ] TEST: Tier 3 adversarial question returns is_answerable: false

### Day 2 Afternoon — FastAPI
- [ ] app/main.py — lifespan startup, app_state initialised
- [ ] app/api/routes.py — /query and /strategies endpoints
- [ ] TEST: curl /health returns healthy
- [ ] TEST: curl /api/v1/query returns full structured JSON
- [ ] TEST: Swagger UI loads at localhost:8000/docs
- [ ] Structured logging writing to logs/app.log

### Day 3 — Hybrid Search
- [ ] app/retrieval/strategies/hybrid.py — BM25 + vector + RRF
- [ ] TEST: Same query via naive vs hybrid shows different chunk order
- [ ] /query endpoint accepts strategy: "hybrid"

### Day 4 — HyDE
- [ ] app/retrieval/strategies/hyde.py — hypothetical doc generation + embed
- [ ] TEST: Inspect hypothetical document generated for a query
- [ ] /query endpoint accepts strategy: "hyde"

### Day 5 — Reranked Hybrid
- [ ] app/retrieval/strategies/reranked.py — cross-encoder on hybrid results
- [ ] TEST: Top chunks reordered vs hybrid baseline
- [ ] /query endpoint accepts strategy: "reranked"

### Day 6 — Parallel Benchmark Runner
- [ ] All 4 strategies run in parallel via asyncio
- [ ] /benchmark endpoint returns all 4 results in one call
- [ ] Latency comparison visible in response

### Day 7 — Evaluation Engine
- [ ] app/evaluation/metrics.py — Recall@K, MRR, NDCG
- [ ] app/evaluation/judge.py — faithfulness + relevance via LLM
- [ ] TEST: Scores computed for all 30 golden questions per strategy

### Day 8 — Report Generator
- [ ] scripts/run_benchmark.py — full benchmark run
- [ ] HTML report generated with strategy comparison table
- [ ] Final scores logged and saved

### Polish
- [ ] README.md with architecture diagram (Mermaid)
- [ ] Dockerfile complete
- [ ] .github/workflows/test.yml — CI on push
- [ ] docker-compose up — one command runs everything
- [ ] Demo script that intentionally breaks and recovers

---

## Current Status

**Currently working on:**
<!-- Update this every session -->
Session 1 — Day 0 Setup

**Last completed task:**
<!-- e.g. "app/config.py — tested, reading .env correctly" -->
Nothing yet — starting fresh

**Blockers / errors:**
<!-- Paste exact error message here -->
None

---

## Files Created So Far
<!-- Tick these off as you go -->
- [ ] docker-compose.yml
- [ ] Dockerfile
- [ ] requirements.txt
- [ ] .env.example
- [ ] app/__init__.py
- [ ] app/config.py
- [ ] app/main.py
- [ ] app/database.py
- [ ] app/models.py
- [ ] app/ingestion/__init__.py
- [ ] app/ingestion/loader.py
- [ ] app/ingestion/chunker.py
- [ ] app/ingestion/embedder.py
- [ ] app/retrieval/__init__.py
- [ ] app/retrieval/vector_store.py
- [ ] app/retrieval/bm25_store.py
- [ ] app/retrieval/strategies/__init__.py
- [ ] app/retrieval/strategies/naive.py
- [ ] app/retrieval/strategies/hybrid.py
- [ ] app/retrieval/strategies/hyde.py
- [ ] app/retrieval/strategies/reranked.py
- [ ] app/evaluation/__init__.py
- [ ] app/evaluation/metrics.py
- [ ] app/evaluation/judge.py
- [ ] app/api/__init__.py
- [ ] app/api/routes.py
- [ ] scripts/ingest.py
- [ ] scripts/verify_setup.py
- [ ] scripts/download_beir.py
- [ ] scripts/test_naive_rag.py
- [ ] scripts/run_golden_test.py
- [ ] scripts/run_benchmark.py
- [ ] data/golden_dataset.json
- [ ] tests/test_chunker.py
- [ ] README.md
- [ ] .github/workflows/test.yml

---

## Test Results Log
<!-- Paste actual test outputs here as you go — this is gold for interviews -->

### Chunk Counts (after ingest)
```
Fixed size chunks:   [fill in]
Semantic chunks:     [fill in]
Parent-child chunks: [fill in]
Total chunks:        [fill in]
Qdrant points:       [fill in]
```

### Naive RAG — Example Output
```json
// Paste a real response here after Day 2
```

### Abstention Test — Tier 3 Results
```
Adversarial questions correctly refused: [X]/10
```

### Final Benchmark Scores (fill in after Day 7-8)
```
Strategy          | Recall@5 | MRR  | Faithfulness | Avg Latency
------------------|----------|------|--------------|------------
Naive RAG         |          |      |              |
Hybrid Search     |          |      |              |
HyDE              |          |      |              |
Reranked Hybrid   |          |      |              |
```

---

## How to Start Each Opus Session

Paste this at the top of every new conversation:

```
I am building a multi-strategy RAG benchmark platform.
Here is my current checkpoint:

[PASTE THIS ENTIRE FILE]

Today I want to work on: [SESSION NAME FROM TRACKER ABOVE]

Here is the relevant existing code:
[PASTE ONLY FILES RELEVANT TO TODAY'S TASK — NOT EVERYTHING]
```

---

## Token-Saving Rules
- Paste ONLY files relevant to today's task
- Share exact error messages — not descriptions
- Ask one focused thing per session
- Update this file BEFORE closing the tab each session

---

## Session Notes
<!-- Free space — add anything useful per session -->

**Session 1:**

**Session 2:**

**Session 3:**

**Session 4:**

**Session 5:**

**Session 6:**

**Session 7:**

**Session 8:**

**Session 9:**

**Session 10:**

**Session 11:**

**Session 12:**

**Session 13:**
