# Setup & Usage Guide

## Prerequisites
- Python 3.11+
- Groq API key (free tier works: https://console.groq.com)

---

## Installation

```bash
python3 -m venv .rag_bench

# Mac/Linux
source .rag_bench/bin/activate

# Windows
.rag_bench\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key from console.groq.com |
| `LLM_MODEL` | No | Defaults to `qwen/qwen3-32b` |
| `QDRANT_HOST` | Docker only | Set to `qdrant` (triggers server mode) |
| `QDRANT_PORT` | Docker only | Set to `6333` |
| `DB_PATH` | Docker only | Path to shared SQLite DB |

---

## Running Locally

### Step 1 — Get the corpus

You need raw documents in `data/raw/` before ingesting. Either bring your own PDFs/markdown files, or download the BeIR/SciFact dataset:
```bash
pip install datasets
python3 scripts/beirpapers.py
```
This downloads ~5k scientific abstracts into `data/raw/beir/`.

### Step 2 — Ingest corpus
```bash
python3 scripts/ingest.py
```
Chunks and embeds all documents into Qdrant local storage and builds the BM25 index. Takes ~10 mins on first run. Outputs:
- `qdrant_storage/` — vector DB (local file mode)
- `bm25_index.pkl` — keyword search index

### Step 3 — Verify setup
```bash
python3 scripts/verify_setup.py
```
All checks should pass before continuing.

### Step 4 — Start the API
```bash
uvicorn app.main:app --reload

# API:       http://localhost:8000
# Swagger:   http://localhost:8000/docs
```

### Step 5 — Start the dashboard
```bash
streamlit run streamlit_app.py

# Dashboard: http://localhost:8501
```

---

## Running with Docker

> Requires `docker compose` (v2). If you only have `docker-compose` (v1), install the plugin: `sudo apt install docker-compose-plugin`

```bash
cp .env.example .env
# Add GROQ_API_KEY

docker compose up -d

# API:       http://localhost:8000
# Dashboard: http://localhost:8501
# Qdrant UI: http://localhost:6333/dashboard
```

The first `docker compose up` starts an **empty** Qdrant server. You need to ingest into it once:
```bash
# With containers running, ingest from your local venv:
QDRANT_HOST=localhost python3 scripts/ingest.py
```
This populates the Qdrant container over HTTP. Only needed once — data persists in the `qdrant_storage` Docker volume.

To stop:
```bash
docker compose down
```

---

## Running the Benchmark

```bash
python3 scripts/run_benchmark.py
```
- Runs all 4 strategies across 30 golden questions
- Resumes automatically if interrupted
- Output: `results/benchmark_TIMESTAMP.csv`

---

## Running Evaluation

```bash
# Structural metrics (no LLM calls)
python3 -m app.evaluation.metrics

# LLM judge (uses llama-3.1-8b-instant)
python3 -m app.evaluation.judge

# Output: results/judge_TIMESTAMP.csv
```

---

## Re-ingesting Corpus

```bash
python3 scripts/ingest.py --force
```

---

## API Reference

```
GET  /api/v1/health              component status check
GET  /api/v1/strategies          list available strategies
POST /api/v1/query               single strategy query
POST /api/v1/benchmark           all 4 strategies in parallel
POST /api/v1/feedback            save thumbs up/down rating
```

### Example — single query
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LoRA?", "strategy": "hybrid"}'
```

### Example — benchmark all 4
```bash
curl -X POST http://localhost:8000/api/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LoRA and how does it reduce trainable parameters?"}'
```

### Example response
```json
{
  "query": "What is LoRA...",
  "best_strategy": "hybrid",
  "fastest_strategy": "hybrid",
  "total_time": 4.009,
  "total_cost_usd": 0.00182,
  "strategies": [
    {
      "strategy": "hybrid",
      "answer": "LoRA reduces trainable parameters by 10,000x...",
      "confidence": 0.9,
      "is_answerable": true,
      "latency": {"retrieve": 1.9, "generate": 0.7, "total": 2.6},
      "cost_usd": 0.000441
    }
  ]
}
```

---

## MCP Server (Claude Desktop)

Register the RAG benchmark as a tool Claude Desktop can call directly.

```json
// %APPDATA%\Claude\claude_desktop_config.json
{
  "mcpServers": {
    "rag-benchmark": {
      "command": "python3",
      "args": ["path/to/app/mcp/server.py"],
      "env": { "RAG_API_BASE": "http://localhost:8000/api/v1" }
    }
  }
}
```

Available tools: `rag_benchmark`, `rag_query`, `rag_health`

Requires uvicorn running locally before starting Claude Desktop.


---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key from console.groq.com |
| `QDRANT_HOST` | Docker only | Set to `qdrant` |
| `QDRANT_PORT` | Docker only | Set to `6333` |
| `REDIS_HOST` | Docker only | Set to `redis` |
| `REDIS_PORT` | Docker only | Set to `6379` |

---

## Running Locally

### Step 1 — Ingest corpus
```bash
python scripts/ingest.py
```
Chunks and embeds 219 documents into Qdrant (30,432 vectors). Takes ~10 mins on first run.

### Step 2 — Start the API
```bash
uvicorn app.main:app --reload

# API:       http://localhost:8000
# Swagger:   http://localhost:8000/docs
```

### Step 3 — Start the dashboard
```bash
streamlit run streamlit_app.py

# Dashboard: http://localhost:8501
```

---

## Running with Docker

```bash
cp .env.example .env
# Add GROQ_API_KEY

docker-compose up

# API:       http://localhost:8000
# Dashboard: http://localhost:8501
# Qdrant UI: http://localhost:6333/dashboard
```

---

## Running the Benchmark

```bash
python scripts/run_benchmark.py
```
- Runs all 4 strategies across 30 golden questions
- Resumes automatically if interrupted
- Output: `results/benchmark_TIMESTAMP.csv`

---

## Running Evaluation

```bash
# Structural metrics (no LLM calls)
python -m app.evaluation.metrics

# LLM judge (uses llama-3.1-8b-instant)
python -m app.evaluation.judge

# Output: results/judge_TIMESTAMP.csv
```

---

## Re-ingesting Corpus

```bash
python scripts/ingest.py --force
```

---

## API Reference

```
GET  /api/v1/health              component status check
GET  /api/v1/strategies          list available strategies
POST /api/v1/query               single strategy query
POST /api/v1/benchmark           all 4 strategies in parallel
POST /api/v1/feedback            save thumbs up/down rating
```

### Example — single query
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LoRA?", "strategy": "hybrid"}'
```

### Example — benchmark all 4
```bash
curl -X POST http://localhost:8000/api/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LoRA and how does it reduce trainable parameters?"}'
```

### Example response
```json
{
  "query": "What is LoRA...",
  "best_strategy": "hybrid",
  "fastest_strategy": "hybrid",
  "total_time": 4.009,
  "total_cost_usd": 0.00182,
  "strategies": [
    {
      "strategy": "hybrid",
      "answer": "LoRA reduces trainable parameters by 10,000x...",
      "confidence": 0.9,
      "is_answerable": true,
      "latency": {"retrieve": 1.9, "generate": 0.7, "total": 2.6},
      "cost_usd": 0.000441
    }
  ]
}
```

---

## MCP Server (Claude Desktop)

Register the RAG benchmark as a tool Claude Desktop can call directly.

```json
// %APPDATA%\Claude\claude_desktop_config.json
{
  "mcpServers": {
    "rag-benchmark": {
      "command": "python",
      "args": ["path/to/app/mcp/server.py"],
      "env": { "RAG_API_BASE": "http://localhost:8000/api/v1" }
    }
  }
}
```

Available tools: `rag_benchmark`, `rag_query`, `rag_health`

Requires uvicorn running locally before starting Claude Desktop.