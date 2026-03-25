import os
import aiosqlite
from pathlib import Path
from loguru import logger

DB_PATH = os.environ.get("DB_PATH", "pipeline_monitor.db")


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:

        await db.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP,
                strategy         TEXT,
                query            TEXT,
                answer           TEXT,
                confidence       REAL,
                is_answerable    INTEGER,
                retrieve_latency REAL,
                generate_latency REAL,
                total_latency    REAL,
                top_chunk_score  REAL
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS observability_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         DATETIME DEFAULT CURRENT_TIMESTAMP,
                request_id        TEXT,
                endpoint          TEXT,
                strategy          TEXT,
                query             TEXT,
                prompt_tokens     INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens      INTEGER DEFAULT 0,
                cost_usd          REAL DEFAULT 0.0,
                retrieve_ms       REAL DEFAULT 0.0,
                generate_ms       REAL DEFAULT 0.0,
                total_ms          REAL DEFAULT 0.0,
                confidence        REAL DEFAULT 0.0,
                is_answerable     INTEGER DEFAULT 1,
                model             TEXT,
                error             TEXT
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP,
                request_id   TEXT,
                strategy     TEXT,
                question     TEXT,
                answer       TEXT,
                rating       INTEGER,
                comment      TEXT
            )
        """)

        await db.commit()
        logger.info("Database initialized — query_logs, observability_log, feedback tables ready")


async def log_query(
    strategy:         str,
    query:            str,
    answer:           str,
    confidence:       float,
    is_answerable:    bool,
    retrieve_latency: float,
    generate_latency: float,
    total_latency:    float,
    top_chunk_score:  float = 0.0,
):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO query_logs
            (strategy, query, answer, confidence, is_answerable,
             retrieve_latency, generate_latency, total_latency, top_chunk_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy, query[:500], answer[:500],
            confidence, int(is_answerable),
            retrieve_latency, generate_latency, total_latency,
            top_chunk_score,
        ))
        await db.commit()


async def log_observability(
    request_id:        str,
    endpoint:          str,
    strategy:          str,
    query:             str,
    prompt_tokens:     int   = 0,
    completion_tokens: int   = 0,
    total_tokens:      int   = 0,
    cost_usd:          float = 0.0,
    retrieve_ms:       float = 0.0,
    generate_ms:       float = 0.0,
    total_ms:          float = 0.0,
    confidence:        float = 0.0,
    is_answerable:     bool  = True,
    model:             str   = "",
    error:             str   = "",
):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO observability_log
            (request_id, endpoint, strategy, query,
             prompt_tokens, completion_tokens, total_tokens, cost_usd,
             retrieve_ms, generate_ms, total_ms,
             confidence, is_answerable, model, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request_id, endpoint, strategy, query[:300],
            prompt_tokens, completion_tokens, total_tokens, cost_usd,
            retrieve_ms, generate_ms, total_ms,
            confidence, int(is_answerable),
            model, error[:300] if error else "",
        ))
        await db.commit()


async def save_feedback(
    request_id: str,
    strategy:   str,
    question:   str,
    answer:     str,
    rating:     int,
    comment:    str = "",
):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO feedback
            (request_id, strategy, question, answer, rating, comment)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (request_id, strategy, question[:300], answer[:300], rating, comment))
        await db.commit()


async def get_feedback_stats():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT
                strategy,
                COUNT(*) AS total,
                SUM(CASE WHEN rating =  1 THEN 1 ELSE 0 END) AS thumbs_up,
                SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS thumbs_down,
                ROUND(
                    100.0 * SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) / COUNT(*), 1
                ) AS positive_pct
            FROM feedback
            GROUP BY strategy
            ORDER BY strategy
        """)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_observability_stats():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute("""
            SELECT
                strategy,
                COUNT(*)                        AS total_requests,
                AVG(total_tokens)               AS avg_tokens,
                SUM(cost_usd)                   AS total_cost,
                AVG(cost_usd)                   AS avg_cost,
                AVG(total_ms)                   AS avg_latency_ms,
                AVG(confidence)                 AS avg_confidence,
                SUM(CASE WHEN error != '' THEN 1 ELSE 0 END) AS error_count,
                SUM(CASE WHEN is_answerable = 1 THEN 1 ELSE 0 END) AS answered_count
            FROM observability_log
            WHERE strategy != ''
            GROUP BY strategy
            ORDER BY strategy
        """)
        rows         = await cursor.fetchall()
        per_strategy = [dict(r) for r in rows]

        cursor = await db.execute("""
            SELECT
                timestamp, request_id, endpoint, strategy,
                query, total_tokens, cost_usd, total_ms,
                confidence, is_answerable, error
            FROM observability_log
            ORDER BY id DESC
            LIMIT 50
        """)
        rows   = await cursor.fetchall()
        recent = [dict(r) for r in rows]

        cursor = await db.execute("""
            SELECT
                COUNT(*)          AS total_requests,
                SUM(cost_usd)     AS total_cost,
                SUM(total_tokens) AS total_tokens,
                AVG(total_ms)     AS avg_latency_ms
            FROM observability_log
        """)
        row     = await cursor.fetchone()
        summary = dict(row) if row else {}

        return {
            "per_strategy": per_strategy,
            "recent":       recent,
            "summary":      summary,
        }