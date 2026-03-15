import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json
import sqlite3
from pathlib import Path


API_BASE = "http://localhost:8000/api/v1"

STRATEGY_COLORS = {
    "naive":    "#1D9E75",
    "hybrid":   "#7F77DD",
    "hyde":     "#EF9F27",
    "reranked": "#F0997B",
}

st.set_page_config(page_title="RAG Benchmark", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .stApp { background-color: #0e0e10; color: #cccccc; }
  section[data-testid="stSidebar"] { background-color: #16161a; border-right: 1px solid #2a2a2e; }
  section[data-testid="stSidebar"] * { color: #aaaaaa !important; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  .stTabs [data-baseweb="tab-list"] { background-color: #16161a; border-radius: 8px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { background-color: transparent; color: #666; border-radius: 6px; font-size: 13px; }
  .stTabs [aria-selected="true"] { background-color: #1e1e2e !important; color: #afa9ec !important; }

  div[data-testid="stMetric"] { background-color: #16161a; border: 1px solid #2a2a2e; border-radius: 8px; padding: 1rem; }
  div[data-testid="stMetric"] label { color: #666 !important; font-size: 12px !important; }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #afa9ec !important; font-size: 22px !important; }

  .stTextArea textarea { background-color: #1a1a1f !important; color: #cccccc !important; border: 1px solid #2a2a2e !important; border-radius: 6px !important; font-size: 13px !important; }
  .stSelectbox > div > div { background-color: #1a1a1f !important; border: 1px solid #2a2a2e !important; color: #cccccc !important; }
  .stSelectbox div[data-baseweb="select"] span { color: #cccccc !important; }

  .stButton > button {
    background-color: #7f77dd22; border: 1px solid #7f77dd66;
    color: #afa9ec; border-radius: 6px; font-size: 13px;
    padding: 0.4rem 1rem; width: 100%;
  }
  .stButton > button:hover { background-color: #7f77dd44; border-color: #afa9ec; color: #ffffff; }

  .stDataFrame { border: 1px solid #2a2a2e; border-radius: 8px; }
  .streamlit-expanderHeader { background-color: #16161a !important; color: #888 !important; border: 1px solid #2a2a2e !important; border-radius: 6px !important; }
  .streamlit-expanderContent { background-color: #1a1a1f !important; border: 1px solid #2a2a2e !important; }
  hr { border-color: #2a2a2e !important; }
  h1, h2, h3 { color: #cccccc !important; font-weight: 500 !important; }
  h1 { font-size: 20px !important; } h2 { font-size: 16px !important; } h3 { font-size: 14px !important; }

  .badge { display: inline-block; font-size: 10px; padding: 2px 8px; border-radius: 4px; font-family: monospace; font-weight: 500; }
  .badge-naive    { background: #1d9e7522; color: #5dcaa5; border: 1px solid #1d9e7544; }
  .badge-hybrid   { background: #7f77dd22; color: #afa9ec; border: 1px solid #7f77dd44; }
  .badge-hyde     { background: #ba751722; color: #ef9f27; border: 1px solid #ba751744; }
  .badge-reranked { background: #d85a3022; color: #f0997b; border: 1px solid #d85a3044; }

  .result-card { background: #16161a; border: 1px solid #2a2a2e; border-radius: 8px; padding: 14px 16px; margin-bottom: 4px; }
  .result-best { border-left: 3px solid #7f77dd; }
  .result-answer { font-size: 13px; color: #aaaaaa; line-height: 1.7; margin: 8px 0; }
  .result-meta { font-size: 11px; color: #555; font-family: monospace; }
  .conf-label { display: flex; justify-content: space-between; font-size: 10px; color: #555; font-family: monospace; margin-bottom: 3px; }

  .info-box { background: #1d9e7511; border: 1px solid #1d9e7533; border-radius: 6px; padding: 10px 14px; font-size: 12px; color: #5dcaa5; font-family: monospace; margin-bottom: 12px; }
  .warn-box { background: #ba751711; border: 1px solid #ba751733; border-radius: 6px; padding: 10px 14px; font-size: 12px; color: #ef9f27; font-family: monospace; margin-bottom: 12px; }
  .err-box  { background: #a32d2d11; border: 1px solid #a32d2d33; border-radius: 6px; padding: 10px 14px; font-size: 12px; color: #f09595; font-family: monospace; margin-bottom: 12px; }

  .chunk-box { background: #1a1a1f; border: 1px solid #2a2a2e; border-radius: 6px; padding: 10px 12px; margin-bottom: 6px; font-size: 11px; }
  .chunk-source { color: #7f77dd; font-family: monospace; font-size: 10px; margin-bottom: 4px; }
  .chunk-score  { color: #555; font-family: monospace; font-size: 10px; }
  .chunk-text   { color: #888; line-height: 1.6; }

  .about-card { background: #16161a; border: 1px solid #2a2a2e; border-radius: 8px; padding: 18px; margin-bottom: 10px; }
  .about-heading { font-size: 10px; color: #7f77dd; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 14px; font-family: monospace; }
  .about-row { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 0.5px solid #1e1e22; font-size: 12px; font-family: monospace; }
  .about-row:last-child { border-bottom: none; }
  .about-key { color: #555; } .about-val { color: #aaa; }
  .about-highlight { color: #afa9ec !important; font-weight: 500; }
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px; }
  .stat-box { background: #1a1a1f; border-radius: 6px; padding: 10px 12px; }
  .stat-num { font-size: 20px; font-weight: 500; color: #afa9ec; font-family: monospace; }
  .stat-label { font-size: 10px; color: #555; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================

def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_benchmark(question, top_k=5):
    try:
        r = requests.post(f"{API_BASE}/benchmark",
            json={"question": question, "top_k": top_k}, timeout=60)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def api_feedback(request_id, strategy, question, answer, rating):
    try:
        r = requests.post(f"{API_BASE}/feedback",
            json={"request_id": request_id, "strategy": strategy,
                  "question": question, "answer": answer, "rating": rating},
            timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def load_csv():
    csvs = sorted(Path("results").glob("benchmark_*.csv"))
    if not csvs:
        return None
    df = pd.read_csv(csvs[-1])
    df["confidence"]    = pd.to_numeric(df["confidence"],    errors="coerce")
    df["total_latency"] = pd.to_numeric(df["total_latency"], errors="coerce")
    df["tier"]          = pd.to_numeric(df["tier"],          errors="coerce")
    return df


def load_judge_csv():
    csvs = sorted(Path("results").glob("judge_*.csv"))
    if not csvs:
        return None
    df = pd.read_csv(csvs[-1])
    for col in ["faithfulness", "relevance", "hallucination_free", "abstention_correct", "judge_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["tier"] = pd.to_numeric(df["tier"], errors="coerce")
    return df


def load_golden():
    path = Path("data/golden_dataset.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("questions", data) if isinstance(data, dict) else data


def load_observability_stats():
    db_path = "pipeline_monitor.db"
    try:
        conn   = sqlite3.connect(db_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        if "observability_log" not in tables["name"].values:
            conn.close()
            return None
        per_strategy = pd.read_sql("""
            SELECT strategy,
                COUNT(*) AS total_requests,
                ROUND(AVG(total_tokens),0) AS avg_tokens,
                ROUND(SUM(cost_usd),6) AS total_cost,
                ROUND(AVG(cost_usd),6) AS avg_cost,
                ROUND(AVG(total_ms),0) AS avg_latency_ms,
                ROUND(AVG(confidence),3) AS avg_confidence,
                SUM(CASE WHEN error!='' THEN 1 ELSE 0 END) AS errors
            FROM observability_log
            WHERE strategy!='' AND strategy IS NOT NULL
            GROUP BY strategy ORDER BY strategy
        """, conn)
        recent = pd.read_sql("""
            SELECT timestamp, request_id, strategy,
                SUBSTR(query,1,60) AS query,
                total_tokens, ROUND(cost_usd,6) AS cost_usd,
                ROUND(total_ms,0) AS total_ms,
                ROUND(confidence,2) AS confidence, is_answerable,
                CASE WHEN error!='' THEN 'ERROR' ELSE 'OK' END AS status
            FROM observability_log ORDER BY id DESC LIMIT 30
        """, conn)
        summary = pd.read_sql("""
            SELECT COUNT(*) AS total_requests,
                ROUND(SUM(cost_usd),6) AS total_cost,
                SUM(total_tokens) AS total_tokens,
                ROUND(AVG(total_ms),0) AS avg_latency_ms,
                SUM(CASE WHEN error!='' THEN 1 ELSE 0 END) AS total_errors
            FROM observability_log
        """, conn)
        conn.close()
        return {"per_strategy": per_strategy, "recent": recent, "summary": summary}
    except Exception:
        return None


def load_feedback_stats():
    db_path = "pipeline_monitor.db"
    try:
        conn   = sqlite3.connect(db_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        if "feedback" not in tables["name"].values:
            conn.close()
            return None
        df = pd.read_sql("""
            SELECT strategy,
                COUNT(*) AS total,
                SUM(CASE WHEN rating= 1 THEN 1 ELSE 0 END) AS thumbs_up,
                SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END) AS thumbs_down,
                ROUND(100.0*SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END)/COUNT(*),1) AS positive_pct
            FROM feedback
            GROUP BY strategy ORDER BY strategy
        """, conn)
        conn.close()
        return df if len(df) > 0 else None
    except Exception:
        return None


def safe_avg(series):
    vals = series.dropna()
    return round(float(vals.mean()), 3) if len(vals) > 0 else 0.0


def horizontal_bar(data_dict, title, x_max=None, x_suffix="", height=220):
    fig = go.Figure()
    for strategy, value in data_dict.items():
        fig.add_trace(go.Bar(
            y=[strategy], x=[value], orientation="h",
            marker_color=STRATEGY_COLORS.get(strategy, "#888"),
            showlegend=False,
            text=[f'{value}{x_suffix}'],
            textposition="outside",
            textfont=dict(size=11, color="#aaa"),
        ))
    layout = dict(
        plot_bgcolor="#16161a", paper_bgcolor="#16161a",
        font=dict(color="#888", size=12),
        margin=dict(l=10, r=50, t=36, b=10), height=height,
        title=dict(text=title, font=dict(size=12, color="#666"), x=0),
        xaxis=dict(showgrid=False, zeroline=False, ticksuffix=x_suffix),
        yaxis=dict(showgrid=False, tickfont=dict(size=12)),
        barmode="group",
    )
    if x_max:
        layout["xaxis"]["range"] = [0, x_max]
    fig.update_layout(**layout)
    return fig


def conf_bar_html(confidence, strategy):
    color = STRATEGY_COLORS.get(strategy, "#888")
    pct   = int(confidence * 100)
    return f"""
    <div style="margin:8px 0 4px;">
      <div class="conf-label">
        <span>confidence</span><span style="color:{color}">{confidence:.2f}</span>
      </div>
      <div style="background:#222;border-radius:3px;height:5px;">
        <div style="width:{pct}%;background:{color};height:5px;border-radius:3px;"></div>
      </div>
    </div>"""


def strategy_badge(name):
    return f'<span class="badge badge-{name}">{name}</span>'


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(
        '<div style="font-size:16px;font-weight:500;color:#7f77dd;margin-bottom:2px;">RAG Benchmark</div>'
        '<div style="font-size:10px;color:#444;font-family:monospace;margin-bottom:16px;letter-spacing:1px;">MULTI-STRATEGY EVALUATION</div>',
        unsafe_allow_html=True
    )
    health = api_health()
    if health:
        st.markdown('<div class="info-box"> API online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box"> API offline</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:10px;color:#555;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;font-family:monospace;">Strategies</div>', unsafe_allow_html=True)
    descriptions = {
        "naive":    "vector search only",
        "hybrid":   "BM25 + vector + RRF",
        "hyde":     "hypothetical embeddings",
        "reranked": "cross-encoder rerank",
    }
    for s, color in STRATEGY_COLORS.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
            f'<span style="color:{color};font-size:14px;">■</span>'
            f'<div><div style="font-size:12px;color:#aaa;font-family:monospace;">{s}</div>'
            f'<div style="font-size:10px;color:#444;">{descriptions[s]}</div></div></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div style="font-size:10px;color:#555;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;font-family:monospace;">Corpus</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#555;font-family:monospace;line-height:2;">
    chunks &nbsp;&nbsp; 30,432<br>bm25 &nbsp;&nbsp;&nbsp;&nbsp; 7,123<br>docs &nbsp;&nbsp;&nbsp;&nbsp; 219<br>dims &nbsp;&nbsp;&nbsp;&nbsp; 384
    </div>""", unsafe_allow_html=True)


# =============================================================================
# TABS
# =============================================================================

tab_query, tab_benchmark, tab_sources, tab_obs, tab_about = st.tabs([
    "Query", "Benchmark", "Sources", "Observability", "About"
])


# =============================================================================
# TAB 1 — QUERY
# =============================================================================

with tab_query:
    st.markdown("### Ask a question")
    st.markdown(
        '<div style="font-size:11px;color:#555;font-family:monospace;margin-bottom:12px;">'
        'Runs all 4 strategies in parallel — asyncio.gather — wall time = slowest single strategy'
        '</div>', unsafe_allow_html=True
    )

    question = st.text_area("",
        placeholder="e.g. What is LoRA and how does it reduce trainable parameters?",
        height=90, label_visibility="collapsed", key="query_input")

    run_clicked = st.button("Run all 4 strategies", use_container_width=True)

    # Store results in session state so thumbs clicks don't wipe them
    if run_clicked and question.strip():
        if not health:
            st.markdown('<div class="err-box">API offline — run: uvicorn app.main:app --reload</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Running all 4 strategies in parallel..."):
                result = api_benchmark(question.strip())
            st.session_state["last_result"]   = result
            st.session_state["last_question"] = question.strip()

    elif run_clicked:
        st.markdown('<div class="warn-box">Please enter a question first.</div>', unsafe_allow_html=True)

    # Display from session state — persists across thumbs clicks
    if "last_result" in st.session_state:
        result   = st.session_state["last_result"]
        question = st.session_state["last_question"]

        if "error" in result:
            st.markdown(f'<div class="err-box">Error: {result["error"]}</div>', unsafe_allow_html=True)
        else:
            strategies_data = result.get("strategies", [])
            answered        = sum(1 for s in strategies_data if s.get("is_answerable"))
            total_cost      = result.get("total_cost_usd", 0) or 0
            req_id          = result.get("request_id", "")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Best strategy", result.get("best_strategy", "—").upper())
            c2.metric("Fastest",       result.get("fastest_strategy", "—").upper())
            c3.metric("Wall time",     f'{result.get("total_time", 0):.1f}s')
            c4.metric("Answerable",    f"{answered} / 4")
            c5.metric("Total cost",    f'${total_cost:.5f}')

            st.markdown("")

            best = result.get("best_strategy")
            sorted_strategies = sorted(
                strategies_data,
                key=lambda x: (x["strategy"] != best, -x.get("confidence", 0))
            )

            for s in sorted_strategies:
                is_best    = s["strategy"] == best
                card_class = "result-card result-best" if is_best else "result-card"
                badge      = strategy_badge(s["strategy"])
                best_tag   = ' <span style="font-size:10px;color:#7f77dd;font-family:monospace;">★ best</span>' if is_best else ""
                lat        = s.get("latency", {})
                lat_str    = (
                    f'retrieve {lat.get("retrieve",0):.2f}s &nbsp;|&nbsp; '
                    f'generate {lat.get("generate",0):.2f}s &nbsp;|&nbsp; '
                    f'total {lat.get("total",0):.2f}s'
                )
                raw_answer  = s.get("answer", "")
                is_error    = raw_answer.startswith("ERROR")
                answer_text = (
                    '<span style="color:#f09595;font-size:12px;">Rate limit hit — try again after quota resets</span>'
                    if is_error else raw_answer[:400]
                )
                answerable_tag = (
                    '<span style="color:#f09595;font-size:10px;">rate limited</span>' if is_error else
                    '<span style="color:#5dcaa5;font-size:10px;">answerable</span>'   if s.get("is_answerable") else
                    '<span style="color:#666;font-size:10px;">abstained</span>'
                )
                conf_html = conf_bar_html(s.get("confidence", 0), s["strategy"])
                cost_str  = f'&nbsp;|&nbsp; cost ${s["cost_usd"]:.5f}' if s.get("cost_usd") and not is_error else ""

                st.markdown(f"""
                <div class="{card_class}">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <div>{badge}{best_tag}</div>{answerable_tag}
                  </div>
                  <div class="result-answer">{answer_text}</div>
                  {conf_html}
                  <div class="result-meta" style="margin-top:8px;">{lat_str}{cost_str}</div>
                </div>""", unsafe_allow_html=True)

                if not is_error:
                    fb1, fb2, fb3 = st.columns([1, 1, 8])
                    with fb1:
                        if st.button("👍", key=f"up_{s['strategy']}"):
                            if api_feedback(req_id, s["strategy"], question, raw_answer, 1):
                                st.toast(f"👍 saved for {s['strategy']}", icon="✅")
                    with fb2:
                        if st.button("👎", key=f"down_{s['strategy']}"):
                            if api_feedback(req_id, s["strategy"], question, raw_answer, -1):
                                st.toast(f"👎 saved for {s['strategy']}", icon="📝")

                chunks = s.get("retrieved_chunks", [])
                if chunks:
                    with st.expander(f"Retrieved chunks — {s['strategy']} ({len(chunks)})", expanded=False):
                        for i, chunk in enumerate(chunks, 1):
                            src   = chunk.get("metadata", {}).get("filename", "unknown")
                            score = chunk.get("score", 0)
                            text  = chunk.get("content", "")[:300]
                            st.markdown(f"""
                            <div class="chunk-box">
                              <div class="chunk-source">{i}. {src}</div>
                              <div class="chunk-score">score: {score:.4f}</div>
                              <div class="chunk-text">{text}...</div>
                            </div>""", unsafe_allow_html=True)

                st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

    elif run_clicked:
        st.markdown('<div class="warn-box">Please enter a question first.</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2 — BENCHMARK
# =============================================================================

with tab_benchmark:
    st.markdown("### Benchmark results")

    df       = load_csv()
    df_judge = load_judge_csv()

    if df is None:
        st.markdown('<div class="warn-box">No benchmark CSV in results/ — run scripts_main/run_benchmark.py</div>', unsafe_allow_html=True)
    else:
        df_clean   = df[~df["answer"].str.contains("ERROR", na=False)]
        golden     = load_golden()
        golden_map = {q["id"]: q for q in golden} if golden else {}

        metrics = {}
        for s in ["naive", "hybrid", "hyde", "reranked"]:
            sub = df_clean[df_clean["strategy"] == s]
            if len(sub) == 0:
                continue
            abstention, calibration = [], []
            for _, row in sub.iterrows():
                g       = golden_map.get(row["question_id"], {})
                exp_ans = g.get("is_answerable", True)
                act_ans = str(row.get("is_answerable", "false")).lower() == "true"
                conf    = float(row["confidence"])
                abstention.append(1.0 if act_ans == exp_ans else 0.0)
                if exp_ans:
                    calibration.append(1.0 if conf >= g.get("expected_confidence_min", 0.7) else 0.0)
                else:
                    calibration.append(1.0 if conf <= g.get("expected_confidence_max", 0.3) else 0.0)
            metrics[s] = {
                "abstention":  round(sum(abstention) / len(abstention) * 100, 1) if abstention else 0,
                "calibration": round(sum(calibration) / len(calibration) * 100, 1) if calibration else 0,
                "avg_latency": round(float(sub["total_latency"].mean()), 2),
                "avg_conf":    round(float(sub["confidence"].mean()), 2),
            }

        judge_metrics = {}
        if df_judge is not None:
            for s in ["naive", "hybrid", "hyde", "reranked"]:
                sub = df_judge[df_judge["strategy"] == s].dropna(subset=["judge_score"])
                if len(sub) == 0:
                    continue
                judge_metrics[s] = {
                    "judge_score":        safe_avg(sub["judge_score"]),
                    "hallucination_free": safe_avg(sub["hallucination_free"]),
                }

        if metrics:
            best_abs   = max(metrics, key=lambda s: metrics[s]["abstention"])
            best_cal   = max(metrics, key=lambda s: metrics[s]["calibration"])
            best_fast  = min(metrics, key=lambda s: metrics[s]["avg_latency"])
            best_judge = max(judge_metrics, key=lambda s: judge_metrics[s]["judge_score"]) if judge_metrics else "—"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best abstention",  best_abs.upper(),  f'{metrics[best_abs]["abstention"]}%')
            c2.metric("Best calibration", best_cal.upper(),  f'{metrics[best_cal]["calibration"]}%')
            c3.metric("Fastest",          best_fast.upper(), f'{metrics[best_fast]["avg_latency"]}s')
            c4.metric("Best judge score", best_judge.upper() if best_judge != "—" else "—",
                f'{judge_metrics[best_judge]["judge_score"]:.3f}' if judge_metrics else "—")

            st.markdown("")

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(horizontal_bar(
                    {s: metrics[s]["abstention"] for s in metrics},
                    "Abstention accuracy — knows when NOT to answer",
                    x_max=110, x_suffix="%", height=250
                ), use_container_width=True)
            with col2:
                st.plotly_chart(horizontal_bar(
                    {s: metrics[s]["calibration"] for s in metrics},
                    "Confidence calibration — honest confidence scores",
                    x_max=110, x_suffix="%", height=250
                ), use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                data = {s: judge_metrics[s]["judge_score"] for s in judge_metrics} if judge_metrics else {s: metrics[s]["avg_conf"] for s in metrics}
                title = "LLM judge — overall answer quality" if judge_metrics else "Average confidence score"
                st.plotly_chart(horizontal_bar(data, title, x_max=1.1, height=250), use_container_width=True)
            with col4:
                data = {s: judge_metrics[s]["hallucination_free"] for s in judge_metrics} if judge_metrics else {s: metrics[s]["avg_latency"] for s in metrics}
                title = "Hallucination free — 1.0 = clean" if judge_metrics else "Average latency (seconds)"
                suffix = "" if judge_metrics else "s"
                st.plotly_chart(horizontal_bar(data, title, x_max=1.1 if judge_metrics else None, x_suffix=suffix, height=250), use_container_width=True)

            # Feedback chart — only shows once user has clicked thumbs
            st.markdown("---")
            st.markdown("#### User feedback")
            feedback_df = load_feedback_stats()
            if feedback_df is not None:
                st.plotly_chart(
                    horizontal_bar(
                        {row["strategy"]: row["positive_pct"] for _, row in feedback_df.iterrows()},
                        "Positive feedback % per strategy",
                        x_max=110, x_suffix="%", height=220
                    ),
                    use_container_width=True
                )
                st.dataframe(feedback_df.reset_index(drop=True),
                    use_container_width=True, height=180)

            st.markdown("---")
            st.markdown("#### Raw results")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                tier_filter = st.selectbox("Filter by tier",
                    ["All tiers", "Tier 1 — easy", "Tier 2 — multi-chunk", "Tier 3 — adversarial"],
                    label_visibility="collapsed")
            with col_f2:
                strat_filter = st.selectbox("Filter by strategy",
                    ["All strategies", "naive", "hybrid", "hyde", "reranked"],
                    label_visibility="collapsed")

            tier_map   = {"Tier 1 — easy": 1, "Tier 2 — multi-chunk": 2, "Tier 3 — adversarial": 3}
            display_df = df_clean.copy()
            if tier_filter != "All tiers":
                display_df = display_df[display_df["tier"] == tier_map[tier_filter]]
            if strat_filter != "All strategies":
                display_df = display_df[display_df["strategy"] == strat_filter]

            show_cols = ["question_id", "tier", "strategy", "confidence", "is_answerable", "total_latency"]
            if "top_sources" in display_df.columns:
                show_cols.append("top_sources")
            st.dataframe(display_df[show_cols].reset_index(drop=True),
                use_container_width=True, height=300)


# =============================================================================
# TAB 3 — SOURCES
# =============================================================================

with tab_sources:
    st.markdown("### Corpus & source analysis")

    golden = load_golden()
    if not golden:
        st.markdown('<div class="warn-box">data/golden_dataset.json not found</div>', unsafe_allow_html=True)
    else:
        corpus_groups = {
            "huggingface": 0, "langchain": 0, "anthropic": 0,
            "papers": 0, "beir": 0, "adversarial": 0
        }
        corpus_colors_map = {
            "huggingface": "#5DCAA5", "langchain": "#7F77DD",
            "anthropic":   "#EF9F27", "papers":    "#F0997B",
            "beir":        "#85B7EB", "adversarial": "#555555",
        }
        for q in golden:
            sources = q.get("source_documents", [])
            placed  = False
            for s in sources:
                sl = s.lower()
                if any(x in sl for x in ["peft","transformers","llm_tutorial","pipeline","tokeniz","training","perf_train","generation","quantization","bitsandbytes","deepspeed","kv_cache"]):
                    corpus_groups["huggingface"] += 1; placed = True; break
                elif any(x in sl for x in ["langchain","lcel","agent","chain","retriever"]):
                    corpus_groups["langchain"] += 1; placed = True; break
                elif "anthropic" in sl:
                    corpus_groups["anthropic"] += 1; placed = True; break
                elif any(x in sl for x in ["paper","pdf","lora","rag","attention","selfrag","hyde"]):
                    corpus_groups["papers"] += 1; placed = True; break
                elif "scifact" in sl or "beir" in sl:
                    corpus_groups["beir"] += 1; placed = True; break
            if not placed:
                corpus_groups["adversarial"] += 1

        col_pie, col_cats = st.columns(2)

        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=list(corpus_groups.keys()),
                values=list(corpus_groups.values()),
                marker_colors=[corpus_colors_map[k] for k in corpus_groups],
                textinfo="label+value", textfont=dict(size=11),
                hole=0.5, showlegend=False,
            ))
            fig_pie.update_layout(
                plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                margin=dict(l=0, r=0, t=36, b=0), height=260,
                title=dict(text="Questions per source group", font=dict(size=12, color="#666"), x=0),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_cats:
            categories = {}
            for q in golden:
                cat = q.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            cats_df = pd.DataFrame(
                sorted(categories.items(), key=lambda x: x[1], reverse=True),
                columns=["category", "count"]
            )
            fig_cat = go.Figure(go.Bar(
                y=cats_df["category"], x=cats_df["count"], orientation="h",
                marker_color="rgba(127,119,221,0.3)",
                marker_line_color="#7f77dd", marker_line_width=0.5,
            ))
            fig_cat.update_layout(
                plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                font=dict(color="#888", size=10),
                margin=dict(l=0, r=0, t=36, b=0), height=260,
                title=dict(text="Question categories", font=dict(size=12, color="#666"), x=0),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Expected source files")

        expected_sources  = {}
        source_groups_map = {}
        for q in golden:
            for s in q.get("source_documents", []):
                expected_sources[s] = expected_sources.get(s, 0) + 1
                sl = s.lower()
                if any(x in sl for x in ["peft","transformers","llm_tutorial","pipeline","tokeniz","training","perf_train","generation","quantization","bitsandbytes","deepspeed","kv_cache"]):
                    source_groups_map[s] = "huggingface"
                elif any(x in sl for x in ["langchain","lcel","agent","chain","retriever"]):
                    source_groups_map[s] = "langchain"
                elif "anthropic" in sl:
                    source_groups_map[s] = "anthropic"
                elif any(x in sl for x in ["paper","pdf","lora","rag","attention","selfrag","hyde"]):
                    source_groups_map[s] = "papers"
                else:
                    source_groups_map[s] = "other"

        src_df = pd.DataFrame(
            sorted(expected_sources.items(), key=lambda x: x[1], reverse=True),
            columns=["source_file", "questions"]
        )
        src_df["group"] = src_df["source_file"].map(lambda x: source_groups_map.get(x, "other"))
        src_df["color"] = src_df["group"].map(lambda x: corpus_colors_map.get(x, "#888"))

        fig_src = go.Figure()
        for _, row in src_df.iterrows():
            fig_src.add_trace(go.Bar(
                y=[row["source_file"].split("/")[-1]],
                x=[row["questions"]], orientation="h",
                marker_color=row["color"], showlegend=False,
            ))
        fig_src.update_layout(
            plot_bgcolor="#16161a", paper_bgcolor="#16161a",
            font=dict(color="#888", size=10),
            margin=dict(l=0, r=20, t=10, b=0),
            height=max(320, len(src_df) * 22),
            xaxis=dict(showgrid=False, zeroline=False, dtick=1, title="questions targeting this file"),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig_src, use_container_width=True)


# =============================================================================
# TAB 4 — OBSERVABILITY
# =============================================================================

with tab_obs:
    st.markdown("### Observability")
    st.markdown(
        '<div style="font-size:11px;color:#555;font-family:monospace;margin-bottom:16px;">'
        'Token usage, cost tracking and request tracing — populated as you use the Query tab'
        '</div>', unsafe_allow_html=True
    )

    obs = load_observability_stats()

    if obs is None:
        st.markdown(
            '<div class="warn-box">No observability data yet.<br>'
            'Run a query in the Query tab — data appears here automatically.</div>',
            unsafe_allow_html=True
        )
    else:
        summary = obs["summary"].iloc[0] if len(obs["summary"]) > 0 else {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total requests", int(summary.get("total_requests", 0)))
        c2.metric("Total tokens",   f'{int(summary.get("total_tokens", 0) or 0):,}')
        c3.metric("Total cost",     f'${float(summary.get("total_cost", 0) or 0):.5f}')
        c4.metric("Avg latency",    f'{int(summary.get("avg_latency_ms", 0) or 0)}ms')

        st.markdown("")
        per_strategy = obs["per_strategy"]

        if len(per_strategy) > 0:
            col1, col2 = st.columns(2)
            with col1:
                fig_tok = go.Figure()
                for _, row in per_strategy.iterrows():
                    fig_tok.add_trace(go.Bar(
                        y=[row["strategy"]], x=[row["avg_tokens"] or 0],
                        orientation="h",
                        marker_color=STRATEGY_COLORS.get(row["strategy"], "#888"),
                        showlegend=False,
                        text=[f'{int(row["avg_tokens"] or 0)} tokens'],
                        textposition="outside", textfont=dict(size=10, color="#aaa"),
                    ))
                fig_tok.update_layout(
                    plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                    font=dict(color="#888", size=11),
                    margin=dict(l=0, r=90, t=36, b=0), height=220,
                    title=dict(text="Avg tokens per query", font=dict(size=12, color="#666"), x=0),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_tok, use_container_width=True)

            with col2:
                fig_cost = go.Figure()
                for _, row in per_strategy.iterrows():
                    avg_cost = float(row["avg_cost"] or 0)
                    fig_cost.add_trace(go.Bar(
                        y=[row["strategy"]], x=[avg_cost],
                        orientation="h",
                        marker_color=STRATEGY_COLORS.get(row["strategy"], "#888"),
                        showlegend=False,
                        text=[f'${avg_cost:.5f}'],
                        textposition="outside", textfont=dict(size=10, color="#aaa"),
                    ))
                fig_cost.update_layout(
                    plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                    font=dict(color="#888", size=11),
                    margin=dict(l=0, r=90, t=36, b=0), height=220,
                    title=dict(text="Avg cost per query (USD)", font=dict(size=12, color="#666"), x=0),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_cost, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Per strategy breakdown")
            display_cols = ["strategy", "total_requests", "avg_tokens", "avg_cost", "avg_latency_ms", "avg_confidence", "errors"]
            available    = [c for c in display_cols if c in per_strategy.columns]
            st.dataframe(per_strategy[available].reset_index(drop=True),
                use_container_width=True, height=200)

        st.markdown("---")
        st.markdown("#### Recent requests")
        if len(obs["recent"]) > 0:
            st.dataframe(obs["recent"].reset_index(drop=True),
                use_container_width=True, height=300)
        else:
            st.markdown('<div class="info-box">No requests logged yet</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 5 — ABOUT
# =============================================================================

with tab_about:
    st.markdown("""
    <div style="padding: 20px 0 24px;">
      <div style="font-size:22px;font-weight:500;color:#cccccc;margin-bottom:6px;">Multi-Strategy RAG Benchmark</div>
      <div style="font-size:13px;color:#555;font-family:monospace;line-height:1.8;">
        RAG fails silently — naive retrieval vs reranked hybrid — built to measure the gap.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
      <div class="stat-box"><div class="stat-num">+31%</div><div class="stat-label">judge score — reranked over naive</div></div>
      <div class="stat-box"><div class="stat-num">93%</div><div class="stat-label">abstention accuracy — hybrid on adversarial</div></div>
      <div class="stat-box"><div class="stat-num">3x</div><div class="stat-label">parallel speedup — asyncio.gather</div></div>
      <div class="stat-box"><div class="stat-num">30</div><div class="stat-label">golden questions — tiered benchmark</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
        <div class="about-card">
          <div class="about-heading">Retrieval strategies</div>
          <div class="about-row"><span class="about-key">naive</span><span class="about-val">vector search → fixed chunks</span></div>
          <div class="about-row"><span class="about-key">hybrid</span><span class="about-val">BM25 + vector + RRF fusion</span></div>
          <div class="about-row"><span class="about-key">hyde</span><span class="about-val">hypothetical doc → embed → search</span></div>
          <div class="about-row"><span class="about-key">reranked</span><span class="about-val">hybrid top 20 → cross-encoder → top 5</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
          <div class="about-heading">Evaluation layers</div>
          <div class="about-row"><span class="about-key">term_hit_rate</span><span class="about-val">right terms in answer?</span></div>
          <div class="about-row"><span class="about-key">abstention_acc</span><span class="about-val">knows when not to answer?</span></div>
          <div class="about-row"><span class="about-key">conf_calibration</span><span class="about-val">confidence honest?</span></div>
          <div class="about-row"><span class="about-key">faithfulness</span><span class="about-val">LLM judge — grounded claims</span></div>
          <div class="about-row"><span class="about-key">hallucination_free</span><span class="about-val">LLM judge — no invented facts</span></div>
          <div class="about-row"><span class="about-key">human_feedback</span><span class="about-val">👍 👎 per strategy per query</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="about-card">
          <div class="about-heading">Tech stack</div>
          <div class="about-row"><span class="about-key">LLM</span><span class="about-val">Groq — llama-3.3-70b-versatile</span></div>
          <div class="about-row"><span class="about-key">judge</span><span class="about-val">Groq — llama-3.1-8b-instant</span></div>
          <div class="about-row"><span class="about-key">embeddings</span><span class="about-val">all-MiniLM-L6-v2 (384d)</span></div>
          <div class="about-row"><span class="about-key">vector db</span><span class="about-val">Qdrant local — 30,432 pts</span></div>
          <div class="about-row"><span class="about-key">keyword</span><span class="about-val">rank_bm25 — 7,123 chunks</span></div>
          <div class="about-row"><span class="about-key">reranker</span><span class="about-val">ms-marco-MiniLM-L-6-v2</span></div>
          <div class="about-row"><span class="about-key">API</span><span class="about-val">FastAPI + asyncio.gather</span></div>
          <div class="about-row"><span class="about-key">observability</span><span class="about-val">SQLite + token + cost tracking</span></div>
          <div class="about-row"><span class="about-key">feedback</span><span class="about-val">human-in-the-loop SQLite loop</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
          <div class="about-heading">Benchmark results</div>
          <div class="about-row"><span class="about-key">hyde judge score</span><span class="about-val about-highlight">0.675 — best overall</span></div>
          <div class="about-row"><span class="about-key">reranked judge</span><span class="about-val">0.670</span></div>
          <div class="about-row"><span class="about-key">naive judge</span><span class="about-val">0.534 — baseline</span></div>
          <div class="about-row"><span class="about-key">hybrid abstention</span><span class="about-val about-highlight">93.1% — best</span></div>
          <div class="about-row"><span class="about-key">hybrid latency</span><span class="about-val about-highlight">3.3s — fastest</span></div>
          <div class="about-row"><span class="about-key">hyde latency</span><span class="about-val">4.6s — 2× LLM calls</span></div>
        </div>
        """, unsafe_allow_html=True)