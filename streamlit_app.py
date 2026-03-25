import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json
import sqlite3
import os
from pathlib import Path


API_BASE = os.environ.get("API_BASE", "http://localhost:8000/api/v1")

STRATEGY_COLORS = {
    "naive":    "#1D9E75",
    "hybrid":   "#7F77DD",
    "hyde":     "#EF9F27",
    "reranked": "#F0997B",
}

STRATEGY_DESCRIPTIONS = {
    "naive":    "Vector search baseline — embed query, find nearest chunks, generate",
    "hybrid":   "BM25 + embeddings fused via Reciprocal Rank Fusion",
    "hyde":     "Hypothetical answer-driven retrieval — 2 LLM calls",
    "reranked": "Cross-encoder reranking over hybrid candidates",
}

DEFAULT_QUERY = "What is LoRA and how does it reduce trainable parameters?"

st.set_page_config(page_title="RAG Benchmark", layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# GLOBAL STYLES
# =============================================================================

st.markdown("""
<style>
  /* --- Base --------------------------------------------------- */
  .stApp { background-color: #0e0e10; color: #d0d0d0; }
  .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px; }
  hr { border-color: #2a2a2e !important; }

  /* --- Typography --------------------------------------------- */
  h1 { font-size: 28px !important; color: #e0e0e0 !important; font-weight: 600 !important; }
  h2 { font-size: 20px !important; color: #d0d0d0 !important; font-weight: 500 !important; }
  h3 { font-size: 17px !important; color: #cccccc !important; font-weight: 500 !important; }
  p, li, .stMarkdown { font-size: 14px !important; line-height: 1.7 !important; }

  /* --- Sidebar ------------------------------------------------ */
  section[data-testid="stSidebar"] { background-color: #131316; border-right: 1px solid #222228; }
  section[data-testid="stSidebar"] * { color: #999 !important; }

  /* --- Tabs --------------------------------------------------- */
  .stTabs [data-baseweb="tab-list"] {
    background-color: transparent; border-bottom: 1px solid #2a2a2e;
    padding: 0; gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
    background-color: transparent; color: #666; border-radius: 0;
    font-size: 14px; font-weight: 500; padding: 12px 24px;
    border-bottom: 2px solid transparent;
  }
  .stTabs [aria-selected="true"] {
    background-color: transparent !important;
    color: #afa9ec !important;
    border-bottom: 2px solid #7f77dd !important;
  }

  /* --- Metrics ------------------------------------------------ */
  div[data-testid="stMetric"] {
    background-color: #16161a; border: 1px solid #2a2a2e;
    border-radius: 10px; padding: 1.1rem 1.2rem;
  }
  div[data-testid="stMetric"] label { color: #777 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.5px; }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #afa9ec !important; font-size: 24px !important; font-weight: 600 !important; }
  div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 12px !important; }

  /* --- Inputs ------------------------------------------------- */
  .stTextArea textarea {
    background-color: #1a1a1f !important; color: #d0d0d0 !important;
    border: 1px solid #2a2a2e !important; border-radius: 8px !important;
    font-size: 14px !important; line-height: 1.6 !important;
    padding: 14px !important;
  }
  .stSelectbox > div > div { background-color: #1a1a1f !important; border: 1px solid #2a2a2e !important; color: #d0d0d0 !important; }
  .stSelectbox div[data-baseweb="select"] span { color: #d0d0d0 !important; }

  /* --- Buttons ------------------------------------------------ */
  .stButton > button {
    background: linear-gradient(135deg, #7f77dd33, #7f77dd11);
    border: 1px solid #7f77dd55; color: #afa9ec;
    border-radius: 8px; font-size: 14px; font-weight: 500;
    padding: 0.6rem 1.5rem; width: 100%;
    transition: all 0.2s ease;
  }
  .stButton > button:hover { background: linear-gradient(135deg, #7f77dd55, #7f77dd22); border-color: #afa9ec; color: #fff; }

  /* --- Data --------------------------------------------------- */
  .stDataFrame { border: 1px solid #2a2a2e; border-radius: 10px; }
  .streamlit-expanderHeader { background-color: #16161a !important; color: #999 !important; border: 1px solid #2a2a2e !important; border-radius: 8px !important; font-size: 13px !important; }
  .streamlit-expanderContent { background-color: #1a1a1f !important; border: 1px solid #2a2a2e !important; }

  /* --- Custom elements ---------------------------------------- */
  .badge { display: inline-block; font-size: 11px; padding: 3px 10px; border-radius: 5px; font-family: monospace; font-weight: 600; letter-spacing: 0.3px; }
  .badge-naive    { background: #1d9e7522; color: #5dcaa5; border: 1px solid #1d9e7544; }
  .badge-hybrid   { background: #7f77dd22; color: #afa9ec; border: 1px solid #7f77dd44; }
  .badge-hyde     { background: #ba751722; color: #ef9f27; border: 1px solid #ba751744; }
  .badge-reranked { background: #d85a3022; color: #f0997b; border: 1px solid #d85a3044; }

  .result-card { background: #16161a; border: 1px solid #2a2a2e; border-radius: 10px; padding: 18px 20px; margin-bottom: 8px; }
  .result-best { border-left: 3px solid #7f77dd; }
  .result-answer { font-size: 14px; color: #b0b0b0; line-height: 1.8; margin: 10px 0; }
  .result-meta { font-size: 12px; color: #666; font-family: monospace; }

  .conf-label { display: flex; justify-content: space-between; font-size: 11px; color: #666; font-family: monospace; margin-bottom: 4px; }

  .info-box { background: #1d9e7511; border: 1px solid #1d9e7533; border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #5dcaa5; font-family: monospace; margin-bottom: 14px; }
  .warn-box { background: #ba751711; border: 1px solid #ba751733; border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #ef9f27; font-family: monospace; margin-bottom: 14px; }
  .err-box  { background: #a32d2d11; border: 1px solid #a32d2d33; border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #f09595; font-family: monospace; margin-bottom: 14px; }

  .chunk-box { background: #1a1a1f; border: 1px solid #2a2a2e; border-radius: 8px; padding: 12px 14px; margin-bottom: 8px; }
  .chunk-source { color: #7f77dd; font-family: monospace; font-size: 11px; margin-bottom: 4px; }
  .chunk-score  { color: #666; font-family: monospace; font-size: 11px; }
  .chunk-text   { color: #999; line-height: 1.7; font-size: 12px; }

  .about-card { background: #16161a; border: 1px solid #2a2a2e; border-radius: 10px; padding: 22px; margin-bottom: 12px; }
  .about-heading { font-size: 11px; color: #7f77dd; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 16px; font-family: monospace; font-weight: 600; }
  .about-row { display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 0.5px solid #1e1e22; font-size: 13px; font-family: monospace; }
  .about-row:last-child { border-bottom: none; }
  .about-key { color: #666; } .about-val { color: #bbb; }
  .about-highlight { color: #afa9ec !important; font-weight: 600; }

  .section-label { font-size: 11px; color: #666; letter-spacing: 1.5px; text-transform: uppercase; font-family: monospace; font-weight: 600; margin-bottom: 8px; }
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
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "tier" in df.columns:
        df["tier"] = pd.to_numeric(df["tier"], errors="coerce")
    return df


def load_golden():
    path = Path("data/golden_dataset.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("questions", data) if isinstance(data, dict) else data


def load_observability_stats():
    db_path = os.environ.get("DB_PATH", "pipeline_monitor.db")
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
    db_path = os.environ.get("DB_PATH", "pipeline_monitor.db")
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


def horizontal_bar(data_dict, title, x_max=None, x_suffix="", height=240):
    fig = go.Figure()
    for strategy, value in data_dict.items():
        fig.add_trace(go.Bar(
            y=[strategy], x=[value], orientation="h",
            marker_color=STRATEGY_COLORS.get(strategy, "#888"),
            showlegend=False,
            text=[f'{value}{x_suffix}'],
            textposition="outside",
            textfont=dict(size=12, color="#aaa"),
        ))
    layout = dict(
        plot_bgcolor="#16161a", paper_bgcolor="#16161a",
        font=dict(color="#999", size=13),
        margin=dict(l=10, r=60, t=40, b=10), height=height,
        title=dict(text=title, font=dict(size=13, color="#888"), x=0),
        xaxis=dict(showgrid=False, zeroline=False, ticksuffix=x_suffix),
        yaxis=dict(showgrid=False, tickfont=dict(size=13)),
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
    <div style="margin:10px 0 6px;">
      <div class="conf-label">
        <span>confidence</span><span style="color:{color}">{confidence:.2f}</span>
      </div>
      <div style="background:#222;border-radius:4px;height:6px;">
        <div style="width:{pct}%;background:{color};height:6px;border-radius:4px;"></div>
      </div>
    </div>"""


def strategy_badge(name):
    return f'<span class="badge badge-{name}">{name}</span>'


# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div style="text-align:center; padding: 20px 0 10px;">
  <div style="font-size:32px; font-weight:700; color:#e0e0e0; letter-spacing:-0.5px;">
    RAG Benchmark
  </div>
  <div style="font-size:12px; color:#7f77dd; letter-spacing:3px; text-transform:uppercase; font-family:monospace; font-weight:600; margin-top:4px;">
    Multi-Strategy Evaluation
  </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 12px;">
      <div style="font-size:11px; color:#7f77dd; letter-spacing:2px; text-transform:uppercase; font-family:monospace; font-weight:600;">Control Panel</div>
    </div>
    """, unsafe_allow_html=True)

    health = api_health()
    if health:
        st.markdown('<div class="info-box">API connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">API offline</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-label">Strategies</div>', unsafe_allow_html=True)
    for s, color in STRATEGY_COLORS.items():
        desc = STRATEGY_DESCRIPTIONS[s]
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; gap:10px; padding:8px 0; border-bottom:1px solid #1e1e22;">
          <span style="color:{color}; font-size:16px; margin-top:1px;">&#9679;</span>
          <div>
            <div style="font-size:13px; color:#ccc; font-family:monospace; font-weight:500;">{s}</div>
            <div style="font-size:11px; color:#555; line-height:1.5; margin-top:2px;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-label">Corpus Stats</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px; color:#777; font-family:monospace; line-height:2.2;">
      <div style="display:flex; justify-content:space-between;"><span>vectors</span><span style="color:#afa9ec;">30,432</span></div>
      <div style="display:flex; justify-content:space-between;"><span>BM25 chunks</span><span style="color:#afa9ec;">7,123</span></div>
      <div style="display:flex; justify-content:space-between;"><span>documents</span><span style="color:#afa9ec;">219</span></div>
      <div style="display:flex; justify-content:space-between;"><span>dimensions</span><span style="color:#afa9ec;">384</span></div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TABS
# =============================================================================

tab_query, tab_benchmark, tab_sources, tab_obs, tab_about = st.tabs([
    "  Query  ", "  Benchmark  ", "  Sources  ", "  Observability  ", "  About  "
])


# =============================================================================
# TAB 1 --- QUERY
# =============================================================================

with tab_query:
    st.markdown("")
    st.markdown("## Ask a question")
    st.markdown(
        '<div style="font-size:13px; color:#666; font-family:monospace; margin-bottom:16px;">'
        'All 4 strategies run in parallel via asyncio.gather --- wall time = slowest strategy'
        '</div>', unsafe_allow_html=True
    )

    question = st.text_area("Question",
        value=DEFAULT_QUERY,
        height=80, label_visibility="collapsed", key="query_input")

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Run all 4 strategies", use_container_width=True)

    if run_clicked and question.strip():
        if not health:
            st.markdown('<div class="err-box">API offline --- start the backend first</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Running all 4 strategies in parallel..."):
                result = api_benchmark(question.strip())
            st.session_state["last_result"]   = result
            st.session_state["last_question"] = question.strip()
    elif run_clicked:
        st.markdown('<div class="warn-box">Enter a question first.</div>', unsafe_allow_html=True)

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

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Best", result.get("best_strategy", "---").upper())
            c2.metric("Fastest", result.get("fastest_strategy", "---").upper())
            c3.metric("Wall time", f'{result.get("total_time", 0):.1f}s')
            c4.metric("Answerable", f"{answered}/4")
            c5.metric("Total cost", f'${total_cost:.5f}')

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

            best = result.get("best_strategy")
            sorted_strategies = sorted(
                strategies_data,
                key=lambda x: (x["strategy"] != best, -x.get("confidence", 0))
            )

            for s in sorted_strategies:
                is_best    = s["strategy"] == best
                card_class = "result-card result-best" if is_best else "result-card"
                badge      = strategy_badge(s["strategy"])
                best_tag   = ' <span style="font-size:11px;color:#7f77dd;font-family:monospace;font-weight:600;">&#9733; BEST</span>' if is_best else ""
                lat        = s.get("latency", {})
                lat_str    = (
                    f'retrieve {lat.get("retrieve",0):.2f}s  &middot;  '
                    f'generate {lat.get("generate",0):.2f}s  &middot;  '
                    f'total {lat.get("total",0):.2f}s'
                )
                raw_answer  = s.get("answer", "")
                is_error    = raw_answer.startswith("ERROR")
                answer_text = (
                    '<span style="color:#f09595;">Rate limit --- try again after quota resets</span>'
                    if is_error else raw_answer[:500]
                )
                answerable_tag = (
                    '<span style="color:#f09595;font-size:11px;">rate limited</span>' if is_error else
                    '<span style="color:#5dcaa5;font-size:11px;font-weight:500;">&#9679; answerable</span>'   if s.get("is_answerable") else
                    '<span style="color:#666;font-size:11px;">&#9675; abstained</span>'
                )
                conf_html = conf_bar_html(s.get("confidence", 0), s["strategy"])
                cost_str  = f'  &middot;  ${s["cost_usd"]:.5f}' if s.get("cost_usd") and not is_error else ""

                st.markdown(f"""
                <div class="{card_class}">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <div>{badge}{best_tag}</div>{answerable_tag}
                  </div>
                  <div class="result-answer">{answer_text}</div>
                  {conf_html}
                  <div class="result-meta" style="margin-top:10px;">{lat_str}{cost_str}</div>
                </div>""", unsafe_allow_html=True)

                if not is_error:
                    fb1, fb2, fb3 = st.columns([1, 1, 8])
                    with fb1:
                        if st.button("\U0001f44d", key=f"up_{s['strategy']}"):
                            if api_feedback(req_id, s["strategy"], question, raw_answer, 1):
                                st.toast(f"\U0001f44d saved for {s['strategy']}", icon="\u2705")
                    with fb2:
                        if st.button("\U0001f44e", key=f"down_{s['strategy']}"):
                            if api_feedback(req_id, s["strategy"], question, raw_answer, -1):
                                st.toast(f"\U0001f44e saved for {s['strategy']}", icon="\U0001f4dd")

                chunks = s.get("retrieved_chunks", [])
                if chunks:
                    with st.expander(f"Retrieved chunks --- {s['strategy']} ({len(chunks)})", expanded=False):
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

                st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)


# =============================================================================
# TAB 2 --- BENCHMARK
# =============================================================================

with tab_benchmark:
    st.markdown("")
    st.markdown("## Benchmark Results")
    st.markdown(
        '<div style="font-size:13px; color:#666; font-family:monospace; margin-bottom:20px;">'
        '30 golden questions &middot; 3 difficulty tiers &middot; 4 strategies'
        '</div>', unsafe_allow_html=True
    )

    df       = load_csv()
    df_judge = load_judge_csv()

    if df is None:
        st.markdown('<div class="warn-box">No benchmark CSV in results/ --- run: python3 scripts/run_benchmark.py</div>', unsafe_allow_html=True)
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
                sub = df_judge[df_judge["strategy"] == s]
                if "judge_score" in sub.columns:
                    sub = sub.dropna(subset=["judge_score"])
                if len(sub) == 0:
                    continue
                judge_metrics[s] = {
                    "judge_score":        safe_avg(sub["judge_score"]),
                    "hallucination_free": safe_avg(sub["hallucination_free"]) if "hallucination_free" in sub.columns else 0,
                }

        if metrics:
            best_abs   = max(metrics, key=lambda s: metrics[s]["abstention"])
            best_cal   = max(metrics, key=lambda s: metrics[s]["calibration"])
            best_fast  = min(metrics, key=lambda s: metrics[s]["avg_latency"])

            has_judge  = bool(judge_metrics)
            if has_judge:
                best_judge       = max(judge_metrics, key=lambda s: judge_metrics[s]["judge_score"])
                judge_label      = "Best Judge Score"
                judge_value      = best_judge.upper()
                judge_delta      = f'{judge_metrics[best_judge]["judge_score"]:.3f}'
            else:
                best_conf_strat  = max(metrics, key=lambda s: metrics[s]["avg_conf"])
                judge_label      = "Best Avg Confidence"
                judge_value      = best_conf_strat.upper()
                judge_delta      = f'{metrics[best_conf_strat]["avg_conf"]:.2f}'

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best Abstention",  best_abs.upper(),  f'{metrics[best_abs]["abstention"]}%')
            c2.metric("Best Calibration", best_cal.upper(),  f'{metrics[best_cal]["calibration"]}%')
            c3.metric("Fastest",          best_fast.upper(), f'{metrics[best_fast]["avg_latency"]}s')
            c4.metric(judge_label,         judge_value,       judge_delta)

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(horizontal_bar(
                    {s: metrics[s]["abstention"] for s in metrics},
                    "Abstention Accuracy --- knows when NOT to answer",
                    x_max=110, x_suffix="%"
                ), use_container_width=True)
            with col2:
                st.plotly_chart(horizontal_bar(
                    {s: metrics[s]["calibration"] for s in metrics},
                    "Confidence Calibration --- honest confidence scores",
                    x_max=110, x_suffix="%"
                ), use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                if has_judge:
                    data  = {s: judge_metrics[s]["judge_score"] for s in judge_metrics}
                    title = "LLM Judge --- overall answer quality"
                else:
                    data  = {s: metrics[s]["avg_conf"] for s in metrics}
                    title = "Average Confidence"
                st.plotly_chart(horizontal_bar(data, title, x_max=1.1), use_container_width=True)
            with col4:
                if has_judge:
                    data  = {s: judge_metrics[s]["hallucination_free"] for s in judge_metrics}
                    title = "Hallucination Free --- 1.0 = clean"
                else:
                    data  = {s: metrics[s]["avg_latency"] for s in metrics}
                    title = "Average Latency (seconds)"
                suffix = "" if has_judge else "s"
                st.plotly_chart(horizontal_bar(data, title, x_max=1.1 if has_judge else None, x_suffix=suffix), use_container_width=True)

            if not has_judge:
                st.markdown(
                    '<div class="info-box">Judge scores unavailable --- run: python3 -m app.evaluation.judge</div>',
                    unsafe_allow_html=True
                )

            # Feedback
            st.markdown("---")
            st.markdown("### User Feedback")
            feedback_df = load_feedback_stats()
            if feedback_df is not None:
                st.plotly_chart(
                    horizontal_bar(
                        {row["strategy"]: row["positive_pct"] for _, row in feedback_df.iterrows()},
                        "Positive feedback % per strategy",
                        x_max=110, x_suffix="%"
                    ),
                    use_container_width=True
                )
                st.dataframe(feedback_df.reset_index(drop=True), use_container_width=True, height=180)
            else:
                st.markdown(
                    '<div style="font-size:13px; color:#555; font-family:monospace;">No feedback yet --- use the thumbs in the Query tab</div>',
                    unsafe_allow_html=True
                )

            # Raw results
            st.markdown("---")
            st.markdown("### Raw Results")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                tier_filter = st.selectbox("Filter by tier",
                    ["All tiers", "Tier 1 --- easy", "Tier 2 --- multi-chunk", "Tier 3 --- adversarial"],
                    label_visibility="collapsed")
            with col_f2:
                strat_filter = st.selectbox("Filter by strategy",
                    ["All strategies", "naive", "hybrid", "hyde", "reranked"],
                    label_visibility="collapsed")

            tier_map   = {"Tier 1 --- easy": 1, "Tier 2 --- multi-chunk": 2, "Tier 3 --- adversarial": 3}
            display_df = df_clean.copy()
            if tier_filter != "All tiers":
                display_df = display_df[display_df["tier"] == tier_map[tier_filter]]
            if strat_filter != "All strategies":
                display_df = display_df[display_df["strategy"] == strat_filter]

            show_cols = ["question_id", "tier", "strategy", "confidence", "is_answerable", "total_latency"]
            if "top_sources" in display_df.columns:
                show_cols.append("top_sources")
            st.dataframe(display_df[show_cols].reset_index(drop=True),
                use_container_width=True, height=320)


# =============================================================================
# TAB 3 --- SOURCES
# =============================================================================

with tab_sources:
    st.markdown("")
    st.markdown("## Corpus & Source Analysis")
    st.markdown(
        '<div style="font-size:13px; color:#666; font-family:monospace; margin-bottom:20px;">'
        'How the 30 golden questions map to source documents and categories'
        '</div>', unsafe_allow_html=True
    )

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
            st.markdown('<div class="section-label">Corpus Distribution</div>', unsafe_allow_html=True)
            fig_pie = go.Figure(go.Pie(
                labels=list(corpus_groups.keys()),
                values=list(corpus_groups.values()),
                marker_colors=[corpus_colors_map[k] for k in corpus_groups],
                textinfo="label+value", textfont=dict(size=12),
                hole=0.5, showlegend=False,
            ))
            fig_pie.update_layout(
                plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                margin=dict(l=0, r=0, t=10, b=0), height=280,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_cats:
            st.markdown('<div class="section-label">Question Categories</div>', unsafe_allow_html=True)
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
                font=dict(color="#999", size=11),
                margin=dict(l=0, r=0, t=10, b=0), height=280,
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-label">Source Coverage</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:13px; color:#666; font-family:monospace; margin-bottom:12px;">'
            'Which source documents are targeted by the golden questions'
            '</div>', unsafe_allow_html=True
        )

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
            font=dict(color="#999", size=11),
            margin=dict(l=0, r=20, t=10, b=0),
            height=max(340, len(src_df) * 24),
            xaxis=dict(showgrid=False, zeroline=False, dtick=1, title="questions targeting this file"),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig_src, use_container_width=True)


# =============================================================================
# TAB 4 --- OBSERVABILITY
# =============================================================================

with tab_obs:
    st.markdown("")
    st.markdown("## Observability")
    st.markdown(
        '<div style="font-size:13px; color:#666; font-family:monospace; margin-bottom:20px;">'
        'Token usage, cost tracking, and request tracing --- populated as you query'
        '</div>', unsafe_allow_html=True
    )

    obs = load_observability_stats()

    if obs is None:
        st.markdown(
            '<div class="warn-box">No observability data yet.<br>'
            'Run a query in the Query tab --- data appears here automatically.</div>',
            unsafe_allow_html=True
        )
    else:
        summary = obs["summary"].iloc[0] if len(obs["summary"]) > 0 else {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Requests", int(summary.get("total_requests", 0)))
        c2.metric("Total Tokens",   f'{int(summary.get("total_tokens", 0) or 0):,}')
        c3.metric("Total Cost",     f'${float(summary.get("total_cost", 0) or 0):.5f}')
        c4.metric("Avg Latency",    f'{int(summary.get("avg_latency_ms", 0) or 0)}ms')

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
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
                        text=[f'{int(row["avg_tokens"] or 0)}'],
                        textposition="outside", textfont=dict(size=11, color="#aaa"),
                    ))
                fig_tok.update_layout(
                    plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                    font=dict(color="#999", size=12),
                    margin=dict(l=0, r=80, t=40, b=0), height=240,
                    title=dict(text="Avg Tokens per Query", font=dict(size=13, color="#888"), x=0),
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
                        textposition="outside", textfont=dict(size=11, color="#aaa"),
                    ))
                fig_cost.update_layout(
                    plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                    font=dict(color="#999", size=12),
                    margin=dict(l=0, r=80, t=40, b=0), height=240,
                    title=dict(text="Avg Cost per Query (USD)", font=dict(size=13, color="#888"), x=0),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_cost, use_container_width=True)

            st.markdown("---")
            st.markdown("### Per-Strategy Breakdown")
            display_cols = ["strategy", "total_requests", "avg_tokens", "avg_cost", "avg_latency_ms", "avg_confidence", "errors"]
            available    = [c for c in display_cols if c in per_strategy.columns]
            st.dataframe(per_strategy[available].reset_index(drop=True),
                use_container_width=True, height=200)

        st.markdown("---")
        st.markdown("### Recent Requests")
        if len(obs["recent"]) > 0:
            st.dataframe(obs["recent"].reset_index(drop=True),
                use_container_width=True, height=320)
        else:
            st.markdown('<div class="info-box">No requests logged yet</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 5 --- ABOUT
# =============================================================================

with tab_about:
    st.markdown("")
    st.markdown("""
    <div style="text-align:center; padding: 24px 0 32px;">
      <div style="font-size:28px; font-weight:700; color:#e0e0e0;">Multi-Strategy RAG Benchmark</div>
      <div style="font-size:14px; color:#666; font-family:monospace; line-height:1.8; margin-top:8px; max-width:600px; margin-left:auto; margin-right:auto;">
        RAG pipelines fail silently. This system benchmarks four retrieval strategies
        head-to-head --- measuring hallucination, abstention accuracy, and answer quality
        across an adversarial golden dataset.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
        <div class="about-card">
          <div class="about-heading">Architecture</div>
          <div class="about-row"><span class="about-key">ingestion</span><span class="about-val">PDF / MD / JSONL &rarr; chunk &rarr; embed &rarr; Qdrant</span></div>
          <div class="about-row"><span class="about-key">retrieval</span><span class="about-val">4 strategies via asyncio.gather in parallel</span></div>
          <div class="about-row"><span class="about-key">generation</span><span class="about-val">Instructor + Pydantic &rarr; structured JSON</span></div>
          <div class="about-row"><span class="about-key">evaluation</span><span class="about-val">structural metrics + LLM-as-judge</span></div>
          <div class="about-row"><span class="about-key">feedback</span><span class="about-val">human thumbs per strategy per query</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
          <div class="about-heading">Evaluation Approach</div>
          <div class="about-row"><span class="about-key">abstention_acc</span><span class="about-val">correctly refuses unanswerable questions?</span></div>
          <div class="about-row"><span class="about-key">conf_calibration</span><span class="about-val">confidence scores match actual correctness?</span></div>
          <div class="about-row"><span class="about-key">faithfulness</span><span class="about-val">LLM judge --- claims grounded in context?</span></div>
          <div class="about-row"><span class="about-key">hallucination_free</span><span class="about-val">LLM judge --- no invented facts?</span></div>
          <div class="about-row"><span class="about-key">term_hit_rate</span><span class="about-val">required technical terms present?</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="about-card">
          <div class="about-heading">Tech Stack</div>
          <div class="about-row"><span class="about-key">LLM</span><span class="about-val">Groq --- qwen/qwen3-32b</span></div>
          <div class="about-row"><span class="about-key">judge</span><span class="about-val">Groq --- llama-3.1-8b-instant</span></div>
          <div class="about-row"><span class="about-key">embeddings</span><span class="about-val">all-MiniLM-L6-v2 &middot; 384d &middot; local</span></div>
          <div class="about-row"><span class="about-key">vector db</span><span class="about-val">Qdrant --- local or Docker server mode</span></div>
          <div class="about-row"><span class="about-key">keyword</span><span class="about-val">rank_bm25</span></div>
          <div class="about-row"><span class="about-key">reranker</span><span class="about-val">ms-marco-MiniLM-L-6-v2 &middot; local</span></div>
          <div class="about-row"><span class="about-key">API</span><span class="about-val">FastAPI + asyncio</span></div>
          <div class="about-row"><span class="about-key">frontend</span><span class="about-val">Streamlit + Plotly</span></div>
          <div class="about-row"><span class="about-key">observability</span><span class="about-val">SQLite &middot; token + cost tracking &middot; UUID tracing</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
          <div class="about-heading">Key Insights</div>
          <div class="about-row"><span class="about-key">naive hallucination</span><span class="about-val">fails 2/3 factual grounding checks</span></div>
          <div class="about-row"><span class="about-key">reranked vs naive</span><span class="about-val about-highlight">+26% judge score improvement</span></div>
          <div class="about-row"><span class="about-key">hybrid abstention</span><span class="about-val about-highlight">93% --- best at refusing bad questions</span></div>
          <div class="about-row"><span class="about-key">hyde</span><span class="about-val about-highlight">0.675 --- highest overall judge score</span></div>
          <div class="about-row"><span class="about-key">cost per query</span><span class="about-val">$0.0004 -- $0.0009 depending on strategy</span></div>
        </div>
        """, unsafe_allow_html=True)
