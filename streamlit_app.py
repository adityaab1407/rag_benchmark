import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path


# =============================================================================
# CONFIG
# =============================================================================

API_BASE = "http://localhost:8000/api/v1"

STRATEGY_COLORS = {
    "naive":    "#1D9E75",
    "hybrid":   "#7F77DD",
    "hyde":     "#EF9F27",
    "reranked": "#F0997B",
}

st.set_page_config(
    page_title="RAG Benchmark",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# DARK THEME CSS
# =============================================================================

st.markdown("""
<style>
  /* Global dark background */
  .stApp { background-color: #0e0e10; color: #cccccc; }
  section[data-testid="stSidebar"] { background-color: #16161a; border-right: 1px solid #2a2a2e; }
  section[data-testid="stSidebar"] * { color: #aaaaaa !important; }

  /* Main content */
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background-color: #16161a; border-radius: 8px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { background-color: transparent; color: #666; border-radius: 6px; font-size: 13px; }
  .stTabs [aria-selected="true"] { background-color: #1e1e2e !important; color: #afa9ec !important; }

  /* Cards / containers */
  div[data-testid="stMetric"] { background-color: #16161a; border: 1px solid #2a2a2e; border-radius: 8px; padding: 1rem; }
  div[data-testid="stMetric"] label { color: #666 !important; font-size: 12px !important; }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #afa9ec !important; font-size: 22px !important; }
  div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 11px !important; }

  /* Inputs */
  .stTextArea textarea { background-color: #1a1a1f !important; color: #cccccc !important; border: 1px solid #2a2a2e !important; border-radius: 6px !important; font-size: 13px !important; }
  .stTextInput input { background-color: #1a1a1f !important; color: #cccccc !important; border: 1px solid #2a2a2e !important; }
  .stSelectbox > div > div { background-color: #1a1a1f !important; border: 1px solid #2a2a2e !important; color: #cccccc !important; }
  .stSelectbox div[data-baseweb="select"] span { color: #cccccc !important; }

  /* Buttons */
  .stButton > button {
    background-color: #7f77dd22;
    border: 1px solid #7f77dd66;
    color: #afa9ec;
    border-radius: 6px;
    font-size: 13px;
    padding: 0.4rem 1.2rem;
  }
  .stButton > button:hover { background-color: #7f77dd44; border-color: #afa9ec; color: #ffffff; }

  /* Dataframe */
  .stDataFrame { border: 1px solid #2a2a2e; border-radius: 8px; }

  /* Expander */
  .streamlit-expanderHeader { background-color: #16161a !important; color: #888 !important; border: 1px solid #2a2a2e !important; border-radius: 6px !important; }
  .streamlit-expanderContent { background-color: #1a1a1f !important; border: 1px solid #2a2a2e !important; }

  /* Divider */
  hr { border-color: #2a2a2e !important; }

  /* Headings */
  h1, h2, h3 { color: #cccccc !important; }
  h1 { font-size: 20px !important; font-weight: 500 !important; }
  h2 { font-size: 16px !important; font-weight: 500 !important; }
  h3 { font-size: 14px !important; font-weight: 500 !important; }

  /* Section labels */
  .section-label {
    font-size: 10px;
    color: #555;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-family: monospace;
  }

  /* Strategy badges */
  .badge {
    display: inline-block;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: monospace;
    font-weight: 500;
  }
  .badge-naive    { background: #1d9e7522; color: #5dcaa5; border: 1px solid #1d9e7544; }
  .badge-hybrid   { background: #7f77dd22; color: #afa9ec; border: 1px solid #7f77dd44; }
  .badge-hyde     { background: #ba751722; color: #ef9f27; border: 1px solid #ba751744; }
  .badge-reranked { background: #d85a3022; color: #f0997b; border: 1px solid #d85a3044; }

  /* Result card */
  .result-card {
    background: #16161a;
    border: 1px solid #2a2a2e;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
  }
  .result-answer { font-size: 13px; color: #aaaaaa; line-height: 1.7; margin: 8px 0; }
  .result-meta { font-size: 11px; color: #555; font-family: monospace; }
  .result-best { border-left: 3px solid #7f77dd; }

  /* Confidence bar */
  .conf-wrap { margin: 8px 0 4px; }
  .conf-label { display: flex; justify-content: space-between; font-size: 10px; color: #555; font-family: monospace; margin-bottom: 3px; }

  /* Info box */
  .info-box {
    background: #1d9e7511;
    border: 1px solid #1d9e7533;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    color: #5dcaa5;
    font-family: monospace;
    margin-bottom: 12px;
  }
  .warn-box {
    background: #ba751711;
    border: 1px solid #ba751733;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    color: #ef9f27;
    font-family: monospace;
    margin-bottom: 12px;
  }
  .err-box {
    background: #a32d2d11;
    border: 1px solid #a32d2d33;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    color: #f09595;
    font-family: monospace;
    margin-bottom: 12px;
  }

  /* Chunk box */
  .chunk-box {
    background: #1a1a1f;
    border: 1px solid #2a2a2e;
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 6px;
    font-size: 11px;
  }
  .chunk-source { color: #7f77dd; font-family: monospace; font-size: 10px; margin-bottom: 4px; }
  .chunk-score  { color: #555; font-family: monospace; font-size: 10px; }
  .chunk-text   { color: #888; line-height: 1.6; }

  /* Plotly charts dark */
  .js-plotly-plot .plotly { background: transparent !important; }
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


def api_query(question: str, strategy: str, top_k: int = 5):
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "strategy": strategy, "top_k": top_k},
            timeout=30,
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def api_benchmark(question: str, top_k: int = 5):
    try:
        r = requests.post(
            f"{API_BASE}/benchmark",
            json={"question": question, "top_k": top_k},
            timeout=60,
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def load_csv():
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    csvs = sorted(results_dir.glob("benchmark_*.csv"))
    if not csvs:
        return None
    df = pd.read_csv(csvs[-1])
    df["confidence"]     = pd.to_numeric(df["confidence"], errors="coerce")
    df["total_latency"]  = pd.to_numeric(df["total_latency"], errors="coerce")
    df["tier"]           = pd.to_numeric(df["tier"], errors="coerce")
    return df


def load_golden():
    path = Path("data/golden_dataset.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("questions", data) if isinstance(data, dict) else data


def make_bar_chart(df_plot, x_col, y_col, color_map, title, x_label="%"):
    fig = go.Figure()
    for _, row in df_plot.iterrows():
        fig.add_trace(go.Bar(
            x=[row[x_col]],
            y=[row[y_col]],
            name=row[y_col],
            marker_color=color_map.get(row[y_col], "#888"),
            orientation="h",
            showlegend=False,
        ))
    fig.update_layout(
        plot_bgcolor="#16161a",
        paper_bgcolor="#16161a",
        font=dict(color="#888", size=11),
        margin=dict(l=0, r=0, t=28, b=0),
        height=160,
        title=dict(text=title, font=dict(size=11, color="#555"), x=0),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=10),
            title=x_label,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=11),
        ),
        barmode="overlay",
    )
    return fig


def conf_bar_html(confidence: float, strategy: str) -> str:
    color = STRATEGY_COLORS.get(strategy, "#888")
    pct = int(confidence * 100)
    return f"""
    <div class="conf-wrap">
      <div class="conf-label"><span>confidence</span><span style="color:{color}">{confidence:.2f}</span></div>
      <div style="background:#222;border-radius:3px;height:5px;">
        <div style="width:{pct}%;background:{color};height:5px;border-radius:3px;"></div>
      </div>
    </div>"""


def strategy_badge(name: str) -> str:
    return f'<span class="badge badge-{name}">{name}</span>'


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown('<div style="font-size:16px;font-weight:500;color:#7f77dd;margin-bottom:4px;">RAG Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#444;font-family:monospace;margin-bottom:20px;">multi-strategy evaluation</div>', unsafe_allow_html=True)

    health = api_health()
    if health:
        st.markdown('<div class="info-box">API online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">API offline — start uvicorn</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Corpus</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#555;font-family:monospace;line-height:2;">
    Chunks &nbsp;&nbsp; 30,432<br>
    BM25 &nbsp;&nbsp;&nbsp;&nbsp; 7,123<br>
    Docs &nbsp;&nbsp;&nbsp;&nbsp; 219<br>
    Dims &nbsp;&nbsp;&nbsp;&nbsp; 384
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Strategies</div>', unsafe_allow_html=True)
    for s, color in STRATEGY_COLORS.items():
        st.markdown(
            f'<div style="font-size:11px;font-family:monospace;padding:3px 0;">'
            f'<span style="color:{color};">■</span> {s}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div class="section-label">Top K</div>', unsafe_allow_html=True)
    top_k = st.slider("", min_value=3, max_value=10, value=5, label_visibility="collapsed")


# =============================================================================
# TABS
# =============================================================================

tab_query, tab_benchmark, tab_sources, tab_about = st.tabs([
    "Query", "Benchmark", "Sources", "About"
])


# =============================================================================
# TAB 1 — QUERY
# =============================================================================

with tab_query:
    st.markdown("### Ask a question")

    question = st.text_area(
        "",
        placeholder="What is LoRA and how does it reduce trainable parameters?",
        height=80,
        label_visibility="collapsed",
        key="query_input"
    )

    col_mode, col_strat, col_run = st.columns([2, 2, 1])
    with col_mode:
        mode = st.selectbox(
            "Mode",
            ["All 4 strategies (parallel)", "Single strategy"],
            label_visibility="collapsed"
        )
    with col_strat:
        single_strategy = st.selectbox(
            "Strategy",
            ["naive", "hybrid", "hyde", "reranked"],
            label_visibility="collapsed",
            disabled=(mode == "All 4 strategies (parallel)")
        )
    with col_run:
        run_clicked = st.button("Run", use_container_width=True)

    if run_clicked and question.strip():
        if not health:
            st.markdown('<div class="err-box">API is offline. Run: uvicorn app.main:app --reload</div>', unsafe_allow_html=True)
        else:
            if mode == "All 4 strategies (parallel)":
                with st.spinner("Running all 4 strategies in parallel..."):
                    result = api_benchmark(question.strip(), top_k)

                if "error" in result:
                    st.markdown(f'<div class="err-box">Error: {result["error"]}</div>', unsafe_allow_html=True)
                else:
                    # Summary row
                    c1, c2, c3, c4 = st.columns(4)
                    strategies_data = result.get("strategies", [])
                    answered = sum(1 for s in strategies_data if s.get("is_answerable"))

                    c1.metric("Best strategy",    result.get("best_strategy", "—").upper())
                    c2.metric("Fastest",          result.get("fastest_strategy", "—").upper())
                    c3.metric("Wall time",        f'{result.get("total_time", 0):.1f}s')
                    c4.metric("Answerable",       f"{answered}/4")

                    st.markdown("")

                    # Sort: best first, then by confidence
                    best = result.get("best_strategy")
                    sorted_strategies = sorted(
                        strategies_data,
                        key=lambda x: (x["strategy"] != best, -x.get("confidence", 0))
                    )

                    for s in sorted_strategies:
                        is_best = s["strategy"] == best
                        card_class = "result-card result-best" if is_best else "result-card"
                        badge = strategy_badge(s["strategy"])
                        best_tag = ' <span style="font-size:10px;color:#7f77dd;font-family:monospace;">★ best</span>' if is_best else ""
                        lat = s.get("latency", {})
                        lat_str = (
                            f'retrieve {lat.get("retrieve", 0):.2f}s &nbsp;|&nbsp; '
                            f'generate {lat.get("generate", 0):.2f}s &nbsp;|&nbsp; '
                            f'total {lat.get("total", 0):.2f}s'
                        )
                        conf_html = conf_bar_html(s.get("confidence", 0), s["strategy"])
                        answer_text = s.get("answer", "")[:400]

                        answerable_tag = (
                            '<span style="color:#5dcaa5;font-size:10px;">answerable</span>'
                            if s.get("is_answerable")
                            else '<span style="color:#f09595;font-size:10px;">abstained</span>'
                        )

                        st.markdown(f"""
                        <div class="{card_class}">
                          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                            {badge}{best_tag}
                            {answerable_tag}
                          </div>
                          <div class="result-answer">{answer_text}</div>
                          {conf_html}
                          <div class="result-meta" style="margin-top:8px;">{lat_str}</div>
                        </div>""", unsafe_allow_html=True)

                        # Show chunks if available
                        chunks = s.get("retrieved_chunks", [])
                        if chunks:
                            with st.expander(f"Retrieved chunks ({len(chunks)})", expanded=False):
                                for i, chunk in enumerate(chunks, 1):
                                    src = chunk.get("metadata", {}).get("filename", "unknown")
                                    score = chunk.get("score", 0)
                                    text = chunk.get("content", "")[:300]
                                    st.markdown(f"""
                                    <div class="chunk-box">
                                      <div class="chunk-source">{i}. {src}</div>
                                      <div class="chunk-score">score: {score:.4f}</div>
                                      <div class="chunk-text">{text}...</div>
                                    </div>""", unsafe_allow_html=True)

            else:
                # Single strategy
                with st.spinner(f"Running {single_strategy}..."):
                    result = api_query(question.strip(), single_strategy, top_k)

                if "error" in result:
                    st.markdown(f'<div class="err-box">Error: {result["error"]}</div>', unsafe_allow_html=True)
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Strategy",    single_strategy.upper())
                    c2.metric("Confidence",  f'{result.get("confidence", 0):.2f}')
                    c3.metric("Latency",     f'{result.get("latency", {}).get("total", 0):.2f}s')

                    is_ans = result.get("is_answerable", False)
                    ans_html = (
                        '<span style="color:#5dcaa5;font-size:10px;font-family:monospace;">answerable</span>'
                        if is_ans else
                        '<span style="color:#f09595;font-size:10px;font-family:monospace;">abstained</span>'
                    )

                    conf_html = conf_bar_html(result.get("confidence", 0), single_strategy)
                    badge = strategy_badge(single_strategy)
                    lat = result.get("latency", {})

                    st.markdown(f"""
                    <div class="result-card">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        {badge} {ans_html}
                      </div>
                      <div class="result-answer">{result.get("answer", "")}</div>
                      <div style="font-size:11px;color:#666;font-family:monospace;margin:8px 0 4px;">
                        reasoning: {result.get("reasoning", "")[:200]}
                      </div>
                      {conf_html}
                      <div class="result-meta" style="margin-top:8px;">
                        retrieve {lat.get("retrieve",0):.2f}s &nbsp;|&nbsp;
                        generate {lat.get("generate",0):.2f}s &nbsp;|&nbsp;
                        total {lat.get("total",0):.2f}s
                      </div>
                    </div>""", unsafe_allow_html=True)

                    chunks = result.get("retrieved_chunks", [])
                    if chunks:
                        with st.expander(f"Retrieved chunks ({len(chunks)})", expanded=False):
                            for i, chunk in enumerate(chunks, 1):
                                src = chunk.get("metadata", {}).get("filename", "unknown")
                                score = chunk.get("score", 0)
                                text = chunk.get("content", "")[:300]
                                st.markdown(f"""
                                <div class="chunk-box">
                                  <div class="chunk-source">{i}. {src}</div>
                                  <div class="chunk-score">score: {score:.4f}</div>
                                  <div class="chunk-text">{text}...</div>
                                </div>""", unsafe_allow_html=True)

    elif run_clicked:
        st.markdown('<div class="warn-box">Please enter a question first.</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2 — BENCHMARK
# =============================================================================

with tab_benchmark:
    st.markdown("### Benchmark results")

    df = load_csv()
    if df is None:
        st.markdown('<div class="warn-box">No benchmark CSV found in results/ — run scripts_main/run_benchmark.py first</div>', unsafe_allow_html=True)
    else:
        df_clean = df[~df["answer"].str.contains("ERROR", na=False)]
        total_q = df_clean["question_id"].nunique()
        strategies_found = df_clean["strategy"].unique().tolist()

        st.markdown(f'<div class="info-box">Loaded {len(df_clean)} rows — {total_q}/30 questions — {len(strategies_found)} strategies</div>', unsafe_allow_html=True)

        # Per-strategy metrics
        metrics = []
        for s in ["naive", "hybrid", "hyde", "reranked"]:
            sub = df_clean[df_clean["strategy"] == s]
            if len(sub) == 0:
                continue
            golden = load_golden()
            golden_map = {q["id"]: q for q in golden} if golden else {}

            term_hits = []
            abstention = []
            calibration = []
            for _, row in sub.iterrows():
                qid = row["question_id"]
                g = golden_map.get(qid, {})
                terms = g.get("expected_retrieval", {}).get("must_contain_terms", [])
                if terms:
                    answer_lower = str(row["answer"]).lower()
                    hit = sum(1 for t in terms if t.lower() in answer_lower)
                    term_hits.append(hit / len(terms))

                exp_ans = g.get("is_answerable", True)
                act_ans = str(row.get("is_answerable", "false")).lower() == "true"
                abstention.append(1.0 if act_ans == exp_ans else 0.0)

                conf = float(row["confidence"])
                if exp_ans:
                    exp_min = g.get("expected_confidence_min", 0.7)
                    calibration.append(1.0 if conf >= exp_min else 0.0)
                else:
                    exp_max = g.get("expected_confidence_max", 0.3)
                    calibration.append(1.0 if conf <= exp_max else 0.0)

            metrics.append({
                "strategy":    s,
                "term_hit":    round(sum(term_hits) / len(term_hits) * 100, 1) if term_hits else 0,
                "abstention":  round(sum(abstention) / len(abstention) * 100, 1) if abstention else 0,
                "calibration": round(sum(calibration) / len(calibration) * 100, 1) if calibration else 0,
                "avg_conf":    round(sub["confidence"].mean(), 2),
                "avg_latency": round(sub["total_latency"].mean(), 2),
                "answered":    int((sub["is_answerable"].astype(str).str.lower() == "true").sum()),
                "total":       len(sub),
            })

        if metrics:
            # Top summary cards
            best_cal  = max(metrics, key=lambda x: x["calibration"])
            best_abs  = max(metrics, key=lambda x: x["abstention"])
            best_fast = min(metrics, key=lambda x: x["avg_latency"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best calibration",  best_cal["strategy"].upper(),  f'{best_cal["calibration"]}%')
            c2.metric("Best abstention",   best_abs["strategy"].upper(),  f'{best_abs["abstention"]}%')
            c3.metric("Fastest",           best_fast["strategy"].upper(), f'{best_fast["avg_latency"]}s')
            c4.metric("Questions done",    f"{total_q}/30")

            st.markdown("")

            # Charts row
            col_l, col_r = st.columns(2)

            with col_l:
                df_m = pd.DataFrame(metrics)

                fig_abs = go.Figure()
                for _, row in df_m.iterrows():
                    fig_abs.add_trace(go.Bar(
                        y=[row["strategy"]],
                        x=[row["abstention"]],
                        orientation="h",
                        marker_color=STRATEGY_COLORS.get(row["strategy"], "#888"),
                        showlegend=False,
                        text=[f'{row["abstention"]}%'],
                        textposition="outside",
                        textfont=dict(size=10, color="#888"),
                    ))
                fig_abs.update_layout(
                    plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                    font=dict(color="#888", size=11),
                    margin=dict(l=0, r=30, t=28, b=0),
                    height=180,
                    title=dict(text="Abstention accuracy", font=dict(size=11, color="#555"), x=0),
                    xaxis=dict(showgrid=False, zeroline=False, range=[0, 110], ticksuffix="%"),
                    yaxis=dict(showgrid=False),
                    barmode="group",
                )
                st.plotly_chart(fig_abs, use_container_width=True)

                fig_lat = go.Figure()
                for _, row in df_m.iterrows():
                    fig_lat.add_trace(go.Bar(
                        y=[row["strategy"]],
                        x=[row["avg_latency"]],
                        orientation="h",
                        marker_color=STRATEGY_COLORS.get(row["strategy"], "#888"),
                        showlegend=False,
                        text=[f'{row["avg_latency"]}s'],
                        textposition="outside",
                        textfont=dict(size=10, color="#888"),
                    ))
                fig_lat.update_layout(
                    plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                    font=dict(color="#888", size=11),
                    margin=dict(l=0, r=40, t=28, b=0),
                    height=180,
                    title=dict(text="Avg latency (s)", font=dict(size=11, color="#555"), x=0),
                    xaxis=dict(showgrid=False, zeroline=False, ticksuffix="s"),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_lat, use_container_width=True)

            with col_r:
                fig_cal = go.Figure()
                for _, row in df_m.iterrows():
                    fig_cal.add_trace(go.Bar(
                        y=[row["strategy"]],
                        x=[row["calibration"]],
                        orientation="h",
                        marker_color=STRATEGY_COLORS.get(row["strategy"], "#888"),
                        showlegend=False,
                        text=[f'{row["calibration"]}%'],
                        textposition="outside",
                        textfont=dict(size=10, color="#888"),
                    ))
                fig_cal.update_layout(
                    plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                    font=dict(color="#888", size=11),
                    margin=dict(l=0, r=30, t=28, b=0),
                    height=180,
                    title=dict(text="Confidence calibration", font=dict(size=11, color="#555"), x=0),
                    xaxis=dict(showgrid=False, zeroline=False, range=[0, 110], ticksuffix="%"),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_cal, use_container_width=True)

                fig_conf = go.Figure()
                for _, row in df_m.iterrows():
                    fig_conf.add_trace(go.Bar(
                        y=[row["strategy"]],
                        x=[row["avg_conf"]],
                        orientation="h",
                        marker_color=STRATEGY_COLORS.get(row["strategy"], "#888"),
                        showlegend=False,
                        text=[f'{row["avg_conf"]}'],
                        textposition="outside",
                        textfont=dict(size=10, color="#888"),
                    ))
                fig_conf.update_layout(
                    plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                    font=dict(color="#888", size=11),
                    margin=dict(l=0, r=40, t=28, b=0),
                    height=180,
                    title=dict(text="Avg confidence", font=dict(size=11, color="#555"), x=0),
                    xaxis=dict(showgrid=False, zeroline=False, range=[0, 1.2]),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_conf, use_container_width=True)

            # Confidence by tier scatter
            st.markdown("---")
            st.markdown("#### Confidence by tier and strategy")
            fig_scatter = px.strip(
                df_clean,
                x="strategy", y="confidence",
                color="strategy",
                color_discrete_map=STRATEGY_COLORS,
                facet_col="tier",
                stripmode="overlay",
                labels={"confidence": "Confidence", "strategy": ""},
            )
            fig_scatter.update_layout(
                plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                font=dict(color="#888", size=11),
                margin=dict(l=0, r=0, t=40, b=0),
                height=240,
                showlegend=False,
            )
            fig_scatter.update_xaxes(showgrid=False)
            fig_scatter.update_yaxes(showgrid=False, range=[0, 1.1])
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Raw data table
            st.markdown("---")
            st.markdown("#### Raw results")
            tier_filter = st.selectbox(
                "Filter by tier",
                ["All", "Tier 1 (easy)", "Tier 2 (multi-chunk)", "Tier 3 (adversarial)"],
                key="tier_filter"
            )
            tier_map = {
                "Tier 1 (easy)": 1,
                "Tier 2 (multi-chunk)": 2,
                "Tier 3 (adversarial)": 3
            }
            display_df = df_clean.copy()
            if tier_filter != "All":
                display_df = display_df[display_df["tier"] == tier_map[tier_filter]]

            show_cols = ["question_id", "tier", "strategy", "confidence",
                         "is_answerable", "total_latency"]
            if "top_sources" in display_df.columns:
                show_cols.append("top_sources")

            st.dataframe(
                display_df[show_cols].reset_index(drop=True),
                use_container_width=True,
                height=300,
            )


# =============================================================================
# TAB 3 — SOURCES
# =============================================================================

with tab_sources:
    st.markdown("### Corpus & source analysis")

    golden = load_golden()
    if not golden:
        st.markdown('<div class="warn-box">data/golden_dataset.json not found</div>', unsafe_allow_html=True)
    else:
        golden_map = {q["id"]: q for q in golden}

        # Corpus group counts
        corpus_groups = {
            "huggingface": 0, "langchain": 0, "anthropic": 0,
            "papers": 0, "beir": 0, "adversarial": 0
        }
        corpus_colors = {
            "huggingface": "#5DCAA5",
            "langchain":   "#7F77DD",
            "anthropic":   "#EF9F27",
            "papers":      "#F0997B",
            "beir":        "#85B7EB",
            "adversarial": "#555",
        }
        for q in golden:
            sources = q.get("source_documents", [])
            placed = False
            for s in sources:
                sl = s.lower()
                if any(x in sl for x in ["peft", "transformers", "llm_tutorial", "pipeline", "tokeniz", "training", "perf_train", "generation", "quantization", "bitsandbytes", "deepspeed", "kv_cache"]):
                    corpus_groups["huggingface"] += 1; placed = True; break
                elif any(x in sl for x in ["langchain", "lcel", "agent", "chain", "retriever"]):
                    corpus_groups["langchain"] += 1; placed = True; break
                elif "anthropic" in sl:
                    corpus_groups["anthropic"] += 1; placed = True; break
                elif any(x in sl for x in ["paper", "pdf", "lora", "rag", "attention", "selfrag", "hyde"]):
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
                marker_colors=[corpus_colors[k] for k in corpus_groups],
                textinfo="label+value",
                textfont=dict(size=11, color="#aaa"),
                hole=0.5,
                showlegend=False,
            ))
            fig_pie.update_layout(
                plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                margin=dict(l=0, r=0, t=28, b=0),
                height=240,
                title=dict(text="Questions per source group", font=dict(size=11, color="#555"), x=0),
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
                y=cats_df["category"],
                x=cats_df["count"],
                orientation="h",
                marker_color="rgba(127,119,221,0.3)",
                marker_line_color="#7f77dd",
                marker_line_width=0.5,
            ))
            fig_cat.update_layout(
                plot_bgcolor="#16161a", paper_bgcolor="#16161a",
                font=dict(color="#888", size=10),
                margin=dict(l=0, r=0, t=28, b=0),
                height=240,
                title=dict(text="Question categories", font=dict(size=11, color="#555"), x=0),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        # Expected source files
        st.markdown("---")
        st.markdown("#### Expected source files")
        expected_sources = {}
        source_groups_map = {}
        for q in golden:
            sources = q.get("source_documents", [])
            for s in sources:
                expected_sources[s] = expected_sources.get(s, 0) + 1
                sl = s.lower()
                if any(x in sl for x in ["peft", "transformers", "llm_tutorial", "pipeline", "tokeniz", "training", "perf_train", "generation", "quantization", "bitsandbytes", "deepspeed", "kv_cache"]):
                    source_groups_map[s] = "huggingface"
                elif any(x in sl for x in ["langchain", "lcel", "agent", "chain", "retriever"]):
                    source_groups_map[s] = "langchain"
                elif "anthropic" in sl:
                    source_groups_map[s] = "anthropic"
                elif any(x in sl for x in ["paper", "pdf", "lora", "rag", "attention", "selfrag", "hyde"]):
                    source_groups_map[s] = "papers"
                else:
                    source_groups_map[s] = "other"

        src_df = pd.DataFrame(
            sorted(expected_sources.items(), key=lambda x: x[1], reverse=True),
            columns=["source_file", "questions"]
        )
        src_df["group"] = src_df["source_file"].map(lambda x: source_groups_map.get(x, "other"))
        src_df["color"] = src_df["group"].map(lambda x: corpus_colors.get(x, "#888"))

        fig_src = go.Figure()
        for _, row in src_df.iterrows():
            fig_src.add_trace(go.Bar(
                y=[row["source_file"].split("/")[-1]],
                x=[row["questions"]],
                orientation="h",
                marker_color=row["color"],
                marker_line_color=row["color"],
                marker_line_width=0.5,
                showlegend=False,
            ))
        fig_src.update_layout(
            plot_bgcolor="#16161a", paper_bgcolor="#16161a",
            font=dict(color="#888", size=10),
            margin=dict(l=0, r=20, t=10, b=0),
            height=max(300, len(src_df) * 20),
            xaxis=dict(showgrid=False, zeroline=False, dtick=1, title="questions targeting this file"),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig_src, use_container_width=True)

        # Coverage gaps
        st.markdown("---")
        st.markdown("#### Corpus gaps")
        st.markdown('<div class="warn-box">Langchain agent files not indexed — q005, q006, q012, q017 unanswerable across ALL strategies. Anthropic docs = 0 questions (not in corpus).</div>', unsafe_allow_html=True)

        df = load_csv()
        if df is not None:
            df_clean = df[~df["answer"].str.contains("ERROR", na=False)]
            low_conf = df_clean[df_clean["confidence"] < 0.3].groupby("question_id").size()
            if len(low_conf) > 0:
                gap_ids = low_conf[low_conf == 4].index.tolist()
                if gap_ids:
                    st.markdown(f'<div class="err-box">Consistent failures (all 4 strategies low confidence): {", ".join(gap_ids)}</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 4 — ABOUT
# =============================================================================

with tab_about:
    st.markdown("### Architecture")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div style="background:#16161a;border:1px solid #2a2a2e;border-radius:8px;padding:16px;margin-bottom:10px;">
          <div style="font-size:11px;color:#7f77dd;letter-spacing:1px;margin-bottom:12px;">RETRIEVAL STRATEGIES</div>
          <div style="font-size:11px;font-family:monospace;color:#555;line-height:2.2;">
            <span style="color:#5dcaa5;">naive</span> &nbsp;&nbsp;&nbsp;&nbsp; vector search &rarr; fixed_size chunks<br>
            <span style="color:#afa9ec;">hybrid</span> &nbsp;&nbsp;&nbsp; BM25 + vector + RRF fusion<br>
            <span style="color:#ef9f27;">hyde</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; hypothetical doc &rarr; embed &rarr; search<br>
            <span style="color:#f0997b;">reranked</span> &nbsp; hybrid top 20 &rarr; cross-encoder &rarr; top 5
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#16161a;border:1px solid #2a2a2e;border-radius:8px;padding:16px;margin-bottom:10px;">
          <div style="font-size:11px;color:#7f77dd;letter-spacing:1px;margin-bottom:12px;">EVALUATION METRICS</div>
          <div style="font-size:11px;font-family:monospace;color:#555;line-height:2.2;">
            term_hit_rate &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; must_contain_terms in answer<br>
            abstention_acc &nbsp;&nbsp;&nbsp;&nbsp; is_answerable == expected<br>
            conf_calibration &nbsp;&nbsp; confidence in expected range<br>
            recall@k &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; after full benchmark run<br>
            mrr &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mean reciprocal rank
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style="background:#16161a;border:1px solid #2a2a2e;border-radius:8px;padding:16px;margin-bottom:10px;">
          <div style="font-size:11px;color:#7f77dd;letter-spacing:1px;margin-bottom:12px;">TECH STACK</div>
          <div style="font-size:11px;font-family:monospace;color:#555;line-height:2.2;">
            llm &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Groq llama-3.3-70b-versatile<br>
            embeddings &nbsp; all-MiniLM-L6-v2 (384d)<br>
            vector db &nbsp;&nbsp; Qdrant local (30,432 pts)<br>
            bm25 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; rank_bm25<br>
            reranker &nbsp;&nbsp;&nbsp; ms-marco-MiniLM-L-6-v2<br>
            api &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; FastAPI + asyncio.gather<br>
            outputs &nbsp;&nbsp;&nbsp;&nbsp; instructor + pydantic
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#16161a;border:1px solid #2a2a2e;border-radius:8px;padding:16px;">
          <div style="font-size:11px;color:#7f77dd;letter-spacing:1px;margin-bottom:12px;">CORPUS</div>
          <div style="font-size:11px;font-family:monospace;color:#555;line-height:2.2;">
            huggingface docs &nbsp; peft, transformers, training<br>
            langchain docs &nbsp;&nbsp;&nbsp; agents, retrievers, lcel<br>
            research papers &nbsp;&nbsp; lora, rag, attention, hyde<br>
            golden dataset &nbsp;&nbsp;&nbsp; 30 tiered questions<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tier1=easy tier2=multi tier3=adversarial
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;font-family:monospace;color:#444;line-height:2;text-align:center;padding:8px;">
      RAG fails silently &mdash; naive retrieval vs reranked hybrid &mdash; built to measure the gap
    </div>
    """, unsafe_allow_html=True)