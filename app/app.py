"""LakehouseIQ - Streamlit front end.

UI only: no SQL and no prompt text live here. Everything that touches the model or the
warehouse goes through ``lakehouseiq.pipeline.answer_question``.
"""

import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lakehouseiq import store, tracing, warehouse
from lakehouseiq.config import Settings
from lakehouseiq.parsing import format_sql
from lakehouseiq.pipeline import AnswerResult, answer_question
from lakehouseiq.semantic import load_registry

SEMANTIC_DIR = Path(__file__).resolve().parent / "semantic"
COLORS = ["#2F80ED", "#F2994A", "#27AE60", "#EB5757", "#9B51E0", "#00A3A3"]
SCHEMA_ICONS = {"sales": "📈", "hr": "👥", "finance": "💰"}

QUICK_QUESTIONS = {
    "Sales and revenue": [
        "Monthly revenue trend for last 6 months?",
        "Top 5 products by net revenue?",
        "Revenue by customer segment?",
        "Which channel has the highest order volume?",
        "Which 5 products have the highest return rate?",
    ],
    "HR and payroll": [
        "Average salary by department?",
        "Monthly payroll cost by department?",
        "Which departments have the best average performance rating?",
    ],
    "Finance and budget": [
        "Which departments are more than 10% over budget year-to-date?",
        "Top expense categories this year?",
        "How much invoice balance is outstanding in each aging bucket?",
    ],
    "Inventory": ["Which products are below their reorder point in the latest snapshot?"],
}

VERDICT_LABELS = {
    "GOOD": ("High confidence", "success"),
    "NEEDS REVIEW": ("Needs review", "warning"),
    "RISKY": ("Risky", "error"),
    "UNSCORED": ("Unscored", "info"),
    "BLOCKED": ("Blocked", "error"),
    "ERROR": ("Error", "error"),
}

st.set_page_config(page_title="LakehouseIQ", page_icon="🏠", layout="wide")
st.markdown(
    """
<style>
  [data-testid="stChatMessage"] { border-radius: 12px; border: 1px solid rgba(128,128,128,.25); }
  [data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: .75rem 1rem; }
  .welcome { text-align: center; padding: 2.5rem 1rem 1.5rem; }
  .welcome h2 { margin-bottom: .25rem; }
  .welcome p { opacity: .7; }
</style>
""",
    unsafe_allow_html=True,
)


# ── cached context ────────────────────────────────────────────────────────────


@st.cache_resource(ttl=3600, show_spinner="Reading catalog metadata...")
def load_context():
    settings = Settings.from_env()
    tracing.configure(settings.mlflow_experiment_id)
    warehouse.configure(settings)
    registry = load_registry(SEMANTIC_DIR)
    schema_ctx = warehouse.get_live_schema(settings)
    return settings, registry, schema_ctx


def user_email() -> str:
    try:
        return st.context.headers.get("X-Forwarded-Email") or "unknown"
    except Exception:
        return "unknown"


# ── session state ─────────────────────────────────────────────────────────────

st.session_state.setdefault("session_id", str(uuid.uuid4()))
st.session_state.setdefault("turns", [])  # [{"question": str, "result": AnswerResult}]

try:
    settings, registry, schema_ctx = load_context()
except Exception as exc:
    st.error(f"Could not load the catalog metadata: {exc}")
    st.caption(
        "Check that the app's service principal has USE CATALOG / USE SCHEMA / SELECT on sales, hr and "
        "finance, that a SQL warehouse resource is bound, and that notebooks 00-05 have run."
    )
    st.stop()


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## LakehouseIQ")
    st.caption("Ask your Unity Catalog metric views in plain English")
    st.divider()
    st.markdown("##### Quick questions")
    for i, (group, questions) in enumerate(QUICK_QUESTIONS.items()):
        with st.expander(group, expanded=(i == 0)):
            for q in questions:
                if st.button(q, key=f"ex_{q}", width="stretch"):
                    st.session_state["pending_question"] = q

    st.divider()
    with st.expander("Metric views"):
        by_schema: dict[str, list] = {}
        for mv in registry:
            by_schema.setdefault(mv.schema, []).append(mv)
        for schema, views in by_schema.items():
            st.markdown(f"**{SCHEMA_ICONS.get(schema, '')} {schema}**")
            for mv in views:
                st.caption(f"{mv.name}: {mv.description}")
    st.caption(f"Data as of {schema_ctx.as_of_date}")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("Refresh", width="stretch", help="Reload catalog metadata"):
        load_context.clear()
        st.rerun()
    if c2.button("Clear chat", width="stretch"):
        st.session_state["turns"] = []
        st.rerun()


# ── charts ────────────────────────────────────────────────────────────────────

TEMPORAL_HINTS = ("month", "period", "date", "quarter", "week")


def build_chart(df: pd.DataFrame):
    numeric = df.select_dtypes(include="number").columns.tolist()
    labels = [c for c in df.columns if c not in numeric]
    if not numeric or not labels or len(df) < 2:
        return None
    label, values = labels[0], numeric[:3]
    temporal = pd.api.types.is_datetime64_any_dtype(df[label]) or any(h in str(label).lower() for h in TEMPORAL_HINTS)

    if temporal:
        fig = go.Figure()
        for i, col in enumerate(values):
            fig.add_trace(
                go.Scatter(x=df[label], y=df[col], mode="lines+markers", name=col.replace("_", " ").title(),
                           line={"color": COLORS[i % len(COLORS)], "width": 2.5})
            )
    elif len(values) == 1 and 2 <= len(df) <= 8 and df[values[0]].min() >= 0 and df[values[0]].sum() > 0:
        fig = px.pie(df, names=label, values=values[0], color_discrete_sequence=COLORS, hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
    else:
        fig = go.Figure()
        for i, col in enumerate(values):
            fig.add_trace(go.Bar(x=df[label], y=df[col], name=col.replace("_", " ").title(), marker_color=COLORS[i % len(COLORS)]))
        fig.update_layout(barmode="group")
    fig.update_layout(margin={"l": 40, "r": 20, "t": 30, "b": 40}, legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"})
    return fig


# ── rendering ─────────────────────────────────────────────────────────────────


def render_quality(res: AnswerResult) -> None:
    label, kind = VERDICT_LABELS.get(res.verdict, (res.verdict, "info"))
    getattr(st, kind)(label)
    s = res.scores
    if not s:
        st.caption("No scores: the query did not run.")
        return

    def fmt(v):
        return f"{v}/10" if v is not None else "n/a"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Relevance (judge)", fmt(s.get("relevance")))
    c2.metric("Schema fit (code)", fmt(s.get("schema")))
    c3.metric("SQL quality (judge)", fmt(s.get("sql_quality")))
    c4.metric("Value check (code)", fmt(s.get("value")))
    if s.get("misuse"):
        st.error("Metric view rules: " + "; ".join(s["misuse"]))
    if s.get("bad_values"):
        st.warning("Unrecognised filter values: " + ", ".join(repr(v) for v in s["bad_values"][:5]))
    with st.expander("Score details"):
        st.markdown(f"- **Relevance:** {s.get('relevance_reason')}")
        st.markdown(f"- **Schema fit:** {s.get('schema_reason')}")
        st.markdown(f"- **SQL quality:** {s.get('sql_quality_reason')}")
        st.markdown(f"- **Value check:** {s.get('value_reason')}")


def render_trace(res: AnswerResult) -> None:
    timings = pd.DataFrame({"step": list(res.timings_ms), "ms": list(res.timings_ms.values())})
    st.dataframe(timings, hide_index=True, width="stretch")
    st.markdown(f"- **Generator:** `{settings.generator_endpoint}`")
    st.markdown(f"- **Judge:** `{settings.judge_endpoint}` (a different model family from the generator)")
    st.markdown(f"- **Repaired:** {'yes' if res.repaired else 'no'}")
    st.markdown(f"- **Source view:** `{res.source_view or 'base tables'}`")
    st.markdown(f"- **MLflow trace id:** `{res.trace_id or 'tracing not configured'}`")


def render_report_mistake(res: AnswerResult, key: str) -> None:
    with st.expander("Report a mistake"):
        fixed = st.text_area("Corrected SQL", value=res.sql or "", height=120, key=f"fix_{key}")
        reason = st.text_input("What was wrong?", key=f"why_{key}")
        if st.button("Save correction", key=f"save_{key}"):
            if not fixed.strip() or fixed.strip() == (res.sql or "").strip():
                st.warning("Change the SQL before saving.")
            elif store.save_correction(
                settings=settings, user_email=user_email(), question=res.question, bad_sql=res.sql or "",
                corrected_sql=fixed.strip(), reason=reason, bad_values=res.scores.get("bad_values"),
            ):
                st.success("Correction saved. Similar questions will see it.")
            else:
                st.error("Could not save the correction.")


def render_result(res: AnswerResult, key: str, animate: bool = False) -> None:
    if res.blocked_reason:
        st.error(f"Blocked: {res.blocked_reason}")
    if res.error:
        st.error(res.error)
    if res.sql is None and not res.blocked_reason:
        return

    df = res.df
    has_data = df is not None and not df.empty
    if df is not None and not has_data:
        st.warning("The query returned no rows.")

    names = (["Data", "Chart"] if has_data else []) + ["SQL", "Quality", "Trace"]
    tabs = dict(zip(names, st.tabs(names), strict=True))
    if has_data:
        with tabs["Data"]:
            st.dataframe(df, width="stretch", height=min(len(df) * 38 + 50, 500))
            if df.attrs.get("truncated"):
                st.caption(f"Showing the first {len(df)} rows; add filters to narrow the result.")
        with tabs["Chart"]:
            fig = build_chart(df)
            if fig:
                st.plotly_chart(fig, width="stretch", key=f"chart_{key}")
            else:
                st.info("No chart for this result shape.")
    with tabs["SQL"]:
        st.code(format_sql(res.sql or ""), language="sql")
        if res.repaired:
            st.caption("This query was repaired once after the warehouse rejected the first attempt.")
    with tabs["Quality"]:
        render_quality(res)
    with tabs["Trace"]:
        render_trace(res)

    if res.summary:
        st.markdown("---")
        st.markdown("#### Summary")
        box = st.empty()
        if animate:
            for i in range(0, len(res.summary), 40):
                box.markdown(res.summary[: i + 40])
        box.markdown(res.summary)
    if res.sql and not res.blocked_reason:
        render_report_mistake(res, key)


# ── chat ──────────────────────────────────────────────────────────────────────

turns = st.session_state["turns"]

if not turns:
    st.markdown(
        '<div class="welcome"><h2>Welcome to LakehouseIQ</h2>'
        "<p>Ask about sales, HR and finance in plain English. Answers come from governed metric views.</p></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for col, (icon, label, q) in zip(
        cols,
        [("📈", "Sales", "Monthly revenue trend for last 6 months?"),
         ("👥", "HR", "Average salary by department?"),
         ("💰", "Finance", "Which departments are more than 10% over budget year-to-date?")],
        strict=True,
    ):
        if col.button(f"{icon} {label}\n{q}", key=f"welcome_{label}", width="stretch"):
            st.session_state["pending_question"] = q
            st.rerun()

# Prior turns are re-rendered from the stored AnswerResult: no re-query, nothing lost on rerun.
for i, turn in enumerate(turns):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_result(turn["result"], key=f"t{i}")

question = st.session_state.pop("pending_question", None) or st.chat_input("Ask a question about your data...")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        history = [{"question": t["question"], "sql": t["result"].sql} for t in turns if t["result"].sql][-2:]
        with st.spinner("Thinking..."):
            corrections = store.get_relevant_corrections(question, settings=settings)
            result = answer_question(
                question, settings=settings, schema_ctx=schema_ctx, registry=registry,
                history=history, corrections=corrections,
            )
        turns.append({"question": question, "result": result})
        render_result(result, key=f"t{len(turns) - 1}", animate=True)
        store.save_turn(
            settings=settings, session_id=st.session_state["session_id"], turn_id=len(turns), user_email=user_email(),
            question=question, generated_sql=result.sql, source_view=result.source_view, verdict=result.verdict,
            row_count=None if result.df is None else len(result.df), error=result.error or result.blocked_reason,
            latency_ms=result.timings_ms.get("total"),
        )
