from typing import List, Dict, Any
import streamlit as st


def render_response_header(status: str) -> None:
    """Renders the top label of the enterprise response container with appropriate iconography."""
    icons = {
        "answered": "⚙️ GENERATED ENTERPRISE RESPONSE:",
        "insufficient_evidence": "⚠️ INSUFFICIENT EVIDENCE NOTICE:",
        "clarification_needed": "💬 CLARIFICATION REQUESTED:",
        "conflict_detected": "🚨 CONTRADICTION / CONFLICT DETECTED:",
        "error": "❌ SYSTEM ERROR:"
    }
    label = icons.get(status, "⚙️ GENERATED ENTERPRISE RESPONSE:")
    st.caption(label)


def render_citation_cards(sources: List[Dict[str, Any]]) -> None:
    """
    Renders source citations as side-by-side cards with document name,
    section, page number, and relevance percentage matching the hackathon UI wireframe.
    """
    if not sources:
        return

    st.markdown("#### 📌 Source Citations")

    # Render up to 2 cards per row
    cols = st.columns(min(len(sources), 2))
    for idx, src in enumerate(sources[:4]):
        col_idx = idx % 2
        with cols[col_idx]:
            with st.container(border=True):
                doc_name = src.get("document", "Document")
                sec = src.get("section", "General")
                page = src.get("page", "N/A")
                sim = src.get("similarity", 0.0)

                st.markdown(f"📄 **{doc_name}**")
                st.caption(f"**Section:** {sec}")
                page_str = str(page) if page and page != "N/A" else "N/A"
                st.caption(f"**Page:** {page_str} &nbsp;|&nbsp; **Relevance:** `{sim:.0%}`")

    # Collapsible excerpts expander
    with st.expander("🔍 View Supporting Evidence Excerpts"):
        for i, src in enumerate(sources, 1):
            doc = src.get("document", "Document")
            page = src.get("page", "N/A")
            sec = src.get("section", "General")
            text = src.get("text", src.get("snippet", "Detailed text excerpt available in indexed documentation."))
            st.markdown(f"**[{i}] {doc}** — Page {page} (*{sec}*):")
            st.info(f'"{text}"')
            if i < len(sources):
                st.divider()


def render_scenario_alert(status: str, warnings: List[str]) -> None:
    """Renders contextual alert banners based on response state."""
    if status == "insufficient_evidence":
        st.warning("ℹ️ **Notice:** No matching evidence found in the uploaded documents to substantiate this question.")
    elif status == "clarification_needed":
        st.info("ℹ️ **Clarification Required:** The query is ambiguous. Please provide more specifics.")
    elif status == "conflict_detected":
        warn_msg = warnings[0] if warnings else "Contradictory statements found across uploaded document versions."
        st.error(f"⚠️ **Conflict Alert:** {warn_msg}")
    elif status == "error":
        err_msg = warnings[0] if warnings else "An error occurred during retrieval or processing."
        st.error(f"❌ {err_msg}")
