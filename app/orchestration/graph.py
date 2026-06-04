"""
LangGraph state machine for the AI News Aggregator pipeline.

Graph topology
──────────────
START → scrape → retrieve → digest → fact_check ──(rate > 10% & revisions < 3)──→ fact_check
                                                ↓ clean
                                           save_preview ──(should_send_email)──→ email → END
                                                ↓ no send
                                               END
"""
import logging
from langgraph.graph import StateGraph, END

from app.orchestration.state import PipelineState
from app.orchestration.nodes import (
    scrape_node,
    retrieve_node,
    digest_node,
    fact_check_node,
    email_node,
    save_preview_node,
)

logger = logging.getLogger(__name__)


def should_retry_fact_check(state: PipelineState) -> str:
    """
    Conditional edge after fact_check.
    Loops back to fact_check when hallucination is still high and budget remains.
    """
    if state.get("error") and "fact_check" in state.get("error", ""):
        return "save_preview"

    rate = state.get("hallucination_rate", 0.0)
    revisions = state.get("revision_count", 0)

    if rate > 0.1 and revisions < 3:
        logger.info("Hallucination rate %.2f%% > 10%%. Retrying fact check.", rate * 100)
        return "fact_check"

    logger.info("Hallucination rate %.2f%%. Moving to preview.", rate * 100)
    return "save_preview"


def should_send_email(state: PipelineState) -> str:
    """Conditional edge after save_preview: send email or end."""
    if state.get("should_send_email"):
        return "email"
    return END


def build_pipeline_graph():
    """Build and compile the LangGraph state machine."""
    workflow = StateGraph(PipelineState)

    workflow.add_node("scrape", scrape_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("digest", digest_node)
    workflow.add_node("fact_check", fact_check_node)
    workflow.add_node("save_preview", save_preview_node)
    workflow.add_node("email", email_node)

    workflow.set_entry_point("scrape")
    workflow.add_edge("scrape", "retrieve")
    workflow.add_edge("retrieve", "digest")
    workflow.add_edge("digest", "fact_check")

    workflow.add_conditional_edges(
        "fact_check",
        should_retry_fact_check,
        {
            "fact_check": "fact_check",
            "save_preview": "save_preview",
        },
    )

    workflow.add_conditional_edges(
        "save_preview",
        should_send_email,
        {
            "email": "email",
            END: END,
        },
    )

    workflow.add_edge("email", END)

    return workflow.compile()


def visualize_graph():
    """Generate and save a Mermaid diagram of the pipeline."""
    import os
    os.makedirs("docs", exist_ok=True)

    graph = build_pipeline_graph()
    mermaid = graph.get_graph().draw_mermaid()
    print(mermaid)

    with open("docs/architecture.mmd", "w") as f:
        f.write(mermaid)

    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open("docs/architecture.png", "wb") as f:
            f.write(png_data)
        print("Saved architecture diagram to docs/architecture.png")
    except Exception as e:
        print(f"Could not generate PNG (need graphviz or similar): {e}")
        print("Mermaid source saved to docs/architecture.mmd")
