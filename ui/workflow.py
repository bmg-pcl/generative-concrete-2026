"""Workflow tab — the deeper reference docs/WORKFLOW.md provides beyond the sidebar.

R8.6 WP-U1: the sidebar's "Step-by-step guide" expander already gives the numbered
walkthrough (Config -> Compare Mixes -> Inverse Design -> Pareto Optimization ->
Calibration), so this tab used to be a second, longer copy of the same steps. It now
renders only what the sidebar does NOT cover: the "three questions, three tools"
table (section 1) and the decision guidance on how the amortized flow and NSGA-II/III
compose plus the loop summary (sections 3-4). Section 2 ("The workflow, in order" —
the mermaid flow diagram and its step-by-step prose) is dropped here; the surviving
sections' text is rendered verbatim, unaltered.
"""
import re

import streamlit as st

_DROP_STEP_BY_STEP = re.compile(
    r"\n---\n\n## 2\. The workflow, in order.*?\n---\n\n(?=## 3\. How the amortized flow)",
    re.DOTALL,
)


def render_workflow():
    with open("docs/WORKFLOW.md", "r", encoding="utf-8") as f:
        workflow_md = f.read()
    workflow_md = _DROP_STEP_BY_STEP.sub("\n", workflow_md)
    st.markdown(workflow_md)
