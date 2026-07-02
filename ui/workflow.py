"""Workflow tab — renders docs/WORKFLOW.md (mermaid stripped for Streamlit)."""
import re

import streamlit as st


def render_workflow():
    with open("docs/WORKFLOW.md", "r", encoding="utf-8") as f:
        workflow_md = f.read()
    # Streamlit's markdown doesn't render mermaid; drop the diagram block (it renders
    # on GitHub) and keep the ordered step-by-step text.
    workflow_md = re.sub(r"```mermaid.*?```",
                         "_(flow diagram renders on GitHub — the ordered steps are below)_",
                         workflow_md, flags=re.DOTALL)
    st.markdown(workflow_md)
