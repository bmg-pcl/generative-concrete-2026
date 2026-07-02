"""UI layer for the Streamlit app.

Each tab is a `render(ctx)` function in its own module; `ui/config.py` builds the
`AppContext` and the other renderers consume it. Pure number-crunching stays in
`src/ui_logic.py` — modules here hold only Streamlit wiring.
"""
