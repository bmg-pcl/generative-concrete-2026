# CLAUDE.md - Genetic Algorithm Experiment

## Project Overview
This project focuses on implementing a Genetic Algorithm (GA) with interactive visualizations. It starts as a Jupyter Notebook (`.ipynb`) for exploration and will expand into a Streamlit application.

## Development Commands
- **Install Dependencies**: `uv pip install -r requirements.txt` or `pip install -r requirements.txt`
- **Run Jupyter Lab**: `jupyter lab`
- **Run Streamlit App**: `streamlit run app.py`
- **Run CLI Test**: `python -m src.main` (or specific module tests)
- **Linting/Formatting**: `uv run ruff check .` / `uv run ruff format .` (if installed)

## Project Structure
- `notebooks/`: Initial exploration and visualizations.
- `src/`: Core Genetic Algorithm logic.
  - `src/ga.py`: Genetic Algorithm engine.
  - `src/main.py`: CLI entry point for testing and running simulations.
- `app.py`: Streamlit application entry point.
- `requirements.txt`: Python package dependencies.

## Coding Style & Patterns
- **Environment**: Use `uv` for fast package management. Avoid `conda`.
- **Python**: Use type hints for all functions. 
- **CLI Ready**: Ensure core logic is importable and can be run via a CLI entry point (e.g., `if __name__ == "__main__":`).
- **Genetic Algorithm**:
  - Keep core logic (selection, crossover, mutation) decoupled from visualization and UI.
  - Implement configurable parameters via CLI arguments or config dicts.
- **Visualizations**: 
  - Notebooks: `ipywidgets`, `plotly`.
  - Streamlit: Native elements.
- **Testing**: Structure code so modules in `src/` can be tested independently.

## Dependencies (Initial)
- `numpy`: Numerical operations.
-  `plotly`: Visualizations.
- `ipywidgets`: Interactive UI in notebooks.
- `streamlit`: Dashboarding.



## Project Details

-basic idea is to use the UIUC concrete mix design dataset to train a very simple (statical learning) base model, probably boosted decision forest, and then to add additional admixture and carbon questions (transport distance, clinker production method, waste factor  etc) to synthesize different concrete mix design candidates, and then evaluate them on a few different imputed characteristics:

- compressive strength
- tensile strength
- curing time
- embodied carbon via chemical analysis and factoring 

This should be presented to users as an A/B test of different inputs. 

A secondary optimization phase after the A/(optional B) test can be  offered to find the best mix design for a given set of inputs, calculating the pareto front of the mix designs and then offering the user the best mix design for their inputs.

Provide the means to import other datasets to retrain the model on, e.g. base UIUC dataset or others in the same format. 

Provide a final area of bayesian amortization to evaluate the *empty spaces* of the mix design space   . Use pydata modules not R.  Explain this - it will be unfamiliar to users. 

Quality of visualizations of the optimziation are important. Visualize inline in the notebook at least the GA optimization process (generation by generation) and the "best" and "average" of the population.  Present a high-res heatmap of the parameters of the best candidates at the end. 

Later we'll use annealing or ACO so keep it generalised. 


Make a test suite. 

Download the UIUC dataset and make a test suite for it. 