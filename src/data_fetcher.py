import os
import requests
import pandas as pd
import argparse

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Available datasets for concrete research
AVAILABLE_DATASETS = {
    "uci_yeh": {
        "name": "UCI Concrete (Yeh, 1998)",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        "format": "xls",
        "samples": 1030,
        "description": "Classic benchmark dataset from Taiwan. 8 features, 28-day strength."
    },
    "kaggle_slump": {
        "name": "Kaggle Concrete Slump",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data",
        "format": "csv",
        "samples": 103,
        "description": "Slump test data with workability metrics.",
        # R8.1: the canonical UCI host is unreachable from this environment (403 on
        # CONNECT at the proxy gateway), so this corpus is COMMITTED to the repo
        # rather than fetched via `url` above -- see load_slump_data() and
        # data/slump_test.PROVENANCE.md for the mirror source and integrity checks.
        "committed": True,
        "local_file": "data/slump_test.data",
        "provenance": "data/slump_test.PROVENANCE.md",
    },
    # Note: These would require manual download or API keys
    "aci_mix_design": {
        "name": "ACI Mix Design Reference (Manual)",
        "url": None,
        "format": "csv",
        "samples": "varies",
        "description": "Standard ACI 211.1 mix proportioning tables. Requires manual entry."
    },
    "dot_highway": {
        "name": "State DOT Highway Pavements (Manual)",
        "url": None,
        "format": "csv",
        "samples": "varies",
        "description": "High-performance pavement mixes from state DOT databases."
    }
}

LOCAL_FILE = os.path.join(DATA_DIR, "Concrete_Data.xls")
OVERLAY_FILE = os.path.join(DATA_DIR, "Experimental_Overlay.csv")
SLUMP_LOCAL_FILE = os.path.join(DATA_DIR, "slump_test.data")

# Dataset column -> repo key, per docs/specs/R8.1's mapping table. "No" (row index)
# is dropped; "Compressive Strength (28-day)(Mpa)" is kept as "strength" for
# completeness/inspection only -- see load_slump_data()'s docstring for why it is
# never used to train anything.
_SLUMP_COLUMN_MAP = {
    "Cement": "cement",
    "Slag": "slag",
    "Fly ash": "ash",
    "Water": "water",
    "SP": "superplasticizer",
    "Coarse Aggr.": "coarse_agg",
    "Fine Aggr.": "fine_agg",
    "SLUMP(cm)": "slump_cm",
    "FLOW(cm)": "flow_cm",
    "Compressive Strength (28-day)(Mpa)": "strength",
}
_SLUMP_COLUMN_ORDER = [
    "cement", "slag", "ash", "water", "superplasticizer", "coarse_agg", "fine_agg",
    "slump_cm", "flow_cm", "strength",
]

# Default URL for backward compatibility
UCI_URL = AVAILABLE_DATASETS["uci_yeh"]["url"]

def download_dataset(url: str = None, force: bool = False) -> str:
    """Downloads the UIUC/UCI Concrete dataset if it doesn't exist."""
    if url is None:
        url = UCI_URL
        
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    if os.path.exists(LOCAL_FILE) and not force:
        print(f"Dataset already exists at {LOCAL_FILE}")
        return LOCAL_FILE

    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, verify=True)
    except requests.exceptions.SSLError:
        print("SSL Verification failed. Retrying without verification (CAUTION)...")
        response = requests.get(url, verify=False)
    
    response.raise_for_status()
    
    with open(LOCAL_FILE, "wb") as f:
        f.write(response.content)
    
    print(f"Dataset saved to {LOCAL_FILE}")
    return LOCAL_FILE

def load_data() -> pd.DataFrame:
    """Loads the dataset into a pandas DataFrame, merging with local overlays."""
    if not os.path.exists(LOCAL_FILE):
        download_dataset()
    
    # UCI Concrete dataset is an Excel file
    df = pd.read_excel(LOCAL_FILE)
    
    # Standardize column names
    new_columns = [
        "cement", "slag", "ash", "water", "superplasticizer", 
        "coarse_agg", "fine_agg", "age", "strength"
    ]
    df.columns = new_columns
    
    # Merge with experimental overlay if exists
    if os.path.exists(OVERLAY_FILE):
        overlay_df = pd.read_csv(OVERLAY_FILE)
        df = pd.concat([df, overlay_df], ignore_index=True)
        print(f"Merged {len(overlay_df)} local experimental records.")
        
    return df

def load_slump_data() -> pd.DataFrame:
    """Loads the COMMITTED UCI concrete slump corpus (103 rows) and applies the
    dataset -> repo key column mapping from docs/specs/R8.1.

    Unlike ``load_data()``, this never downloads: the canonical UCI host
    (archive.ics.uci.edu) is blocked from this environment, so ``slump_test.data`` is
    committed to the repo (mirror-sourced and integrity-checked -- see
    ``data/slump_test.PROVENANCE.md``) precisely so this function never needs network
    access, at import time or otherwise. A missing file is a repo problem, not a
    "go fetch it" situation, so this raises rather than falling back to a download.

    The corpus's own ``Compressive Strength (28-day)(Mpa)`` column is loaded and
    kept as ``strength`` for completeness/inspection, but it is NOT used to train any
    model in this repo: it is a different 103-row experiment from a different lab
    than the 1030-row strength corpus ``load_data()`` returns, and mixing the two is a
    data-provenance decision with its own validation burden -- explicitly out of scope
    for R8.1. Callers that want the strength model use ``load_data()`` /
    ``StrengthPredictor``, never this column.
    """
    if not os.path.exists(SLUMP_LOCAL_FILE):
        raise FileNotFoundError(
            f"{SLUMP_LOCAL_FILE} not found. This corpus is committed to the repo (see "
            "data/slump_test.PROVENANCE.md); it is not downloaded at runtime, so a "
            "missing file means the checkout is incomplete, not that a fetch is needed."
        )
    df = pd.read_csv(SLUMP_LOCAL_FILE)
    df = df.drop(columns=["No"])
    df = df.rename(columns=_SLUMP_COLUMN_MAP)
    return df[_SLUMP_COLUMN_ORDER]


def append_experimental_results(results_df: pd.DataFrame):
    """Appends new lab results to the local overlay."""
    if os.path.exists(OVERLAY_FILE):
        existing = pd.read_csv(OVERLAY_FILE)
        updated = pd.concat([existing, results_df], ignore_index=True)
        updated.to_csv(OVERLAY_FILE, index=False)
    else:
        results_df.to_csv(OVERLAY_FILE, index=False)
    print(f"Appended {len(results_df)} results to {OVERLAY_FILE}")

def main():
    parser = argparse.ArgumentParser(description="UCI Concrete Dataset Fetcher")
    parser.add_argument("--check", action="store_true", help="Check if dataset exists and try loading it")
    parser.add_argument("--force", action="store_true", help="Force download even if file exists")
    args = parser.parse_args()

    try:
        download_dataset(force=args.force)
        if args.check:
            df = load_data()
            print("\nDataset loaded successfully!")
            print(f"Shape: {df.shape}")
            print("\nFirst 5 rows:")
            print(df.head())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
