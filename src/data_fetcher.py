import os
import requests
import pandas as pd
from typing import Optional
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
        "description": "Slump test data with workability metrics."
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
        path = download_dataset(force=args.force)
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
