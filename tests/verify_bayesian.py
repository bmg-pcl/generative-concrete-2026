import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bayesian import BayesFlowExplorer

def test_bayesian_updates():
    explorer = BayesFlowExplorer()
    
    print("Testing sample_posterior with carbon target...")
    samples_no_carbon = explorer.sample_posterior(target_strength=40.0, n_samples=100)
    samples_with_carbon = explorer.sample_posterior(target_strength=40.0, carbon_target=250.0, n_samples=100)
    
    print(f"Samples without carbon: {len(samples_no_carbon)}")
    print(f"Samples with carbon: {len(samples_with_carbon)}")
    
    print("\nTesting suggest_tests...")
    top_tests = explorer.suggest_tests(target_strength=45.0, carbon_target=300.0, n_tests=5)
    
    print("\nTop 5 Suggested Tests:")
    print(top_tests[["predicted_strength", "embodied_carbon", "uncertainty_score", "merit_score"]])
    
    assert len(top_tests) == 5
    assert "merit_score" in top_tests.columns
    assert "uncertainty_score" in top_tests.columns
    
    print("\nVerification Successful!")

if __name__ == "__main__":
    try:
        test_bayesian_updates()
    except Exception as e:
        print(f"Verification Failed: {e}")
        sys.exit(1)
