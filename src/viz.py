import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any

def plot_ga_progress(history: List[Dict[str, Any]]):
    """Plots best and average fitness over generations with a dark theme."""
    plt.style.use('dark_background')
    gens = [h["generation"] for h in history]
    best = [h["best_fitness"] for h in history]
    avg = [h["avg_fitness"] for h in history]

    plt.figure(figsize=(10, 5), facecolor='#121212')
    plt.plot(gens, best, label="Best Fitness", color="#00E676", linewidth=2)
    plt.plot(gens, avg, label="Average Fitness", color="#2979FF", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Optimization Progress (GA)", color="white")
    plt.legend()
    plt.grid(True, alpha=0.1)
    plt.show()

def plot_parameter_heatmap(best_individual: np.ndarray, parameter_names: List[str]):
    """Presents a high-res heatmap of the parameters of the best candidate in dark mode."""
    data = best_individual.reshape(1, -1)
    
    fig = px.imshow(
        data,
        labels=dict(x="Parameters", y="Candidate", color="Normalized Value"),
        x=parameter_names,
        y=["Best Mix"],
        color_continuous_scale="Magma",
        template="plotly_dark",
        aspect="auto"
    )
    fig.update_layout(
        title="Best Mix Design Parameter Profile",
        paper_bgcolor="#121212",
        plot_bgcolor="#121212"
    )
    fig.show()

def plot_posterior_3d(samples: np.ndarray, param_indices: Tuple[int, int], param_names: List[str]):
    """Creates a 3D mesh view of the posterior distribution for two selected parameters."""
    x = samples[:, param_indices[0]]
    y = samples[:, param_indices[1]]
    
    # Kernel Density Estimation for smooth mesh
    from scipy.stats import gaussian_kde
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    
    # Create grid
    xi, yi = np.mgrid[x.min():x.max():50j, y.min():y.max():50j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    fig = go.Figure(data=[go.Surface(
        z=zi, x=xi, y=yi, 
        colorscale='Viridis',
        opacity=0.9,
        lighting=dict(ambient=0.4, diffuse=0.5, roughness=0.1, specular=1.0)
    )])
    
    fig.update_layout(
        title=f"3D Amortized Posterior: {param_names[param_indices[0]]} vs {param_names[param_indices[1]]}",
        scene=dict(
            xaxis_title=param_names[param_indices[0]],
            yaxis_title=param_names[param_indices[1]],
            zaxis_title="Probability Density",
            bgcolor="#121212"
        ),
        template="plotly_dark",
        paper_bgcolor="#121212",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()

def plot_bayesflow_diagnostics(loss_history: List[float]):
    """Visualizes training convergence for the BayesFlow amortizer."""
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 4), facecolor='#121212')
    plt.plot(loss_history, color="#FF5252", alpha=0.8)
    plt.title("BayesFlow Amortizer Convergence (KLD Loss)")
    plt.xlabel("Training Step")
    plt.ylabel("Negative Log-Likelihood")
    plt.yscale('log')
    plt.grid(True, alpha=0.1)
    plt.show()
