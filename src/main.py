from src.ga import GeneticOptimizer
from src.viz import plot_ga_progress, plot_parameter_heatmap
import numpy as np

def sphere_fn(x):
    """Simple test function: minimize sum of squares."""
    return -np.sum(x**2)  # Negative because GA maximizes

def test_optimizer_flow():
    bounds = [(-5.0, 5.0)] * 8
    param_names = [f"Param_{i}" for i in range(8)]
    
    optimizer = GeneticOptimizer(
        objective_fn=sphere_fn, 
        bounds=bounds, 
        pop_size=50, 
        maximize=True
    )

    print("Running optimization...")
    # Run 50 generations
    for _ in range(50):
        stats = optimizer.step()
        if stats["generation"] % 10 == 0:
            print(f"Gen {stats['generation']}: Best Fitness = {stats['best_fitness']:.4f}")

    best_v, best_f = optimizer.get_best()
    print(f"\nFinal Best Fitness: {best_f:.4f}")
    
    # In a real CLI env, plots might not show, but we can verify they don't crash
    # plot_ga_progress(optimizer.history)
    # plot_parameter_heatmap(best_v, param_names)

if __name__ == "__main__":
    test_optimizer_flow()
