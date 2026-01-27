"""
Simulated Annealing optimizer for concrete mix design.
"""
import numpy as np
from typing import Callable, List, Tuple, Dict, Any

class SimulatedAnnealing:
    """
    Simulated Annealing optimizer for continuous optimization problems.
    
    Inspired by the metallurgical annealing process where controlled cooling
    allows atoms to find low-energy configurations.
    """
    
    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.95,
        min_temp: float = 1e-6,
        max_iter_per_temp: int = 100,
        maximize: bool = True
    ):
        self.objective_fn = objective_fn
        self.bounds = np.array(bounds)
        self.n_dims = len(bounds)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iter_per_temp = max_iter_per_temp
        self.maximize = maximize
        
        # Initialize with random solution
        self.current = self._random_solution()
        self.current_fitness = self._evaluate(self.current)
        self.best = self.current.copy()
        self.best_fitness = self.current_fitness
        
        self.temperature = initial_temp
        self.history = []
        self.iteration = 0
    
    def _random_solution(self) -> np.ndarray:
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
    
    def _evaluate(self, x: np.ndarray) -> float:
        return self.objective_fn(x)
    
    def _neighbor(self, x: np.ndarray) -> np.ndarray:
        """Generate a neighboring solution by perturbing current."""
        # Step size proportional to temperature
        step_scale = (self.temperature / self.initial_temp) * 0.3
        neighbor = x.copy()
        
        # Perturb a random subset of dimensions
        n_perturb = max(1, int(self.n_dims * 0.3))
        dims_to_perturb = np.random.choice(self.n_dims, n_perturb, replace=False)
        
        for dim in dims_to_perturb:
            range_size = self.bounds[dim, 1] - self.bounds[dim, 0]
            perturbation = np.random.normal(0, step_scale * range_size)
            neighbor[dim] = np.clip(
                neighbor[dim] + perturbation,
                self.bounds[dim, 0],
                self.bounds[dim, 1]
            )
        
        return neighbor
    
    def _acceptance_probability(self, current_fit: float, new_fit: float) -> float:
        """Metropolis acceptance criterion."""
        if self.maximize:
            delta = new_fit - current_fit
        else:
            delta = current_fit - new_fit
        
        if delta > 0:
            return 1.0
        else:
            return np.exp(delta / self.temperature)
    
    def step(self) -> Dict[str, Any]:
        """Perform one temperature step (multiple iterations)."""
        accepted = 0
        
        for _ in range(self.max_iter_per_temp):
            neighbor = self._neighbor(self.current)
            neighbor_fitness = self._evaluate(neighbor)
            
            if np.random.random() < self._acceptance_probability(self.current_fitness, neighbor_fitness):
                self.current = neighbor
                self.current_fitness = neighbor_fitness
                accepted += 1
                
                # Update best
                if self.maximize and neighbor_fitness > self.best_fitness:
                    self.best = neighbor.copy()
                    self.best_fitness = neighbor_fitness
                elif not self.maximize and neighbor_fitness < self.best_fitness:
                    self.best = neighbor.copy()
                    self.best_fitness = neighbor_fitness
        
        # Cool down
        self.temperature *= self.cooling_rate
        self.iteration += 1
        
        stats = {
            "iteration": self.iteration,
            "temperature": self.temperature,
            "current_fitness": self.current_fitness,
            "best_fitness": self.best_fitness,
            "acceptance_rate": accepted / self.max_iter_per_temp
        }
        self.history.append(stats)
        
        return stats
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """Run full optimization until temperature drops below minimum."""
        while self.temperature > self.min_temp:
            stats = self.step()
            if verbose and self.iteration % 10 == 0:
                print(f"Iter {self.iteration}: T={stats['temperature']:.2f}, Best={stats['best_fitness']:.4f}")
        
        return self.best, self.best_fitness
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        return self.best, self.best_fitness


if __name__ == "__main__":
    # Test with Sphere function
    def sphere(x):
        return -np.sum(x**2)  # Negative for maximization
    
    bounds = [(-5, 5)] * 5
    sa = SimulatedAnnealing(sphere, bounds, initial_temp=100, cooling_rate=0.9)
    best, fitness = sa.optimize()
    print(f"\nBest solution: {best}")
    print(f"Best fitness: {fitness}")
