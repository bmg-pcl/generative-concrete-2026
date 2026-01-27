import numpy as np
from typing import List, Dict, Any, Callable, Tuple
from .base_optimizer import Optimizer

class GeneticOptimizer(Optimizer):
    """Genetic Algorithm implementation of the Optimizer."""
    
    def __init__(self, 
                 objective_fn: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]],
                 pop_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 maximize: bool = True):
        super().__init__(objective_fn, bounds, maximize)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Initialize population within bounds
        self.population = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            (self.pop_size, len(self.bounds))
        )
        self.fitness = np.zeros(self.pop_size)
        self.generation = 0
        self.best_individual = None
        self.best_fitness = -np.inf if maximize else np.inf

    def _evaluate(self):
        self.fitness = np.array([self.objective_fn(ind) for ind in self.population])
        
        current_best_idx = np.argmax(self.fitness) if self.maximize else np.argmin(self.fitness)
        current_best_fitness = self.fitness[current_best_idx]
        
        if self.maximize:
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = self.population[current_best_idx].copy()
        else:
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = self.population[current_best_idx].copy()

    def step(self) -> Dict[str, Any]:
        """Runs one generation of GA."""
        self._evaluate()
        
        # Selection (Tournament)
        new_population = []
        for _ in range(self.pop_size):
            i, j = np.random.randint(0, self.pop_size, 2)
            if self.maximize:
                winner = self.population[i] if self.fitness[i] > self.fitness[j] else self.population[j]
            else:
                winner = self.population[i] if self.fitness[i] < self.fitness[j] else self.population[j]
            new_population.append(winner.copy())
        new_population = np.array(new_population)

        # Crossover
        for i in range(0, self.pop_size, 2):
            if i + 1 < self.pop_size and np.random.rand() < self.crossover_rate:
                cp = np.random.randint(1, len(self.bounds))
                parent1, parent2 = new_population[i].copy(), new_population[i+1].copy()
                new_population[i, cp:], new_population[i+1, cp:] = parent2[cp:], parent1[cp:]

        # Mutation
        mask = np.random.rand(*new_population.shape) < self.mutation_rate
        mutation_values = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            new_population.shape
        )
        new_population[mask] = mutation_values[mask]

        self.population = new_population
        self.generation += 1
        
        stats = {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": np.mean(self.fitness),
            "pop_std": np.std(self.fitness)
        }
        self.history.append(stats)
        return stats

    def run(self, iterations: int) -> Dict[str, Any]:
        for _ in range(iterations):
            self.step()
        return self.get_best()

    def get_best(self) -> Tuple[np.ndarray, float]:
        return self.best_individual, self.best_fitness
