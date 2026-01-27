from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Tuple
import numpy as np

class Optimizer(ABC):
    """Base class for all optimization algorithms (GA, Annealing, ACO)."""
    
    def __init__(self, 
                 objective_fn: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]],
                 maximize: bool = True):
        self.objective_fn = objective_fn
        self.bounds = np.array(bounds)
        self.maximize = maximize
        self.history = []  # Stores generation/step metrics

    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """Performs a single iteration (generation or step)."""
        pass

    @abstractmethod
    def run(self, iterations: int) -> Dict[str, Any]:
        """Runs the optimizer for a set number of iterations."""
        pass

    def get_best(self) -> Tuple[np.ndarray, float]:
        """Returns the best solution and its fitness found so far."""
        pass
