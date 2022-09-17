"""Various algorithms to solve preselection bandits problem."""
from preselection.upper_confidence_bound import UCB

regret_minimizing_algorithm = [UCB]
__all__ = [
    preselection_algorithm.__name__
    for preselection_algorithm in regret_minimizing_algorithm
]
