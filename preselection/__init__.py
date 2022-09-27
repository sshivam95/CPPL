"""Various algorithms to solve preselection bandits problem."""
from preselection.double_thompson_sampling import DoubleThompsonSampling
from preselection.relative_confidence_sampling import RelativeConfidenceSampling
from preselection.upper_confidence_bound import UCB

regret_minimizing_algorithm = [UCB, DoubleThompsonSampling, RelativeConfidenceSampling]


__all__ = [
    preselection_algorithm.__name__
    for preselection_algorithm in regret_minimizing_algorithm
]
