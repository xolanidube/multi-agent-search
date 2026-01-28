"""
Random Walk Strategy - Select a random unmarked cell.

Simple random selection with avoidance of already-searched cells.
Provides baseline performance and adds diversity to strategy mix.

Complexity: O(n) to get unmarked list, O(1) to select
Optimal when: No prior information, want unpredictable coverage
"""

from typing import Optional, List
import random

from .base import Strategy
from ..core.cell import Path
from ..core.marking_system import MarkingSystem
from ..core.searcher import Searcher


class RandomWalkStrategy(Strategy):
    """
    Select a random unmarked cell.

    Simple baseline strategy. Useful for:
    - Baseline comparison
    - Adding diversity to strategy mix
    - Breaking symmetry in multi-agent scenarios
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random walk strategy.

        Args:
            seed: Random seed for reproducibility
        """
        super().__init__(name="random_walk")
        if seed is not None:
            random.seed(seed)

    def get_next_cell(
        self,
        searcher: Searcher,
        path: Path,
        marks: MarkingSystem,
        other_positions: Optional[List[int]] = None
    ) -> Optional[int]:
        """
        Select a random unmarked cell.

        Args:
            searcher: The searcher using this strategy
            path: The search path
            marks: Current marking system state
            other_positions: Not used by this strategy

        Returns:
            ID of randomly selected unmarked cell, or None if all searched
        """
        unmarked = marks.get_unmarked_cells()

        if not unmarked:
            return None

        return random.choice(unmarked)


class BiasedRandomWalkStrategy(Strategy):
    """
    Random walk with bias towards nearby cells.

    Combines randomness with locality preference to reduce
    travel time while maintaining some unpredictability.
    """

    def __init__(self, locality_bias: float = 0.5, seed: Optional[int] = None):
        """
        Initialize biased random walk.

        Args:
            locality_bias: Probability of preferring nearby cells (0-1)
                          0 = pure random, 1 = always nearest
            seed: Random seed for reproducibility
        """
        super().__init__(name="biased_random_walk")
        self.locality_bias = locality_bias
        if seed is not None:
            random.seed(seed)

    def get_next_cell(
        self,
        searcher: Searcher,
        path: Path,
        marks: MarkingSystem,
        other_positions: Optional[List[int]] = None
    ) -> Optional[int]:
        """
        Select cell with bias towards nearby cells.

        Args:
            searcher: The searcher using this strategy
            path: The search path
            marks: Current marking system state
            other_positions: Not used

        Returns:
            ID of selected unmarked cell
        """
        unmarked = marks.get_unmarked_cells()

        if not unmarked:
            return None

        # With probability = locality_bias, pick nearest
        if random.random() < self.locality_bias:
            current_pos = searcher.position
            return min(unmarked, key=lambda c: abs(c - current_pos))

        # Otherwise, random
        return random.choice(unmarked)


class WeightedRandomWalkStrategy(Strategy):
    """
    Random walk weighted by cell probabilities.

    Cells with higher prior probability are more likely to be
    selected, but there's still randomness in the selection.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize weighted random walk.

        Args:
            seed: Random seed for reproducibility
        """
        super().__init__(name="weighted_random_walk")
        if seed is not None:
            random.seed(seed)

    def get_next_cell(
        self,
        searcher: Searcher,
        path: Path,
        marks: MarkingSystem,
        other_positions: Optional[List[int]] = None
    ) -> Optional[int]:
        """
        Select cell with probability proportional to prior.

        Args:
            searcher: The searcher using this strategy
            path: The search path
            marks: Current marking system state
            other_positions: Not used

        Returns:
            ID of probabilistically selected unmarked cell
        """
        unmarked = marks.get_unmarked_cells()

        if not unmarked:
            return None

        # Get probabilities for unmarked cells
        probs = [path[cell_id].prior_probability for cell_id in unmarked]

        # Normalize
        total = sum(probs)
        if total == 0:
            return random.choice(unmarked)

        probs = [p / total for p in probs]

        # Weighted random selection
        r = random.random()
        cumulative = 0.0
        for cell_id, prob in zip(unmarked, probs):
            cumulative += prob
            if r <= cumulative:
                return cell_id

        return unmarked[-1]  # Fallback
