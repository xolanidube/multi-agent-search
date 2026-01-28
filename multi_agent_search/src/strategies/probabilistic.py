"""
Probabilistic Hot-Zone Strategy - Weight cells by prior probability.

Uses the prior probability distribution over cells to prioritize
high-probability "hot zones" like rest stops, obstacles, or steep
inclines where items are more likely to fall.

Complexity: O(n) to scan all cells
Optimal when: Good prior model exists for drop likelihood
"""

from typing import Optional, List

from .base import Strategy
from ..core.cell import Path
from ..core.marking_system import MarkingSystem
from ..core.searcher import Searcher


class ProbabilisticStrategy(Strategy):
    """
    Select cells based on prior probability distribution.

    Prioritizes high-probability "hot zones" where the key
    is most likely to be. Effective when prior information
    is available and accurate.
    """

    def __init__(self):
        super().__init__(name="probabilistic")

    def get_next_cell(
        self,
        searcher: Searcher,
        path: Path,
        marks: MarkingSystem,
        other_positions: Optional[List[int]] = None
    ) -> Optional[int]:
        """
        Find the unmarked cell with highest prior probability.

        Args:
            searcher: The searcher using this strategy
            path: The search path
            marks: Current marking system state
            other_positions: Not used by this strategy

        Returns:
            ID of highest-probability unmarked cell, or None if all searched
        """
        unmarked = marks.get_unmarked_cells()

        if not unmarked:
            return None

        # Find unmarked cell with highest prior probability
        best_cell = None
        best_prob = -1.0

        for cell_id in unmarked:
            cell = path[cell_id]
            prob = cell.prior_probability
            if prob > best_prob:
                best_prob = prob
                best_cell = cell_id

        return best_cell


class AdaptiveProbabilisticStrategy(Strategy):
    """
    Probabilistic strategy with distance penalty.

    Balances probability against travel cost. Useful when
    high-probability cells are far away.
    """

    def __init__(self, distance_weight: float = 0.3):
        """
        Initialize with distance weighting.

        Args:
            distance_weight: How much to penalize distance (0-1)
                           0 = pure probability, 1 = pure distance
        """
        super().__init__(name="adaptive_probabilistic")
        self.distance_weight = distance_weight

    def get_next_cell(
        self,
        searcher: Searcher,
        path: Path,
        marks: MarkingSystem,
        other_positions: Optional[List[int]] = None
    ) -> Optional[int]:
        """
        Find cell with best probability-distance tradeoff.

        Score = probability * (1 - distance_weight) - distance * distance_weight

        Args:
            searcher: The searcher using this strategy
            path: The search path
            marks: Current marking system state
            other_positions: Not used

        Returns:
            ID of best-scoring unmarked cell
        """
        unmarked = marks.get_unmarked_cells()

        if not unmarked:
            return None

        current_pos = searcher.position
        path_length = len(path)

        best_cell = None
        best_score = float('-inf')

        for cell_id in unmarked:
            cell = path[cell_id]
            prob = cell.prior_probability

            # Normalize distance to 0-1
            distance = abs(cell_id - current_pos) / path_length

            # Calculate weighted score
            score = prob * (1 - self.distance_weight) - distance * self.distance_weight

            if score > best_score:
                best_score = score
                best_cell = cell_id

        return best_cell
