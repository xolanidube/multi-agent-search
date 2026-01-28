"""
Search strategies adapted for 2D grid exploration.

These strategies determine how agents navigate and explore
a 2D grid space efficiently.
"""

from typing import Optional, List, Tuple, Set
from abc import ABC, abstractmethod
import random
import math
import heapq

from ..core.grid import Grid2D, GridAgent, GridCellStatus


class GridStrategy(ABC):
    """Base class for 2D grid search strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_next_cell(
        self,
        agent: GridAgent,
        grid: Grid2D,
        other_agents: Optional[List[GridAgent]] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Determine next cell to explore.

        Args:
            agent: The agent using this strategy
            grid: The 2D grid
            other_agents: Other agents (for coordination)

        Returns:
            Next cell position or None if nothing available
        """
        pass


class SpiralStrategy(GridStrategy):
    """
    Spiral outward from starting position.

    Efficient for localized search when target is likely near start.
    """

    def __init__(self):
        super().__init__("spiral")
        self.direction_idx = 0
        self.steps_in_direction = 1
        self.steps_taken = 0
        self.direction_changes = 0

    def get_next_cell(
        self,
        agent: GridAgent,
        grid: Grid2D,
        other_agents: Optional[List[GridAgent]] = None
    ) -> Optional[Tuple[int, int]]:
        # Directions: right, down, left, up
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        x, y = agent.position

        # Try spiral movement
        for _ in range(len(grid)):
            dx, dy = directions[self.direction_idx % 4]
            nx, ny = x + dx, y + dy

            self.steps_taken += 1

            if self.steps_taken >= self.steps_in_direction:
                self.steps_taken = 0
                self.direction_idx += 1
                self.direction_changes += 1
                if self.direction_changes % 2 == 0:
                    self.steps_in_direction += 1

            if grid.is_valid(nx, ny) and grid.is_available((nx, ny)):
                return (nx, ny)

            x, y = nx, ny

        # Fallback: any unexplored cell
        unexplored = grid.get_unexplored()
        if unexplored:
            return min(unexplored, key=lambda p: grid.distance(agent.position, p))
        return None


class GreedyGridStrategy(GridStrategy):
    """
    Always move to nearest unexplored cell.

    Simple and effective for general exploration.
    """

    def __init__(self):
        super().__init__("greedy_nearest")

    def get_next_cell(
        self,
        agent: GridAgent,
        grid: Grid2D,
        other_agents: Optional[List[GridAgent]] = None
    ) -> Optional[Tuple[int, int]]:
        unexplored = grid.get_unexplored()
        if not unexplored:
            return None

        # Find nearest unexplored
        return min(unexplored, key=lambda p: grid.distance(agent.position, p))


class QuadrantStrategy(GridStrategy):
    """
    Divide grid into quadrants, each agent takes one.

    Good for multi-agent scenarios to reduce overlap.
    """

    def __init__(self, num_agents: int = 1, agent_idx: int = 0):
        super().__init__("quadrant")
        self.num_agents = num_agents
        self.agent_idx = agent_idx

    def get_next_cell(
        self,
        agent: GridAgent,
        grid: Grid2D,
        other_agents: Optional[List[GridAgent]] = None
    ) -> Optional[Tuple[int, int]]:
        # Determine quadrant based on agent index
        cols = int(math.ceil(math.sqrt(self.num_agents)))
        rows = int(math.ceil(self.num_agents / cols))

        col = self.agent_idx % cols
        row = self.agent_idx // cols

        quad_width = grid.width // cols
        quad_height = grid.height // rows

        x_start = col * quad_width
        x_end = x_start + quad_width if col < cols - 1 else grid.width
        y_start = row * quad_height
        y_end = y_start + quad_height if row < rows - 1 else grid.height

        # Find unexplored in quadrant
        quadrant_unexplored = [
            p for p in grid.get_unexplored()
            if x_start <= p[0] < x_end and y_start <= p[1] < y_end
        ]

        if quadrant_unexplored:
            return min(quadrant_unexplored, key=lambda p: grid.distance(agent.position, p))

        # Fallback to any unexplored
        unexplored = grid.get_unexplored()
        if unexplored:
            return min(unexplored, key=lambda p: grid.distance(agent.position, p))

        return None


class ProbabilisticGridStrategy(GridStrategy):
    """
    Prioritize cells with higher prior probability.

    Useful when there's knowledge about likely target locations.
    """

    def __init__(self, exploitation_factor: float = 0.8):
        super().__init__("probabilistic")
        self.exploitation_factor = exploitation_factor

    def get_next_cell(
        self,
        agent: GridAgent,
        grid: Grid2D,
        other_agents: Optional[List[GridAgent]] = None
    ) -> Optional[Tuple[int, int]]:
        unexplored = grid.get_unexplored()
        if not unexplored:
            return None

        if random.random() < self.exploitation_factor:
            # Exploit: go to highest probability
            return max(unexplored, key=lambda p: grid.cells[p].prior_probability)
        else:
            # Explore: random selection weighted by distance (prefer closer)
            weights = [1.0 / (1 + grid.distance(agent.position, p)) for p in unexplored]
            total = sum(weights)
            weights = [w / total for w in weights]

            r = random.random()
            cumulative = 0
            for i, p in enumerate(unexplored):
                cumulative += weights[i]
                if r <= cumulative:
                    return p
            return unexplored[-1]


class SwarmStrategy(GridStrategy):
    """
    Coordinate with other agents to maximize coverage.

    Agents repel each other to spread out exploration.
    """

    def __init__(self, repulsion_strength: float = 2.0):
        super().__init__("swarm")
        self.repulsion_strength = repulsion_strength

    def get_next_cell(
        self,
        agent: GridAgent,
        grid: Grid2D,
        other_agents: Optional[List[GridAgent]] = None
    ) -> Optional[Tuple[int, int]]:
        unexplored = grid.get_unexplored()
        if not unexplored:
            return None

        if not other_agents:
            # No others, just go to nearest
            return min(unexplored, key=lambda p: grid.distance(agent.position, p))

        # Score cells: prefer unexplored far from other agents
        def score(pos):
            # Distance to nearest other agent
            min_agent_dist = min(
                grid.euclidean_distance(pos, a.position)
                for a in other_agents if a.id != agent.id
            ) if len(other_agents) > 1 else float('inf')

            # Distance from current position (prefer closer)
            travel_dist = grid.distance(agent.position, pos)

            # Score: maximize agent separation, minimize travel
            return self.repulsion_strength * min_agent_dist - travel_dist

        return max(unexplored, key=score)


class WavefrontStrategy(GridStrategy):
    """
    BFS-like exploration expanding from starting point.

    Guarantees systematic coverage of reachable cells.
    """

    def __init__(self):
        super().__init__("wavefront")
        self.frontier: List[Tuple[int, int]] = []
        self.initialized = False

    def get_next_cell(
        self,
        agent: GridAgent,
        grid: Grid2D,
        other_agents: Optional[List[GridAgent]] = None
    ) -> Optional[Tuple[int, int]]:
        # Initialize frontier from current position
        if not self.initialized:
            self.frontier = list(grid.get_neighbors(*agent.position))
            self.initialized = True

        # Remove explored cells from frontier
        self.frontier = [p for p in self.frontier if grid.is_available(p)]

        if not self.frontier:
            # Expand frontier
            unexplored = grid.get_unexplored()
            if unexplored:
                # Jump to nearest unexplored
                nearest = min(unexplored, key=lambda p: grid.distance(agent.position, p))
                self.frontier = [nearest]
            else:
                return None

        # Get nearest frontier cell
        next_cell = min(self.frontier, key=lambda p: grid.distance(agent.position, p))
        self.frontier.remove(next_cell)

        # Add neighbors of next cell to frontier
        for neighbor in grid.get_neighbors(*next_cell):
            if neighbor not in self.frontier and grid.is_available(neighbor):
                self.frontier.append(neighbor)

        return next_cell


class RandomGridStrategy(GridStrategy):
    """
    Random walk with exploration bias.

    Baseline strategy for comparison.
    """

    def __init__(self):
        super().__init__("random")

    def get_next_cell(
        self,
        agent: GridAgent,
        grid: Grid2D,
        other_agents: Optional[List[GridAgent]] = None
    ) -> Optional[Tuple[int, int]]:
        # Prefer unexplored neighbors
        neighbors = grid.get_neighbors(*agent.position)
        unexplored_neighbors = [n for n in neighbors if grid.is_available(n)]

        if unexplored_neighbors:
            return random.choice(unexplored_neighbors)

        # Jump to random unexplored
        unexplored = grid.get_unexplored()
        if unexplored:
            return random.choice(unexplored)

        return None
