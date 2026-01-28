"""
Cell and Path data structures for the search space.

A search space is a discrete path P = {c_0, c_1, ..., c_{n-1}} of n cells,
where each cell has properties like search time, traversal cost, and
prior probability of containing the key.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import random


class CellStatus(Enum):
    """Status of a cell in the search space."""
    UNMARKED = "unmarked"
    IN_PROGRESS = "in_progress"
    SEARCHED = "searched"
    FOUND = "found"


class TerrainType(Enum):
    """Types of terrain affecting search difficulty."""
    NORMAL = "normal"
    DENSE = "dense"        # Harder to search (e.g., thick brush)
    OPEN = "open"          # Easy to search (e.g., clearing)
    ROCKY = "rocky"        # Variable difficulty
    WATER = "water"        # Special handling needed


@dataclass
class Cell:
    """
    Represents a searchable cell in the path.

    Attributes:
        id: Unique identifier for the cell
        search_time: Base time required to search this cell (seconds)
        traversal_cost: Cost to move to this cell from adjacent cell
        detection_modifier: Multiplier on detection probability (1.0 = normal)
        terrain_type: Type of terrain in this cell
        prior_probability: Prior probability that key is in this cell
        has_key: Whether this cell contains the key (hidden from searchers)
    """
    id: int
    search_time: float = 1.0
    traversal_cost: float = 1.0
    detection_modifier: float = 1.0
    terrain_type: TerrainType = TerrainType.NORMAL
    prior_probability: float = 0.0  # Will be normalized across path
    has_key: bool = False

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Cell):
            return self.id == other.id
        return False


@dataclass
class Path:
    """
    Represents the complete search space as a linear path of cells.

    The path models the trail that was hiked, discretized into
    searchable cells. The key was dropped somewhere along this path.
    """
    cells: List[Cell] = field(default_factory=list)
    key_position: Optional[int] = None  # Index of cell containing key

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, idx: int) -> Cell:
        return self.cells[idx]

    def __iter__(self):
        return iter(self.cells)

    @classmethod
    def generate(
        cls,
        length: int,
        key_position: Optional[int] = None,
        heterogeneous: bool = False,
        seed: Optional[int] = None
    ) -> 'Path':
        """
        Generate a path with specified properties.

        Args:
            length: Number of cells in the path
            key_position: Where to place the key (random if None)
            heterogeneous: If True, cells have varying properties
            seed: Random seed for reproducibility

        Returns:
            A new Path instance
        """
        if seed is not None:
            random.seed(seed)

        cells = []
        for i in range(length):
            if heterogeneous:
                # Create varied terrain
                terrain = random.choice(list(TerrainType))
                search_time = cls._terrain_search_time(terrain)
                # Scaled down traversal costs for faster simulation
                traversal_cost = random.uniform(0.05, 0.2)
                detection_mod = cls._terrain_detection_modifier(terrain)
            else:
                terrain = TerrainType.NORMAL
                search_time = 0.1  # Base search time
                traversal_cost = 0.1  # Base traversal cost
                detection_mod = 1.0

            cell = Cell(
                id=i,
                search_time=search_time,
                traversal_cost=traversal_cost,
                detection_modifier=detection_mod,
                terrain_type=terrain,
                prior_probability=1.0 / length  # Uniform prior
            )
            cells.append(cell)

        # Place the key
        if key_position is None:
            key_position = random.randint(0, length - 1)

        cells[key_position].has_key = True

        return cls(cells=cells, key_position=key_position)

    @classmethod
    def generate_with_hotspots(
        cls,
        length: int,
        hotspot_count: int = 3,
        hotspot_probability_boost: float = 3.0,
        seed: Optional[int] = None
    ) -> 'Path':
        """
        Generate a path where some cells have higher prior probability.

        Hotspots represent areas like rest stops, obstacles, or steep
        inclines where items are more likely to fall from pockets.

        Args:
            length: Number of cells
            hotspot_count: Number of high-probability zones
            hotspot_probability_boost: Multiplier for hotspot probability
            seed: Random seed

        Returns:
            A new Path with non-uniform probability distribution
        """
        if seed is not None:
            random.seed(seed)

        path = cls.generate(length, heterogeneous=True, seed=seed)

        # Select hotspot locations
        hotspot_indices = random.sample(range(length), min(hotspot_count, length))

        # Boost probability at hotspots
        for idx in hotspot_indices:
            path.cells[idx].prior_probability *= hotspot_probability_boost

        # Normalize probabilities
        total_prob = sum(c.prior_probability for c in path.cells)
        for cell in path.cells:
            cell.prior_probability /= total_prob

        # Place key according to probability distribution
        r = random.random()
        cumulative = 0.0
        for i, cell in enumerate(path.cells):
            cumulative += cell.prior_probability
            if r <= cumulative:
                # Clear old key position
                if path.key_position is not None:
                    path.cells[path.key_position].has_key = False
                path.key_position = i
                cell.has_key = True
                break

        return path

    @staticmethod
    def _terrain_search_time(terrain: TerrainType) -> float:
        """Get base search time for terrain type (in simulation time units)."""
        # Scaled down for faster simulation - 0.1 = base unit
        times = {
            TerrainType.NORMAL: 0.1,
            TerrainType.DENSE: 0.2,
            TerrainType.OPEN: 0.05,
            TerrainType.ROCKY: 0.15,
            TerrainType.WATER: 0.3,
        }
        return times.get(terrain, 0.1)

    @staticmethod
    def _terrain_detection_modifier(terrain: TerrainType) -> float:
        """Get detection modifier for terrain type."""
        modifiers = {
            TerrainType.NORMAL: 1.0,
            TerrainType.DENSE: 0.7,   # Harder to spot
            TerrainType.OPEN: 1.2,    # Easier to spot
            TerrainType.ROCKY: 0.8,
            TerrainType.WATER: 0.5,
        }
        return modifiers.get(terrain, 1.0)

    def get_cell_ids(self) -> List[int]:
        """Get list of all cell IDs."""
        return [c.id for c in self.cells]

    def distance(self, from_id: int, to_id: int) -> float:
        """
        Calculate traversal distance between two cells.

        For a linear path, this is the sum of traversal costs
        between the cells.
        """
        if from_id == to_id:
            return 0.0

        start, end = min(from_id, to_id), max(from_id, to_id)
        return sum(self.cells[i].traversal_cost for i in range(start + 1, end + 1))

    def get_probability_distribution(self) -> List[float]:
        """Get the prior probability distribution over cells."""
        return [c.prior_probability for c in self.cells]
