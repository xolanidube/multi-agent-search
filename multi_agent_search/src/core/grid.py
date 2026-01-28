"""
2D Grid Search Space for multi-agent search.

Extends the 1D path concept to a 2D grid where agents can move
in 4 or 8 directions. More realistic for:
- Drone swarm searches
- Warehouse robotics
- File system directory trees
- Network topology exploration
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Set, Dict
import random
import math


class GridCellStatus(Enum):
    """Status of a cell in the 2D grid."""
    UNEXPLORED = "unexplored"
    EXPLORING = "exploring"
    EXPLORED = "explored"
    OBSTACLE = "obstacle"
    TARGET_FOUND = "target_found"


class MovementType(Enum):
    """Type of movement allowed on the grid."""
    FOUR_WAY = 4      # Up, Down, Left, Right
    EIGHT_WAY = 8     # Including diagonals


@dataclass
class GridCell:
    """
    A cell in the 2D grid.

    Attributes:
        x, y: Grid coordinates
        search_cost: Time to search this cell
        traversal_cost: Time to move into this cell
        is_obstacle: Whether this cell is impassable
        has_target: Whether target is here (hidden from agents)
        prior_probability: Prior probability of target being here
        metadata: Optional metadata (e.g., filename, directory info)
    """
    x: int
    y: int
    search_cost: float = 0.1
    traversal_cost: float = 0.1
    is_obstacle: bool = False
    has_target: bool = False
    prior_probability: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def id(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, GridCell):
            return self.x == other.x and self.y == other.y
        return False


class Grid2D:
    """
    2D Grid search space.

    Represents a rectangular grid where agents can search for targets.
    Supports obstacles, varying terrain costs, and probability distributions.
    """

    def __init__(
        self,
        width: int,
        height: int,
        movement: MovementType = MovementType.FOUR_WAY
    ):
        """
        Initialize a 2D grid.

        Args:
            width: Grid width
            height: Grid height
            movement: Type of movement allowed
        """
        self.width = width
        self.height = height
        self.movement = movement

        # Create cells
        self.cells: Dict[Tuple[int, int], GridCell] = {}
        for y in range(height):
            for x in range(width):
                self.cells[(x, y)] = GridCell(
                    x=x, y=y,
                    prior_probability=1.0 / (width * height)
                )

        # Target position
        self.target_position: Optional[Tuple[int, int]] = None

        # Cell status tracking (for marking system)
        self.status: Dict[Tuple[int, int], GridCellStatus] = {
            pos: GridCellStatus.UNEXPLORED for pos in self.cells
        }

    def __getitem__(self, pos: Tuple[int, int]) -> GridCell:
        return self.cells[pos]

    def __len__(self) -> int:
        return len(self.cells)

    def is_valid(self, x: int, y: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_passable(self, x: int, y: int) -> bool:
        """Check if cell is passable (not obstacle)."""
        if not self.is_valid(x, y):
            return False
        return not self.cells[(x, y)].is_obstacle

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells based on movement type."""
        neighbors = []

        # 4-way movement
        directions_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # 8-way adds diagonals
        directions_8 = directions_4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        directions = directions_8 if self.movement == MovementType.EIGHT_WAY else directions_4

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny) and self.is_passable(nx, ny):
                neighbors.append((nx, ny))

        return neighbors

    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate distance between two positions."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        if self.movement == MovementType.EIGHT_WAY:
            # Chebyshev distance for 8-way
            return max(dx, dy)
        else:
            # Manhattan distance for 4-way
            return dx + dy

    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def mark(self, pos: Tuple[int, int], status: GridCellStatus) -> None:
        """Mark a cell with a status."""
        if pos in self.cells:
            self.status[pos] = status

    def get_status(self, pos: Tuple[int, int]) -> GridCellStatus:
        """Get status of a cell."""
        return self.status.get(pos, GridCellStatus.UNEXPLORED)

    def is_available(self, pos: Tuple[int, int]) -> bool:
        """Check if cell is available for exploration."""
        if pos not in self.cells:
            return False
        if self.cells[pos].is_obstacle:
            return False
        return self.status[pos] == GridCellStatus.UNEXPLORED

    def get_unexplored(self) -> List[Tuple[int, int]]:
        """Get list of unexplored cell positions."""
        return [pos for pos, status in self.status.items()
                if status == GridCellStatus.UNEXPLORED and not self.cells[pos].is_obstacle]

    def count_explored(self) -> int:
        """Count explored cells."""
        return sum(1 for s in self.status.values()
                   if s in (GridCellStatus.EXPLORED, GridCellStatus.TARGET_FOUND))

    def place_target(self, pos: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Place target at specified or random position."""
        if pos is None:
            # Random non-obstacle position
            valid = [p for p, c in self.cells.items() if not c.is_obstacle]
            pos = random.choice(valid)

        self.target_position = pos
        self.cells[pos].has_target = True
        return pos

    def add_obstacles(self, count: int = 0, density: float = 0.0) -> None:
        """Add random obstacles to the grid."""
        if density > 0:
            count = int(len(self.cells) * density)

        valid = [p for p, c in self.cells.items()
                 if not c.is_obstacle and not c.has_target]

        obstacles = random.sample(valid, min(count, len(valid)))
        for pos in obstacles:
            self.cells[pos].is_obstacle = True
            self.status[pos] = GridCellStatus.OBSTACLE

    def add_obstacle_region(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Add a rectangular obstacle region."""
        for y in range(min(y1, y2), max(y1, y2) + 1):
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if self.is_valid(x, y):
                    self.cells[(x, y)].is_obstacle = True
                    self.status[(x, y)] = GridCellStatus.OBSTACLE

    def set_hotzone(
        self,
        center: Tuple[int, int],
        radius: int,
        probability_boost: float = 3.0
    ) -> None:
        """Create a high-probability zone around a center point."""
        for pos, cell in self.cells.items():
            dist = self.euclidean_distance(pos, center)
            if dist <= radius:
                cell.prior_probability *= probability_boost

        # Normalize
        total = sum(c.prior_probability for c in self.cells.values() if not c.is_obstacle)
        for cell in self.cells.values():
            if not cell.is_obstacle:
                cell.prior_probability /= total

    @classmethod
    def generate_maze(cls, width: int, height: int, complexity: float = 0.3) -> 'Grid2D':
        """Generate a grid with maze-like obstacles."""
        grid = cls(width, height)

        # Add some random walls
        for _ in range(int(width * height * complexity)):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            # Random short wall
            length = random.randint(2, min(5, width // 3))
            horizontal = random.random() > 0.5

            for i in range(length):
                if horizontal:
                    nx = x + i
                    if grid.is_valid(nx, y):
                        grid.cells[(nx, y)].is_obstacle = True
                        grid.status[(nx, y)] = GridCellStatus.OBSTACLE
                else:
                    ny = y + i
                    if grid.is_valid(x, ny):
                        grid.cells[(x, ny)].is_obstacle = True
                        grid.status[(x, ny)] = GridCellStatus.OBSTACLE

        # Ensure start and corners are clear
        for corner in [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]:
            if corner in grid.cells:
                grid.cells[corner].is_obstacle = False
                grid.status[corner] = GridCellStatus.UNEXPLORED

        return grid

    @classmethod
    def generate_rooms(cls, width: int, height: int, room_count: int = 4) -> 'Grid2D':
        """Generate a grid with room-like structure (simulating building/warehouse)."""
        grid = cls(width, height)

        # Create walls around rooms
        room_width = width // int(math.sqrt(room_count))
        room_height = height // int(math.sqrt(room_count))

        for ry in range(0, height, room_height):
            for rx in range(0, width, room_width):
                # Add walls with doorways
                # Top wall
                for x in range(rx, min(rx + room_width, width)):
                    if x != rx + room_width // 2:  # Leave doorway
                        if grid.is_valid(x, ry):
                            grid.cells[(x, ry)].is_obstacle = True
                            grid.status[(x, ry)] = GridCellStatus.OBSTACLE

                # Left wall
                for y in range(ry, min(ry + room_height, height)):
                    if y != ry + room_height // 2:  # Leave doorway
                        if grid.is_valid(rx, y):
                            grid.cells[(rx, y)].is_obstacle = True
                            grid.status[(rx, y)] = GridCellStatus.OBSTACLE

        # Clear corners
        grid.cells[(0, 0)].is_obstacle = False
        grid.status[(0, 0)] = GridCellStatus.UNEXPLORED

        return grid


class GridAgent:
    """
    Agent that operates on a 2D grid.

    Similar to Searcher but adapted for 2D movement.
    """

    def __init__(
        self,
        agent_id: int,
        start_position: Tuple[int, int] = (0, 0),
        speed: float = 1.0,
        detection_probability: float = 1.0
    ):
        self.id = agent_id
        self.position = start_position
        self.speed = speed
        self.detection_probability = detection_probability

        # Tracking
        self.cells_explored = 0
        self.distance_traveled = 0.0
        self.path_history: List[Tuple[int, int]] = [start_position]

    def move_to(self, pos: Tuple[int, int], grid: Grid2D) -> float:
        """Move to a position and return time taken."""
        if pos not in grid.cells:
            return 0.0

        travel_time = grid.cells[pos].traversal_cost / self.speed
        self.distance_traveled += grid.euclidean_distance(self.position, pos)
        self.position = pos
        self.path_history.append(pos)
        return travel_time

    def explore(self, grid: Grid2D) -> Tuple[bool, float]:
        """
        Explore current cell.

        Returns:
            (found_target, time_taken)
        """
        cell = grid.cells[self.position]
        search_time = cell.search_cost / self.speed

        self.cells_explored += 1

        # Check for target
        found = cell.has_target and random.random() < self.detection_probability

        return found, search_time
