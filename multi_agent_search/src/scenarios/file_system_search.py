"""
Realistic File System Search Scenario.

Simulates searching for a file among millions of files in a directory
structure. Models:
- Directory tree as a graph (can be viewed as 2D grid for visualization)
- Variable file counts per directory
- I/O latency for listing and reading
- Caching effects
- Parallel worker agents
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import time


class NodeType(Enum):
    """Type of filesystem node."""
    FILE = "file"
    DIRECTORY = "directory"


@dataclass
class FileNode:
    """
    A node in the simulated file system.

    Can be a file or directory. Directories contain child nodes.
    """
    name: str
    node_type: NodeType
    parent_path: str = ""
    size_bytes: int = 0
    children: List['FileNode'] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    # Simulation properties
    access_time_ms: float = 1.0  # Time to access/list this node
    is_target: bool = False
    is_explored: bool = False

    @property
    def path(self) -> str:
        if self.parent_path:
            return f"{self.parent_path}/{self.name}"
        return self.name

    @property
    def depth(self) -> int:
        return self.path.count('/')


class FileSystemSimulator:
    """
    Simulates a large file system for search experiments.

    Models realistic characteristics:
    - Hierarchical directory structure
    - Variable directory sizes
    - File naming patterns
    - I/O latency
    - Parallel access
    """

    def __init__(
        self,
        total_files: int = 1_000_000,
        max_depth: int = 10,
        avg_files_per_dir: int = 100,
        target_filename: str = "needle.txt"
    ):
        """
        Initialize file system simulator.

        Args:
            total_files: Total number of files to create
            max_depth: Maximum directory depth
            avg_files_per_dir: Average files per directory
            target_filename: Name of file to search for
        """
        self.total_files = total_files
        self.max_depth = max_depth
        self.avg_files_per_dir = avg_files_per_dir
        self.target_filename = target_filename

        # Generated structure
        self.root: Optional[FileNode] = None
        self.all_nodes: List[FileNode] = []
        self.directories: List[FileNode] = []
        self.target_node: Optional[FileNode] = None

        # Search state
        self.explored_nodes: Set[str] = set()
        self.access_count = 0
        self.total_access_time_ms = 0.0

    def generate(self, seed: Optional[int] = None) -> None:
        """Generate the file system structure."""
        if seed:
            random.seed(seed)

        print(f"Generating file system with ~{self.total_files:,} files...")

        # Create root
        self.root = FileNode(
            name="root",
            node_type=NodeType.DIRECTORY,
            access_time_ms=0.5
        )
        self.all_nodes = [self.root]
        self.directories = [self.root]

        # Generate structure
        files_created = 0
        target_placed = False

        while files_created < self.total_files:
            # Pick a random directory to add content to
            parent = random.choice(self.directories)

            if parent.depth >= self.max_depth:
                # Too deep, add only files
                num_files = random.randint(1, self.avg_files_per_dir * 2)
            else:
                # Add mix of files and directories
                num_files = random.randint(1, self.avg_files_per_dir * 2)
                num_dirs = random.randint(0, 5)

                # Create subdirectories
                for i in range(num_dirs):
                    if files_created >= self.total_files:
                        break
                    dir_node = FileNode(
                        name=f"dir_{len(self.directories)}_{i}",
                        node_type=NodeType.DIRECTORY,
                        parent_path=parent.path,
                        access_time_ms=random.uniform(0.5, 2.0)  # Directory listing time
                    )
                    parent.children.append(dir_node)
                    self.directories.append(dir_node)
                    self.all_nodes.append(dir_node)

            # Create files
            for i in range(num_files):
                if files_created >= self.total_files:
                    break

                # Occasional target file placement
                if not target_placed and random.random() < 1.0 / max(1, self.total_files - files_created):
                    filename = self.target_filename
                    is_target = True
                    target_placed = True
                else:
                    filename = self._random_filename()
                    is_target = False

                file_node = FileNode(
                    name=filename,
                    node_type=NodeType.FILE,
                    parent_path=parent.path,
                    size_bytes=random.randint(100, 10_000_000),
                    access_time_ms=random.uniform(0.1, 0.5),  # File stat time
                    is_target=is_target
                )
                parent.children.append(file_node)
                self.all_nodes.append(file_node)
                files_created += 1

                if is_target:
                    self.target_node = file_node

        # Ensure target exists
        if not target_placed and self.directories:
            parent = random.choice(self.directories)
            self.target_node = FileNode(
                name=self.target_filename,
                node_type=NodeType.FILE,
                parent_path=parent.path,
                size_bytes=1024,
                access_time_ms=0.2,
                is_target=True
            )
            parent.children.append(self.target_node)
            self.all_nodes.append(self.target_node)

        print(f"Created {len(self.all_nodes):,} nodes ({len(self.directories):,} directories)")
        print(f"Target file at: {self.target_node.path if self.target_node else 'NOT PLACED'}")

    def _random_filename(self) -> str:
        """Generate a random filename."""
        extensions = ['.txt', '.log', '.dat', '.json', '.xml', '.csv', '.py', '.js', '.md']
        prefixes = ['data', 'file', 'doc', 'report', 'log', 'config', 'temp', 'backup']
        return f"{random.choice(prefixes)}_{random.randint(1, 999999)}{random.choice(extensions)}"

    def list_directory(self, node: FileNode) -> Tuple[List[FileNode], float]:
        """
        List contents of a directory.

        Returns:
            (children, access_time_ms)
        """
        if node.node_type != NodeType.DIRECTORY:
            return [], 0.0

        self.access_count += 1
        access_time = node.access_time_ms * (1 + len(node.children) * 0.001)  # Scale with size
        self.total_access_time_ms += access_time

        return node.children, access_time

    def check_file(self, node: FileNode) -> Tuple[bool, float]:
        """
        Check if a file is the target.

        Returns:
            (is_target, access_time_ms)
        """
        if node.node_type != NodeType.FILE:
            return False, 0.0

        self.access_count += 1
        self.total_access_time_ms += node.access_time_ms

        node.is_explored = True
        self.explored_nodes.add(node.path)

        return node.is_target, node.access_time_ms

    def reset_search_state(self) -> None:
        """Reset search state for new search."""
        self.explored_nodes.clear()
        self.access_count = 0
        self.total_access_time_ms = 0.0
        for node in self.all_nodes:
            node.is_explored = False


class FileSearchAgent:
    """
    Agent that searches the file system.

    Different strategies can be implemented:
    - BFS (breadth-first)
    - DFS (depth-first)
    - Probabilistic (based on file patterns)
    - Parallel (multiple workers)
    """

    def __init__(self, agent_id: int, strategy: str = "bfs"):
        self.id = agent_id
        self.strategy = strategy

        # State
        self.files_checked = 0
        self.dirs_listed = 0
        self.time_spent_ms = 0.0
        self.found_target = False

    def search(
        self,
        fs: FileSystemSimulator,
        time_budget_ms: float = 60000,  # 60 seconds default
        target_name: Optional[str] = None
    ) -> Tuple[bool, float, str]:
        """
        Search for target file.

        Returns:
            (found, time_spent_ms, path_if_found)
        """
        target_name = target_name or fs.target_filename

        if self.strategy == "bfs":
            return self._bfs_search(fs, time_budget_ms, target_name)
        elif self.strategy == "dfs":
            return self._dfs_search(fs, time_budget_ms, target_name)
        elif self.strategy == "probabilistic":
            return self._probabilistic_search(fs, time_budget_ms, target_name)
        else:
            return self._bfs_search(fs, time_budget_ms, target_name)

    def _bfs_search(
        self,
        fs: FileSystemSimulator,
        time_budget_ms: float,
        target_name: str
    ) -> Tuple[bool, float, str]:
        """Breadth-first search."""
        queue = [fs.root]

        while queue and self.time_spent_ms < time_budget_ms:
            node = queue.pop(0)

            if node.node_type == NodeType.DIRECTORY:
                children, access_time = fs.list_directory(node)
                self.time_spent_ms += access_time
                self.dirs_listed += 1

                for child in children:
                    if child.node_type == NodeType.FILE:
                        if child.name == target_name:
                            is_target, check_time = fs.check_file(child)
                            self.time_spent_ms += check_time
                            self.files_checked += 1
                            if is_target:
                                self.found_target = True
                                return True, self.time_spent_ms, child.path
                    else:
                        queue.append(child)

        return False, self.time_spent_ms, ""

    def _dfs_search(
        self,
        fs: FileSystemSimulator,
        time_budget_ms: float,
        target_name: str
    ) -> Tuple[bool, float, str]:
        """Depth-first search."""
        stack = [fs.root]

        while stack and self.time_spent_ms < time_budget_ms:
            node = stack.pop()

            if node.node_type == NodeType.DIRECTORY:
                children, access_time = fs.list_directory(node)
                self.time_spent_ms += access_time
                self.dirs_listed += 1

                for child in children:
                    if child.node_type == NodeType.FILE:
                        if child.name == target_name:
                            is_target, check_time = fs.check_file(child)
                            self.time_spent_ms += check_time
                            self.files_checked += 1
                            if is_target:
                                self.found_target = True
                                return True, self.time_spent_ms, child.path
                    else:
                        stack.append(child)

        return False, self.time_spent_ms, ""

    def _probabilistic_search(
        self,
        fs: FileSystemSimulator,
        time_budget_ms: float,
        target_name: str
    ) -> Tuple[bool, float, str]:
        """
        Probabilistic search - prioritize directories likely to contain target.

        Heuristics:
        - Prefer directories with names matching patterns
        - Prefer shallower directories first
        - Random exploration to avoid getting stuck
        """
        # Priority queue: (priority, node)
        import heapq
        queue = [(0, 0, fs.root)]  # (priority, counter, node)
        counter = 1

        while queue and self.time_spent_ms < time_budget_ms:
            _, _, node = heapq.heappop(queue)

            if node.node_type == NodeType.DIRECTORY:
                children, access_time = fs.list_directory(node)
                self.time_spent_ms += access_time
                self.dirs_listed += 1

                for child in children:
                    if child.node_type == NodeType.FILE:
                        if child.name == target_name:
                            is_target, check_time = fs.check_file(child)
                            self.time_spent_ms += check_time
                            self.files_checked += 1
                            if is_target:
                                self.found_target = True
                                return True, self.time_spent_ms, child.path
                    else:
                        # Calculate priority (lower = better)
                        priority = child.depth  # Prefer shallower
                        # Bonus for certain directory names
                        if any(kw in child.name.lower() for kw in ['data', 'doc', 'file']):
                            priority -= 1

                        heapq.heappush(queue, (priority, counter, child))
                        counter += 1

        return False, self.time_spent_ms, ""


def run_file_search_experiment(
    total_files: int = 100_000,
    strategy: str = "bfs",
    time_budget_ms: float = 30000,
    seed: int = 42
) -> Dict:
    """
    Run a file search experiment.

    Returns dict with metrics.
    """
    # Create file system
    fs = FileSystemSimulator(
        total_files=total_files,
        max_depth=8,
        avg_files_per_dir=50
    )
    fs.generate(seed=seed)

    # Create agent
    agent = FileSearchAgent(0, strategy=strategy)

    # Run search
    start_time = time.time()
    found, search_time_ms, path = agent.search(fs, time_budget_ms)
    wall_time = (time.time() - start_time) * 1000

    return {
        'found': found,
        'search_time_ms': search_time_ms,
        'wall_time_ms': wall_time,
        'files_checked': agent.files_checked,
        'dirs_listed': agent.dirs_listed,
        'total_files': total_files,
        'total_dirs': len(fs.directories),
        'target_path': path if found else fs.target_node.path,
        'target_depth': fs.target_node.depth if fs.target_node else -1,
        'strategy': strategy,
        'fs_access_count': fs.access_count,
    }


def compare_file_search_strategies(
    total_files: int = 100_000,
    time_budget_ms: float = 30000,
    runs: int = 10
) -> Dict[str, List[Dict]]:
    """Compare different search strategies."""
    strategies = ['bfs', 'dfs', 'probabilistic']
    results = {s: [] for s in strategies}

    print(f"\nComparing strategies on {total_files:,} files ({runs} runs each)...")

    for strategy in strategies:
        print(f"\n{strategy.upper()}:")
        for run in range(runs):
            result = run_file_search_experiment(
                total_files=total_files,
                strategy=strategy,
                time_budget_ms=time_budget_ms,
                seed=run * 100
            )
            results[strategy].append(result)

            status = "FOUND" if result['found'] else "NOT FOUND"
            print(f"  Run {run+1}: {status} in {result['search_time_ms']:.1f}ms "
                  f"({result['dirs_listed']} dirs, {result['files_checked']} files)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    for strategy in strategies:
        runs_data = results[strategy]
        found_count = sum(1 for r in runs_data if r['found'])
        avg_time = sum(r['search_time_ms'] for r in runs_data if r['found']) / max(1, found_count)
        avg_dirs = sum(r['dirs_listed'] for r in runs_data) / len(runs_data)

        print(f"  {strategy.upper():15} - Success: {found_count}/{runs} "
              f"- Avg time: {avg_time:.1f}ms - Avg dirs: {avg_dirs:.0f}")

    return results
