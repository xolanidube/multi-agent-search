"""
2D Grid Visualization and Animation.

Creates animated visualizations of multi-agent search on 2D grids.
Shows agent movement, exploration progress, and coordination patterns.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import List, Dict, Optional, Tuple

from ..core.grid import Grid2D, GridAgent, GridCellStatus


# Color schemes
CELL_COLORS = {
    GridCellStatus.UNEXPLORED: '#E8E8E8',    # Light gray
    GridCellStatus.EXPLORING: '#FFE082',      # Amber
    GridCellStatus.EXPLORED: '#A5D6A7',       # Light green
    GridCellStatus.OBSTACLE: '#424242',       # Dark gray
    GridCellStatus.TARGET_FOUND: '#EF5350',   # Red
}

AGENT_COLORS = [
    '#2196F3',  # Blue
    '#9C27B0',  # Purple
    '#FF5722',  # Deep Orange
    '#00BCD4',  # Cyan
    '#E91E63',  # Pink
    '#4CAF50',  # Green
    '#FF9800',  # Orange
    '#673AB7',  # Deep Purple
    '#009688',  # Teal
    '#F44336',  # Red
]


def create_grid_image(grid: Grid2D) -> np.ndarray:
    """Create RGB image array from grid state."""
    image = np.zeros((grid.height, grid.width, 3))

    for (x, y), status in grid.status.items():
        color = CELL_COLORS.get(status, CELL_COLORS[GridCellStatus.UNEXPLORED])
        # Convert hex to RGB
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255
        image[y, x] = [r, g, b]

    return image


class Grid2DAnimator:
    """
    Animated visualization of 2D grid search.

    Shows:
    - Grid cells colored by exploration status
    - Agents as colored markers
    - Target location (when found)
    - Exploration progress and statistics
    - Agent trails
    """

    def __init__(
        self,
        grid: Grid2D,
        agents: List[GridAgent],
        figsize: Tuple[int, int] = (12, 10),
        show_trails: bool = True,
        show_heatmap: bool = False
    ):
        self.grid = grid
        self.agents = agents
        self.figsize = figsize
        self.show_trails = show_trails
        self.show_heatmap = show_heatmap

        self.frames: List[Dict] = []

    def record_frame(
        self,
        time: float,
        target_found: bool = False,
        extra_info: Optional[Dict] = None
    ) -> None:
        """Record current state as animation frame."""
        frame = {
            'time': time,
            'target_found': target_found,
            'status': {pos: status for pos, status in self.grid.status.items()},
            'agent_positions': [a.position for a in self.agents],
            'agent_paths': [list(a.path_history) for a in self.agents],
            'explored_count': self.grid.count_explored(),
            'extra': extra_info or {}
        }
        self.frames.append(frame)

    def create_animation(
        self,
        output_path: str,
        fps: int = 10,
        title: str = "Multi-Agent Grid Search"
    ) -> None:
        """Create and save animation."""
        if not self.frames:
            print("No frames recorded!")
            return

        fig, axes = plt.subplots(1, 2, figsize=self.figsize,
                                  gridspec_kw={'width_ratios': [3, 1]})
        ax_grid = axes[0]
        ax_stats = axes[1]

        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Setup grid axes
        ax_grid.set_xlim(-0.5, self.grid.width - 0.5)
        ax_grid.set_ylim(-0.5, self.grid.height - 0.5)
        ax_grid.set_aspect('equal')
        ax_grid.set_xlabel('X')
        ax_grid.set_ylabel('Y')
        ax_grid.invert_yaxis()  # (0,0) at top-left

        # Create initial grid image
        initial_image = create_grid_image(self.grid)
        im = ax_grid.imshow(initial_image, extent=[-0.5, self.grid.width - 0.5,
                                                     self.grid.height - 0.5, -0.5])

        # Agent markers
        agent_markers = []
        agent_trails = []
        for i, agent in enumerate(self.agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            marker, = ax_grid.plot([], [], 'o', markersize=15, color=color,
                                   markeredgecolor='white', markeredgewidth=2,
                                   label=f'Agent {i+1}')
            agent_markers.append(marker)

            if self.show_trails:
                trail, = ax_grid.plot([], [], '-', color=color, alpha=0.5, linewidth=2)
                agent_trails.append(trail)

        # Target marker (hidden initially)
        target_marker, = ax_grid.plot([], [], '*', markersize=20, color='gold',
                                       markeredgecolor='black', visible=False)

        # Legend
        legend_patches = [
            mpatches.Patch(color=CELL_COLORS[GridCellStatus.UNEXPLORED], label='Unexplored'),
            mpatches.Patch(color=CELL_COLORS[GridCellStatus.EXPLORED], label='Explored'),
            mpatches.Patch(color=CELL_COLORS[GridCellStatus.OBSTACLE], label='Obstacle'),
            mpatches.Patch(color=CELL_COLORS[GridCellStatus.TARGET_FOUND], label='Target'),
        ]
        ax_grid.legend(handles=legend_patches, loc='upper right', fontsize=8)

        # Stats panel
        ax_stats.axis('off')
        ax_stats.set_xlim(0, 10)
        ax_stats.set_ylim(0, 10)

        time_text = ax_stats.text(1, 9, 'Time: 0.0s', fontsize=12, fontweight='bold')
        explored_text = ax_stats.text(1, 8, 'Explored: 0', fontsize=11)
        coverage_text = ax_stats.text(1, 7, 'Coverage: 0%', fontsize=11)
        status_text = ax_stats.text(1, 6, 'Status: Searching...', fontsize=11)

        # Progress bar
        total_cells = sum(1 for c in self.grid.cells.values() if not c.is_obstacle)
        progress_bg = mpatches.Rectangle((1, 4.5), 8, 0.8, facecolor='#E0E0E0',
                                          edgecolor='black')
        progress_fill = mpatches.Rectangle((1, 4.5), 0, 0.8, facecolor='#4CAF50')
        ax_stats.add_patch(progress_bg)
        ax_stats.add_patch(progress_fill)
        ax_stats.text(1, 5.5, 'Progress:', fontsize=10)

        # Agent info
        agent_texts = []
        for i in range(len(self.agents)):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            txt = ax_stats.text(1, 3 - i * 0.6, f'Agent {i+1}: (0, 0)',
                               fontsize=9, color=color)
            agent_texts.append(txt)

        plt.tight_layout()

        def update(frame_idx):
            if frame_idx >= len(self.frames):
                return []

            frame = self.frames[frame_idx]

            # Update grid image
            image = np.zeros((self.grid.height, self.grid.width, 3))
            for (x, y), status in frame['status'].items():
                color = CELL_COLORS.get(status, CELL_COLORS[GridCellStatus.UNEXPLORED])
                r = int(color[1:3], 16) / 255
                g = int(color[3:5], 16) / 255
                b = int(color[5:7], 16) / 255
                image[y, x] = [r, g, b]
            im.set_array(image)

            # Update agent positions
            for i, pos in enumerate(frame['agent_positions']):
                agent_markers[i].set_data([pos[0]], [pos[1]])

                if self.show_trails and i < len(agent_trails):
                    path = frame['agent_paths'][i]
                    if len(path) > 1:
                        xs = [p[0] for p in path]
                        ys = [p[1] for p in path]
                        agent_trails[i].set_data(xs, ys)

            # Show target if found
            if frame['target_found'] and self.grid.target_position:
                target_marker.set_data([self.grid.target_position[0]],
                                       [self.grid.target_position[1]])
                target_marker.set_visible(True)

            # Update stats
            time_text.set_text(f"Time: {frame['time']:.1f}s")
            explored = frame['explored_count']
            explored_text.set_text(f'Explored: {explored}')

            coverage = explored / total_cells * 100 if total_cells > 0 else 0
            coverage_text.set_text(f'Coverage: {coverage:.1f}%')

            if frame['target_found']:
                status_text.set_text('Status: TARGET FOUND!')
                status_text.set_color('green')
            else:
                status_text.set_text('Status: Searching...')
                status_text.set_color('black')

            # Progress bar
            progress_fill.set_width(8 * coverage / 100)

            # Agent positions
            for i, pos in enumerate(frame['agent_positions']):
                if i < len(agent_texts):
                    agent_texts[i].set_text(f'Agent {i+1}: ({pos[0]}, {pos[1]})')

            return []

        anim = FuncAnimation(fig, update, frames=len(self.frames),
                            interval=100, blit=False)

        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        plt.close(fig)
        print(f"Saved: {output_path}")

    def show_final_state(self, save_path: Optional[str] = None) -> None:
        """Display or save final state as static image."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw grid
        image = create_grid_image(self.grid)
        ax.imshow(image, extent=[-0.5, self.grid.width - 0.5,
                                  self.grid.height - 0.5, -0.5])

        # Draw agent trails
        for i, agent in enumerate(self.agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            if len(agent.path_history) > 1:
                xs = [p[0] for p in agent.path_history]
                ys = [p[1] for p in agent.path_history]
                ax.plot(xs, ys, '-', color=color, alpha=0.5, linewidth=2)

            # Current position
            ax.plot(agent.position[0], agent.position[1], 'o', markersize=12,
                   color=color, markeredgecolor='white', markeredgewidth=2,
                   label=f'Agent {i+1}')

        # Target
        if self.grid.target_position:
            ax.plot(self.grid.target_position[0], self.grid.target_position[1],
                   '*', markersize=20, color='gold', markeredgecolor='black')

        ax.set_xlim(-0.5, self.grid.width - 0.5)
        ax.set_ylim(-0.5, self.grid.height - 0.5)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title('Final Search State')
        ax.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {save_path}")
        else:
            plt.show()


def run_grid_simulation(
    grid: Grid2D,
    agents: List[GridAgent],
    strategies: List,
    time_budget: float = 30.0,
    time_step: float = 0.1
) -> Tuple[List[Dict], bool, float]:
    """
    Run simulation on 2D grid and return frames.

    Returns:
        (frames, target_found, elapsed_time)
    """
    frames = []
    current_time = 0.0
    target_found = False

    # Record initial state
    frames.append({
        'time': current_time,
        'target_found': False,
        'status': {pos: status for pos, status in grid.status.items()},
        'agent_positions': [a.position for a in agents],
        'agent_paths': [list(a.path_history) for a in agents],
        'explored_count': 0,
    })

    while current_time < time_budget and not target_found:
        moved_any = False

        for i, (agent, strategy) in enumerate(zip(agents, strategies)):
            if target_found:
                break

            # Get next cell from strategy
            next_cell = strategy.get_next_cell(agent, grid, agents)

            if next_cell and grid.is_available(next_cell):
                moved_any = True

                # Mark as exploring
                grid.mark(next_cell, GridCellStatus.EXPLORING)

                # Move agent
                travel_time = agent.move_to(next_cell, grid)
                current_time += travel_time

                # Explore cell
                found, search_time = agent.explore(grid)
                current_time += search_time

                if found:
                    grid.mark(next_cell, GridCellStatus.TARGET_FOUND)
                    target_found = True
                else:
                    grid.mark(next_cell, GridCellStatus.EXPLORED)

                # Record frame
                frames.append({
                    'time': current_time,
                    'target_found': target_found,
                    'status': {pos: status for pos, status in grid.status.items()},
                    'agent_positions': [a.position for a in agents],
                    'agent_paths': [list(a.path_history) for a in agents],
                    'explored_count': grid.count_explored(),
                })

        if not moved_any:
            break

    return frames, target_found, current_time
