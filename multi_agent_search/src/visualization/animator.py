"""
Matplotlib Animation for visualizing the multi-agent search.

Shows:
- Path as a horizontal line with cells
- Searchers as colored markers
- Cell colors indicating search status
- Time countdown and metrics
"""

from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np

from ..core.cell import Path, CellStatus
from ..core.marking_system import MarkingSystem
from ..core.searcher import Searcher


# Color scheme
COLORS = {
    CellStatus.UNMARKED: '#E0E0E0',      # Light gray
    CellStatus.IN_PROGRESS: '#FFD54F',    # Yellow
    CellStatus.SEARCHED: '#81C784',       # Green
    CellStatus.FOUND: '#EF5350',          # Red
}

SEARCHER_COLORS = [
    '#2196F3',  # Blue
    '#9C27B0',  # Purple
    '#FF9800',  # Orange
    '#00BCD4',  # Cyan
    '#E91E63',  # Pink
    '#4CAF50',  # Green
    '#795548',  # Brown
    '#607D8B',  # Blue Gray
]


class SearchAnimator:
    """
    Animated visualization of the multi-agent search.

    Creates a Matplotlib figure showing:
    - Path cells as colored rectangles
    - Searchers as triangular markers
    - Time remaining countdown
    - Coverage and search statistics
    """

    def __init__(
        self,
        path: Path,
        searchers: List[Searcher],
        marks: MarkingSystem,
        figsize: Tuple[int, int] = (14, 8),
        cell_height: float = 0.6,
        interval: int = 100
    ):
        """
        Initialize the animator.

        Args:
            path: The search path
            searchers: List of searcher agents
            marks: The marking system
            figsize: Figure size (width, height)
            cell_height: Height of cell rectangles
            interval: Animation interval in milliseconds
        """
        self.path = path
        self.searchers = searchers
        self.marks = marks
        self.figsize = figsize
        self.cell_height = cell_height
        self.interval = interval

        # State tracking
        self.current_time = 0.0
        self.total_time = 60.0
        self.key_found = False
        self.animation_frames: List[Dict] = []

        # Matplotlib objects
        self.fig: Optional[plt.Figure] = None
        self.ax_path: Optional[plt.Axes] = None
        self.ax_stats: Optional[plt.Axes] = None
        self.cell_patches: List[mpatches.Rectangle] = []
        self.searcher_markers: List[plt.Line2D] = []

    def setup_figure(self) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        Set up the Matplotlib figure and axes.

        Returns:
            Tuple of (figure, (path_axes, stats_axes))
        """
        self.fig, (self.ax_path, self.ax_stats) = plt.subplots(
            2, 1,
            figsize=self.figsize,
            gridspec_kw={'height_ratios': [2, 1]}
        )

        self.fig.suptitle(
            'Multi-Agent Search Simulation',
            fontsize=14,
            fontweight='bold'
        )

        # Setup path visualization
        self._setup_path_axes()

        # Setup stats panel
        self._setup_stats_axes()

        plt.tight_layout()
        return self.fig, (self.ax_path, self.ax_stats)

    def _setup_path_axes(self) -> None:
        """Set up the path visualization axes."""
        self.ax_path.set_xlim(-1, len(self.path) + 1)
        self.ax_path.set_ylim(-1, 2)
        self.ax_path.set_aspect('equal')
        self.ax_path.set_title('Forest Path (searching for lost key)')
        self.ax_path.set_xlabel('Cell Position')
        self.ax_path.set_ylabel('')
        self.ax_path.set_yticks([])

        # Create cell rectangles
        self.cell_patches = []
        for i, cell in enumerate(self.path):
            rect = mpatches.Rectangle(
                (i, 0),
                0.9,
                self.cell_height,
                facecolor=COLORS[CellStatus.UNMARKED],
                edgecolor='black',
                linewidth=0.5
            )
            self.ax_path.add_patch(rect)
            self.cell_patches.append(rect)

        # Mark key position with star (hidden initially, shown when found)
        key_pos = self.path.key_position
        self.key_marker = self.ax_path.plot(
            key_pos + 0.45, self.cell_height / 2,
            '*', markersize=15, color='gold',
            markeredgecolor='black', visible=False
        )[0]

        # Create searcher markers
        self.searcher_markers = []
        for i, searcher in enumerate(self.searchers):
            color = SEARCHER_COLORS[i % len(SEARCHER_COLORS)]
            marker, = self.ax_path.plot(
                searcher.position + 0.45,
                self.cell_height + 0.3,
                'v',  # Triangle pointing down
                markersize=12,
                color=color,
                markeredgecolor='black',
                label=f'S{i}: {searcher.strategy_name}'
            )
            self.searcher_markers.append(marker)

        # Add legend
        self.ax_path.legend(
            loc='upper right',
            fontsize=8,
            framealpha=0.9
        )

        # Add cell status legend
        legend_patches = [
            mpatches.Patch(color=COLORS[CellStatus.UNMARKED], label='Unmarked'),
            mpatches.Patch(color=COLORS[CellStatus.IN_PROGRESS], label='In Progress'),
            mpatches.Patch(color=COLORS[CellStatus.SEARCHED], label='Searched'),
            mpatches.Patch(color=COLORS[CellStatus.FOUND], label='Found'),
        ]
        self.ax_path.legend(
            handles=legend_patches,
            loc='upper left',
            fontsize=8,
            framealpha=0.9
        )

    def _setup_stats_axes(self) -> None:
        """Set up the statistics panel."""
        self.ax_stats.set_xlim(0, 10)
        self.ax_stats.set_ylim(0, 10)
        self.ax_stats.axis('off')

        # Create text elements
        self.time_text = self.ax_stats.text(
            0.5, 8, 'Time: 60.0s remaining',
            fontsize=14, fontweight='bold'
        )

        self.coverage_text = self.ax_stats.text(
            0.5, 6, 'Coverage: 0 / 100 cells (0.0%)',
            fontsize=12
        )

        self.status_text = self.ax_stats.text(
            0.5, 4, 'Status: Searching...',
            fontsize=12
        )

        self.strategy_text = self.ax_stats.text(
            0.5, 2, 'Strategy: --',
            fontsize=10
        )

        # Progress bar background
        self.progress_bg = mpatches.Rectangle(
            (5, 7.5), 4, 0.6,
            facecolor='#E0E0E0',
            edgecolor='black'
        )
        self.ax_stats.add_patch(self.progress_bg)

        # Progress bar fill
        self.progress_fill = mpatches.Rectangle(
            (5, 7.5), 0, 0.6,
            facecolor='#4CAF50',
            edgecolor='none'
        )
        self.ax_stats.add_patch(self.progress_fill)

    def record_frame(
        self,
        time: float,
        total_time: float,
        key_found: bool = False
    ) -> None:
        """
        Record current state as an animation frame.

        Args:
            time: Current elapsed time
            total_time: Total time budget
            key_found: Whether key has been found
        """
        frame = {
            'time': time,
            'total_time': total_time,
            'key_found': key_found,
            'cell_statuses': [self.marks.get_status(i) for i in range(len(self.path))],
            'searcher_positions': [s.position for s in self.searchers],
            'cells_searched': self.marks.count_searched(),
        }
        self.animation_frames.append(frame)

    def update_display(self, frame_data: Dict) -> None:
        """
        Update the display with frame data.

        Args:
            frame_data: Dictionary with frame state
        """
        # Update cell colors
        for i, status in enumerate(frame_data['cell_statuses']):
            self.cell_patches[i].set_facecolor(COLORS[status])

        # Update searcher positions
        for i, pos in enumerate(frame_data['searcher_positions']):
            self.searcher_markers[i].set_xdata([pos + 0.45])

        # Show key if found
        if frame_data['key_found']:
            self.key_marker.set_visible(True)

        # Update text
        remaining = frame_data['total_time'] - frame_data['time']
        self.time_text.set_text(f'Time: {remaining:.1f}s remaining')

        cells_searched = frame_data['cells_searched']
        total_cells = len(self.path)
        pct = cells_searched / total_cells * 100
        self.coverage_text.set_text(
            f'Coverage: {cells_searched} / {total_cells} cells ({pct:.1f}%)'
        )

        if frame_data['key_found']:
            self.status_text.set_text('Status: KEY FOUND!')
            self.status_text.set_color('green')
        elif remaining <= 0:
            self.status_text.set_text('Status: TIME EXPIRED')
            self.status_text.set_color('red')
        else:
            self.status_text.set_text('Status: Searching...')

        # Update progress bar
        progress = cells_searched / total_cells
        self.progress_fill.set_width(4 * progress)

    def animate(self, save_path: Optional[str] = None) -> Optional[FuncAnimation]:
        """
        Create and run the animation.

        Args:
            save_path: If provided, save animation to this path

        Returns:
            FuncAnimation object if not saving
        """
        if not self.animation_frames:
            print("No frames recorded. Run simulation first.")
            return None

        self.setup_figure()

        def init():
            return []

        def update(frame_idx):
            if frame_idx < len(self.animation_frames):
                self.update_display(self.animation_frames[frame_idx])
            return []

        anim = FuncAnimation(
            self.fig,
            update,
            init_func=init,
            frames=len(self.animation_frames),
            interval=self.interval,
            blit=False
        )

        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()

        return anim

    def show_final_state(self, results: Dict) -> None:
        """
        Show final state as a static image.

        Args:
            results: Simulation results dictionary
        """
        self.setup_figure()

        # Update all cell colors based on final marking state
        for i in range(len(self.path)):
            status = self.marks.get_status(i)
            self.cell_patches[i].set_facecolor(COLORS[status])

        # Update searcher positions
        for i, searcher in enumerate(self.searchers):
            self.searcher_markers[i].set_xdata([searcher.position + 0.45])

        # Show key
        self.key_marker.set_visible(True)

        # Update stats
        self.time_text.set_text(
            f"Time: {results.get('time_remaining', 0):.1f}s remaining"
        )

        cells = results.get('cells_searched', 0)
        total = results.get('total_cells', len(self.path))
        pct = results.get('coverage_rate', 0) * 100
        self.coverage_text.set_text(
            f'Coverage: {cells} / {total} cells ({pct:.1f}%)'
        )

        if results.get('success', False):
            self.status_text.set_text('Status: KEY FOUND!')
            self.status_text.set_color('green')
        else:
            self.status_text.set_text('Status: FAILED')
            self.status_text.set_color('red')

        strategy = results.get('selected_strategy', 'unknown')
        self.strategy_text.set_text(f'Strategy: {strategy}')

        # Progress bar
        self.progress_fill.set_width(4 * results.get('coverage_rate', 0))

        plt.show()


def create_simple_visualization(
    path: Path,
    marks: MarkingSystem,
    searcher_positions: List[int],
    show: bool = True
) -> plt.Figure:
    """
    Create a simple static visualization of the search state.

    Args:
        path: The search path
        marks: Current marking system state
        searcher_positions: List of searcher positions
        show: Whether to call plt.show()

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 3))

    # Draw cells
    for i in range(len(path)):
        status = marks.get_status(i)
        color = COLORS[status]
        rect = mpatches.Rectangle(
            (i, 0), 0.9, 0.6,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(rect)

    # Draw key position
    ax.plot(
        path.key_position + 0.45, 0.3,
        '*', markersize=15, color='gold',
        markeredgecolor='black'
    )

    # Draw searchers
    for i, pos in enumerate(searcher_positions):
        color = SEARCHER_COLORS[i % len(SEARCHER_COLORS)]
        ax.plot(
            pos + 0.45, 0.9,
            'v', markersize=10, color=color,
            markeredgecolor='black'
        )

    ax.set_xlim(-0.5, len(path) + 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Search State')
    ax.set_xlabel('Cell Position')
    ax.set_yticks([])

    plt.tight_layout()

    if show:
        plt.show()

    return fig
