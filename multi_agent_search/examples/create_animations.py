"""
Script to create animated visualizations of the multi-agent search.

Generates GIF animations showing:
- Search progression with different configurations
- Strategy comparison
- Scaling demonstration
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from src.core.cell import Path, CellStatus
from src.core.marking_system import MarkingSystem
from src.core.searcher import Searcher
from src.strategies.greedy_nearest import GreedyNearestStrategy
from src.strategies.territorial import TerritorialStrategy
from src.strategies.probabilistic import ProbabilisticStrategy
from src.strategies.random_walk import RandomWalkStrategy


# Color scheme
COLORS = {
    CellStatus.UNMARKED: '#E0E0E0',
    CellStatus.IN_PROGRESS: '#FFD54F',
    CellStatus.SEARCHED: '#81C784',
    CellStatus.FOUND: '#EF5350',
}

SEARCHER_COLORS = ['#2196F3', '#9C27B0', '#FF9800', '#00BCD4', '#E91E63']


def run_simulation_with_frames(
    path: Path,
    searchers: list,
    marks: MarkingSystem,
    time_budget: float = 20.0,
    time_step: float = 0.1
) -> list:
    """
    Run simulation and capture frames for animation.

    Returns list of frame dictionaries.
    """
    frames = []
    current_time = 0.0
    key_found = False

    # Record initial state
    frames.append({
        'time': current_time,
        'cell_statuses': [marks.get_status(i) for i in range(len(path))],
        'searcher_positions': [s.position for s in searchers],
        'cells_searched': 0,
        'key_found': False,
    })

    while current_time < time_budget and not key_found:
        # Each searcher takes an action
        for searcher in searchers:
            if key_found:
                break

            # Get next cell from strategy
            next_cell = searcher.strategy.get_next_cell(searcher, path, marks)

            if next_cell is not None and marks.is_available(next_cell):
                # Mark as in progress
                marks.mark(next_cell, CellStatus.IN_PROGRESS)

                # Record frame (moving)
                frames.append({
                    'time': current_time,
                    'cell_statuses': [marks.get_status(i) for i in range(len(path))],
                    'searcher_positions': [s.position for s in searchers],
                    'cells_searched': marks.count_searched(),
                    'key_found': False,
                })

                # Move searcher
                cell = path[next_cell]
                current_time += cell.traversal_cost
                searcher.position = next_cell

                # Search cell
                current_time += cell.search_time

                # Check if key found
                if next_cell == path.key_position:
                    marks.mark(next_cell, CellStatus.FOUND)
                    key_found = True
                else:
                    marks.mark(next_cell, CellStatus.SEARCHED)

                # Record frame (after search)
                frames.append({
                    'time': current_time,
                    'cell_statuses': [marks.get_status(i) for i in range(len(path))],
                    'searcher_positions': [s.position for s in searchers],
                    'cells_searched': marks.count_searched(),
                    'key_found': key_found,
                })

        # Small time increment to prevent infinite loop
        if not any(marks.is_available(i) for i in range(len(path))):
            break

    return frames


def create_animation(
    frames: list,
    path: Path,
    num_searchers: int,
    title: str,
    output_path: str,
    fps: int = 5
):
    """Create and save animation from frames."""

    fig, (ax_path, ax_stats) = plt.subplots(
        2, 1,
        figsize=(14, 6),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Setup path axes
    ax_path.set_xlim(-1, len(path) + 1)
    ax_path.set_ylim(-0.5, 2)
    ax_path.set_aspect('equal')
    ax_path.set_xlabel('Cell Position')
    ax_path.set_yticks([])

    # Create cell patches
    cell_patches = []
    for i in range(len(path)):
        rect = mpatches.Rectangle(
            (i, 0), 0.9, 0.6,
            facecolor=COLORS[CellStatus.UNMARKED],
            edgecolor='black',
            linewidth=0.5
        )
        ax_path.add_patch(rect)
        cell_patches.append(rect)

    # Key marker (hidden initially)
    key_marker, = ax_path.plot(
        path.key_position + 0.45, 0.3,
        '*', markersize=20, color='gold',
        markeredgecolor='black', visible=False
    )

    # Searcher markers
    searcher_markers = []
    for i in range(num_searchers):
        color = SEARCHER_COLORS[i % len(SEARCHER_COLORS)]
        marker, = ax_path.plot(
            0.45, 0.9 + i * 0.2,
            'v', markersize=12, color=color,
            markeredgecolor='black',
            label=f'Searcher {i+1}'
        )
        searcher_markers.append(marker)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS[CellStatus.UNMARKED], label='Unmarked'),
        mpatches.Patch(color=COLORS[CellStatus.IN_PROGRESS], label='In Progress'),
        mpatches.Patch(color=COLORS[CellStatus.SEARCHED], label='Searched'),
        mpatches.Patch(color=COLORS[CellStatus.FOUND], label='Key Found!'),
    ]
    ax_path.legend(handles=legend_patches, loc='upper right', fontsize=8)

    # Stats panel
    ax_stats.set_xlim(0, 10)
    ax_stats.set_ylim(0, 6)
    ax_stats.axis('off')

    time_text = ax_stats.text(0.5, 4.5, 'Time: 0.0s', fontsize=14, fontweight='bold')
    coverage_text = ax_stats.text(0.5, 3, f'Coverage: 0 / {len(path)} cells', fontsize=12)
    status_text = ax_stats.text(0.5, 1.5, 'Status: Searching...', fontsize=12)

    # Progress bar
    progress_bg = mpatches.Rectangle((5, 4), 4, 0.6, facecolor='#E0E0E0', edgecolor='black')
    progress_fill = mpatches.Rectangle((5, 4), 0, 0.6, facecolor='#4CAF50')
    ax_stats.add_patch(progress_bg)
    ax_stats.add_patch(progress_fill)

    plt.tight_layout()

    def update(frame_idx):
        if frame_idx >= len(frames):
            return []

        frame = frames[frame_idx]

        # Update cell colors
        for i, status in enumerate(frame['cell_statuses']):
            cell_patches[i].set_facecolor(COLORS[status])

        # Update searcher positions
        for i, pos in enumerate(frame['searcher_positions']):
            if i < len(searcher_markers):
                searcher_markers[i].set_xdata([pos + 0.45])

        # Show key when found
        if frame['key_found']:
            key_marker.set_visible(True)

        # Update text
        time_text.set_text(f"Time: {frame['time']:.1f}s")

        cells = frame['cells_searched']
        total = len(path)
        pct = cells / total * 100 if total > 0 else 0
        coverage_text.set_text(f'Coverage: {cells} / {total} cells ({pct:.0f}%)')

        if frame['key_found']:
            status_text.set_text('Status: KEY FOUND!')
            status_text.set_color('green')
        else:
            status_text.set_text('Status: Searching...')
            status_text.set_color('black')

        # Update progress bar
        progress = cells / total if total > 0 else 0
        progress_fill.set_width(4 * progress)

        return []

    anim = FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=200,
        blit=False
    )

    # Save animation
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    """Generate all animations."""

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    os.makedirs(output_dir, exist_ok=True)

    print("Creating search animations...")
    print("=" * 50)

    # Animation 1: Single searcher with greedy strategy
    print("\n1. Single searcher (Greedy Nearest)...")
    path1 = Path.generate(30, key_position=22, heterogeneous=False)
    marks1 = MarkingSystem(len(path1))
    searcher1 = Searcher(0, start_position=0, strategy=GreedyNearestStrategy())

    frames1 = run_simulation_with_frames(path1, [searcher1], marks1, time_budget=15.0)
    create_animation(
        frames1, path1, 1,
        "Single Agent Search (Greedy Nearest Strategy)",
        os.path.join(output_dir, "anim_single_searcher.gif")
    )

    # Animation 2: Multiple searchers with territorial strategy
    print("2. Three searchers (Territorial Strategy)...")
    path2 = Path.generate(30, key_position=15, heterogeneous=False)
    marks2 = MarkingSystem(len(path2))

    searchers2 = [
        Searcher(0, start_position=0, strategy=TerritorialStrategy(num_searchers=3)),
        Searcher(1, start_position=10, strategy=TerritorialStrategy(num_searchers=3)),
        Searcher(2, start_position=20, strategy=TerritorialStrategy(num_searchers=3)),
    ]

    frames2 = run_simulation_with_frames(path2, searchers2, marks2, time_budget=15.0)
    create_animation(
        frames2, path2, 3,
        "Multi-Agent Territorial Search (3 Searchers)",
        os.path.join(output_dir, "anim_territorial.gif")
    )

    # Animation 3: Probabilistic strategy (key in high-probability zone)
    print("3. Single searcher (Probabilistic Strategy)...")
    path3 = Path.generate(30, key_position=15, heterogeneous=False)
    # Set higher probability in middle section
    for i, cell in enumerate(path3):
        if 10 <= i <= 20:
            cell.prior_probability = 0.08
        else:
            cell.prior_probability = 0.02
    marks3 = MarkingSystem(len(path3))
    searcher3 = Searcher(0, start_position=0, strategy=ProbabilisticStrategy())

    frames3 = run_simulation_with_frames(path3, [searcher3], marks3, time_budget=15.0)
    create_animation(
        frames3, path3, 1,
        "Probabilistic Search (Higher Probability in Center)",
        os.path.join(output_dir, "anim_probabilistic.gif")
    )

    # Animation 4: Random walk for comparison
    print("4. Single searcher (Random Walk)...")
    path4 = Path.generate(30, key_position=22, heterogeneous=False)
    marks4 = MarkingSystem(len(path4))
    searcher4 = Searcher(0, start_position=0, strategy=RandomWalkStrategy())

    frames4 = run_simulation_with_frames(path4, [searcher4], marks4, time_budget=15.0)
    create_animation(
        frames4, path4, 1,
        "Random Walk Search (Baseline)",
        os.path.join(output_dir, "anim_random_walk.gif")
    )

    # Animation 5: Large path with multiple searchers
    print("5. Large path (50 cells, 5 searchers)...")
    path5 = Path.generate(50, key_position=35, heterogeneous=False)
    marks5 = MarkingSystem(len(path5))

    searchers5 = [
        Searcher(i, start_position=i*10, strategy=TerritorialStrategy(num_searchers=5))
        for i in range(5)
    ]

    frames5 = run_simulation_with_frames(path5, searchers5, marks5, time_budget=20.0)
    create_animation(
        frames5, path5, 5,
        "Large-Scale Multi-Agent Search (50 cells, 5 searchers)",
        os.path.join(output_dir, "anim_large_scale.gif")
    )

    print("\n" + "=" * 50)
    print(f"All animations saved to: {output_dir}/")
    print("\nGenerated files:")
    for f in os.listdir(output_dir):
        if f.endswith('.gif'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
