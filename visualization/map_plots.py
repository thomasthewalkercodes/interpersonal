import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns


class CircularGameVisualizer:
    def __init__(self, n_rounds: int):
        self.n_rounds = n_rounds
        # Remove seaborn style setting and use a standard matplotlib style
        plt.style.use("default")

    def create_analysis_plots(self, actions1, actions2, payoffs1, payoffs2):
        # Set figure style parameters
        plt.rcParams["figure.figsize"] = [15, 15]
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["grid.alpha"] = 0.3

        # Create a figure with 2x2 subplots
        fig = plt.figure()

        # 1. Trajectory Plot (top left)
        ax1 = fig.add_subplot(221)
        self._plot_trajectories(ax1, actions1, actions2)

        # 2. Payoff Evolution (top right)
        ax2 = fig.add_subplot(222)
        self._plot_payoffs(ax2, payoffs1, payoffs2)

        # 3. Position Density (bottom left)
        ax3 = fig.add_subplot(223)
        self._plot_position_density(ax3, actions1, actions2)

        # 4. Behavioral Analysis (bottom right)
        ax4 = fig.add_subplot(224)
        self._plot_behavioral_metrics(ax4, actions1, actions2)

        plt.tight_layout()
        plt.show()

        # Create separate animation
        self._create_movement_animation(actions1, actions2)

    def _plot_trajectories(self, ax, actions1, actions2):
        """Plot agent trajectories with time-based coloring"""
        x1, y1 = zip(*actions1)
        x2, y2 = zip(*actions2)

        # Draw unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color="black", linestyle="--")
        ax.add_artist(circle)

        # Plot trajectories with color gradient
        points = np.linspace(0, 1, len(x1))
        scatter1 = ax.scatter(
            x1, y1, c=points, cmap="Blues", label="Agent 1", alpha=0.6
        )
        scatter2 = ax.scatter(x2, y2, c=points, cmap="Reds", label="Agent 2", alpha=0.6)

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlabel("Communion")
        ax.set_ylabel("Agency")
        ax.set_title("Agent Trajectories\n(darker = later in time)")
        ax.legend()

    def _plot_payoffs(self, ax, payoffs1, payoffs2):
        """Plot payoff evolution with rolling average"""
        window = 50  # Rolling average window
        rolling1 = np.convolve(payoffs1, np.ones(window) / window, mode="valid")
        rolling2 = np.convolve(payoffs2, np.ones(window) / window, mode="valid")

        ax.plot(payoffs1, "b-", alpha=0.2, label="Agent 1 (raw)")
        ax.plot(payoffs2, "r-", alpha=0.2, label="Agent 2 (raw)")
        ax.plot(rolling1, "b-", label=f"Agent 1 ({window}-round avg)")
        ax.plot(rolling2, "r-", label=f"Agent 2 ({window}-round avg)")
        ax.set_xlabel("Round")
        ax.set_ylabel("Payoff")
        ax.set_title("Payoff Evolution")
        ax.grid(True)
        ax.legend()

    def _plot_position_density(self, ax, actions1, actions2):
        """Create heatmap of position density"""
        x1, y1 = zip(*actions1)
        x2, y2 = zip(*actions2)

        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            x1 + x2, y1 + y2, bins=20, range=[[-1, 1], [-1, 1]]
        )

        # Plot heatmap
        sns.heatmap(heatmap.T, ax=ax, cmap="viridis")
        ax.set_title("Position Density\n(combined for both agents)")
        ax.set_xlabel("Communion")
        ax.set_ylabel("Agency")

    def _plot_behavioral_metrics(self, ax, actions1, actions2):
        """Plot behavioral metrics over time"""
        x1, y1 = zip(*actions1)
        x2, y2 = zip(*actions2)

        # Calculate various metrics
        communion_diff = [abs(x2[i] - x1[i]) for i in range(len(x1))]
        agency_mirror = [
            abs(y2[i] + y1[i]) for i in range(len(y1))
        ]  # Should be close to 0

        window = 50  # Rolling average window
        comm_rolling = np.convolve(
            communion_diff, np.ones(window) / window, mode="valid"
        )
        agency_rolling = np.convolve(
            agency_mirror, np.ones(window) / window, mode="valid"
        )

        ax.plot(comm_rolling, "g-", label="Communion Difference")
        ax.plot(agency_rolling, "b-", label="Agency Mirror Error")
        ax.set_xlabel("Round")
        ax.set_ylabel("Error Metric")
        ax.set_title("Behavioral Analysis\n(lower = better coordination)")
        ax.grid(True)
        ax.legend()

    def _create_movement_animation(self, actions1, actions2):
        """Create animation of agent movements"""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        circle = plt.Circle((0, 0), 1, fill=False, color="black", linestyle="--")
        ax.add_artist(circle)

        x1, y1 = zip(*actions1)
        x2, y2 = zip(*actions2)

        (line1,) = ax.plot([], [], "bo-", label="Agent 1", alpha=0.6)
        (line2,) = ax.plot([], [], "ro-", label="Agent 2", alpha=0.6)

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return line1, line2

        def animate(i):
            window = 20  # Show last 20 positions
            start = max(0, i - window)
            line1.set_data(x1[start:i], y1[start:i])
            line2.set_data(x2[start:i], y2[start:i])
            return line1, line2

        anim = FuncAnimation(
            fig, animate, init_func=init, frames=len(x1), interval=20, blit=True
        )

        ax.set_xlabel("Communion")
        ax.set_ylabel("Agency")
        ax.set_title("Agent Movement Animation")
        ax.grid(True)
        ax.legend()
        plt.show()
