import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict
from Configurations import N_ROUNDS


class GameVisualizer:
    def __init__(self, n_rounds: int = N_ROUNDS):
        self.n_rounds = n_rounds
        self.window = 50

    def create_plots(
        self,
        actions1: List[str],
        actions2: List[str],
        payoffs1: List[float],
        payoffs2: List[float],
        initial_probs: Dict[str, float],
        game_type: str = "matrix",
    ) -> None:
        """Create and display all game plots"""
        if game_type == "matrix":
            self._create_matrix_plots(actions1, actions2, payoffs1, payoffs2)
        else:
            self._create_circular_plots(actions1, actions2, payoffs1, payoffs2)

    def _create_matrix_plots(
        self,
        actions1: List[str],
        actions2: List[str],
        payoffs1: List[float],
        payoffs2: List[float],
    ) -> None:
        """Create visualization for matrix game"""
        # Calculate probabilities over time
        actions1_bin = [1 if a == "Up" else 0 for a in actions1]
        actions2_bin = [1 if a == "Left" else 0 for a in actions2]

        # Calculate moving averages
        p_history = np.convolve(
            actions1_bin, np.ones(self.window) / self.window, mode="valid"
        )
        q_history = np.convolve(
            actions2_bin, np.ones(self.window) / self.window, mode="valid"
        )

        # Calculate joint probabilities
        prob_UL = p_history * q_history
        prob_UR = p_history * (1 - q_history)
        prob_DL = (1 - p_history) * q_history
        prob_DR = (1 - p_history) * (1 - q_history)

        # Create x-axis values for plotting
        rounds = range(len(prob_UL))

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot joint probabilities
        ax1.plot(rounds, prob_UL, "darkgreen", label="Up-Left", linewidth=2)
        ax1.plot(rounds, prob_UR, "purple", label="Up-Right", linewidth=2)
        ax1.plot(rounds, prob_DL, "orange", label="Down-Left", linewidth=2)
        ax1.plot(rounds, prob_DR, "brown", label="Down-Right", linewidth=2)
        ax1.set_title("Joint Strategy Probabilities")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Probability")
        ax1.legend(loc="center right")
        ax1.grid(True)

        # Plot moving average of payoffs
        payoffs1_ma = np.convolve(
            payoffs1, np.ones(self.window) / self.window, mode="valid"
        )
        payoffs2_ma = np.convolve(
            payoffs2, np.ones(self.window) / self.window, mode="valid"
        )

        # Create x-axis values for payoff plot
        payoff_rounds = range(len(payoffs1_ma))

        ax2.plot(payoff_rounds, payoffs1_ma, "blue", label="Player 1", linewidth=2)
        ax2.plot(payoff_rounds, payoffs2_ma, "red", label="Player 2", linewidth=2)
        ax2.set_title("Average Payoffs")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Payoff")
        ax2.legend(loc="center right")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def _create_circular_plots(
        self,
        actions1: List[str],
        actions2: List[str],
        payoffs1: List[float],
        payoffs2: List[float],
    ) -> None:
        """Create visualization for circular game"""
        fig = plt.figure(figsize=(15, 5))

        # Plot 1: Trajectories in circular space
        ax1 = fig.add_subplot(131)
        x1, y1 = zip(*actions1)  # Split (x,y) coordinates for agent 1
        x2, y2 = zip(*actions2)  # Split (x,y) coordinates for agent 2

        # Draw unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color="black", linestyle="--")
        ax1.add_artist(circle)

        # Plot trajectories with color gradient for time
        points = np.linspace(0, 1, len(x1))
        ax1.scatter(x1, y1, c=points, cmap="Blues", label="Agent 1", alpha=0.6)
        ax1.scatter(x2, y2, c=points, cmap="Reds", label="Agent 2", alpha=0.6)

        ax1.set_xlim(-1.1, 1.1)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_aspect("equal")
        ax1.grid(True)
        ax1.set_xlabel("Communion")
        ax1.set_ylabel("Agency")
        ax1.set_title("Agent Trajectories")
        ax1.legend()

        # Plot 2: Payoffs over time
        ax2 = fig.add_subplot(132)
        ax2.plot(payoffs1, "b-", label="Agent 1", alpha=0.6)
        ax2.plot(payoffs2, "r-", label="Agent 2", alpha=0.6)
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Payoff")
        ax2.set_title("Payoffs over Time")
        ax2.grid(True)
        ax2.legend()

        # Plot 3: Distance between agents over time
        ax3 = fig.add_subplot(133)
        distances = [
            np.sqrt((x2[i] - x1[i]) ** 2 + (y2[i] - y1[i]) ** 2) for i in range(len(x1))
        ]
        ax3.plot(distances, "g-", label="Inter-agent Distance")
        ax3.set_xlabel("Round")
        ax3.set_ylabel("Distance")
        ax3.set_title("Distance between Agents")
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        plt.show()

        # Create animation of movement (optional)
        self._create_animation(x1, y1, x2, y2)

    def _create_animation(self, x1, y1, x2, y2):
        """Create animation of agent movements"""
        fig, ax = plt.subplots()
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        circle = plt.Circle((0, 0), 1, fill=False, color="black", linestyle="--")
        ax.add_artist(circle)
        ax.grid(True)

        (line1,) = ax.plot([], [], "bo-", label="Agent 1", alpha=0.6)
        (line2,) = ax.plot([], [], "ro-", label="Agent 2", alpha=0.6)

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return line1, line2

        def animate(i):
            line1.set_data(x1[:i], y1[:i])
            line2.set_data(x2[:i], y2[:i])
            return line1, line2

        anim = FuncAnimation(
            fig, animate, init_func=init, frames=len(x1), interval=20, blit=True
        )

        ax.set_xlabel("Communion")
        ax.set_ylabel("Agency")
        ax.set_title("Agent Movement Animation")
        ax.legend()
        plt.show()
