import numpy as np
import matplotlib.pyplot as plt
from typing import List


class GameVisualizer:
    """Class responsible for visualizing game results"""

    def __init__(self, n_rounds: int):
        self.n_rounds = n_rounds
        self.window = 50  # Window size for moving average

    def create_plots(
        self,
        actions1: List[str],
        actions2: List[str],
        payoffs1: List[float],
        payoffs2: List[float],
    ) -> None:
        """Create and display all game plots"""
        # Convert actions to binary
        actions1_bin = [1 if a == "Up" else 0 for a in actions1]
        actions2_bin = [1 if a == "Left" else 0 for a in actions2]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        self._plot_strategies(ax1, actions1_bin, actions2_bin)
        self._plot_raw_payoffs(ax2, payoffs1, payoffs2)
        self._plot_average_payoffs(ax3, payoffs1, payoffs2)
        self._plot_joint_frequencies(ax4, actions1_bin, actions2_bin)

        plt.tight_layout()
        plt.show()

    def _plot_strategies(self, ax, actions1: List[int], actions2: List[int]) -> None:
        """Plot moving average of strategies"""
        actions1_ma = np.convolve(
            actions1, np.ones(self.window) / self.window, mode="valid"
        )
        actions2_ma = np.convolve(
            actions2, np.ones(self.window) / self.window, mode="valid"
        )

        ax.plot(actions1_ma, label="Player 1 (Up)")
        ax.plot(actions2_ma, label="Player 2 (Left)")
        ax.set_title("Strategy Evolution")
        ax.set_xlabel("Round")
        ax.set_ylabel("Probability")
        ax.legend()
        ax.grid(True)

    def _plot_raw_payoffs(
        self, ax, payoffs1: List[float], payoffs2: List[float]
    ) -> None:
        """Plot raw payoffs over time"""
        ax.plot(payoffs1, label="Player 1", alpha=0.3)
        ax.plot(payoffs2, label="Player 2", alpha=0.3)
        ax.set_title("Payoffs Over Time")
        ax.set_xlabel("Round")
        ax.set_ylabel("Payoff")
        ax.legend()
        ax.grid(True)

    def _plot_average_payoffs(
        self, ax, payoffs1: List[float], payoffs2: List[float]
    ) -> None:
        """Plot moving average of payoffs"""
        payoffs1_ma = np.convolve(
            payoffs1, np.ones(self.window) / self.window, mode="valid"
        )
        payoffs2_ma = np.convolve(
            payoffs2, np.ones(self.window) / self.window, mode="valid"
        )

        ax.plot(payoffs1_ma, label="Player 1")
        ax.plot(payoffs2_ma, label="Player 2")
        ax.set_title("Moving Average of Payoffs")
        ax.set_xlabel("Round")
        ax.set_ylabel("Average Payoff")
        ax.legend()
        ax.grid(True)

    def _plot_joint_frequencies(
        self, ax, actions1: List[int], actions2: List[int]
    ) -> None:
        """Plot joint action frequencies"""
        action_pairs = list(zip(actions1, actions2))
        frequencies = [
            action_pairs.count((1, 1)) / self.n_rounds,
            action_pairs.count((1, 0)) / self.n_rounds,
            action_pairs.count((0, 1)) / self.n_rounds,
            action_pairs.count((0, 0)) / self.n_rounds,
        ]

        ax.bar(range(4), frequencies)
        ax.set_xticks(range(4))
        ax.set_xticklabels(
            ["Up-Left", "Up-Right", "Down-Left", "Down-Right"], rotation=45
        )
        ax.set_title("Joint Action Frequencies")
        ax.set_ylabel("Frequency")
        ax.grid(True)
