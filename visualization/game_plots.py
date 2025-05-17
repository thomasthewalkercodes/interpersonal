import numpy as np
import matplotlib.pyplot as plt
from typing import List


class GameVisualizer:
    def __init__(self, n_rounds: int):
        self.n_rounds = n_rounds
        self.window = 50

    def create_plots(
        self,
        actions1: List[str],
        actions2: List[str],
        payoffs1: List[float],
        payoffs2: List[float],
    ) -> None:
        """Create and display all game plots"""
        # Calculate probabilities over time using moving average
        actions1_bin = [1 if a == "Up" else 0 for a in actions1]
        actions2_bin = [1 if a == "Left" else 0 for a in actions2]

        p_history = np.convolve(
            actions1_bin, np.ones(self.window) / self.window, mode="valid"
        )
        q_history = np.convolve(
            actions2_bin, np.ones(self.window) / self.window, mode="valid"
        )

        # Compute joint probabilities
        prob_UL = p_history * q_history  # Up-Left
        prob_UR = p_history * (1 - q_history)  # Up-Right
        prob_DL = (1 - p_history) * q_history  # Down-Left
        prob_DR = (1 - p_history) * (1 - q_history)  # Down-Right

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot joint probabilities
        rounds = range(len(p_history))
        ax1.plot(rounds, prob_UL, "darkgreen", label="Up-Left", linewidth=2)
        ax1.plot(rounds, prob_UR, "purple", label="Up-Right", linewidth=2)
        ax1.plot(rounds, prob_DL, "orange", label="Down-Left", linewidth=2)
        ax1.plot(rounds, prob_DR, "brown", label="Down-Right", linewidth=2)
        ax1.set_title("Joint Strategy Probabilities Over Time")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Probability")
        ax1.legend(loc="center right")
        ax1.grid(True)

        # Plot payoffs/utilities
        payoffs1_ma = np.convolve(
            payoffs1, np.ones(self.window) / self.window, mode="valid"
        )
        payoffs2_ma = np.convolve(
            payoffs2, np.ones(self.window) / self.window, mode="valid"
        )

        ax2.plot(payoffs1_ma, "blue", label="Player 1", linewidth=2)
        ax2.plot(payoffs2_ma, "red", label="Player 2", linewidth=2)
        ax2.set_title("Average Payoffs Over Time")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Payoff")
        ax2.legend(loc="center right")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
