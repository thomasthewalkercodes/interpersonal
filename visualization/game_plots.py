import numpy as np
import matplotlib.pyplot as plt
from typing import List
from Configurations import N_ROUNDS  # Add this import


class GameVisualizer:
    def __init__(self, n_rounds: int = N_ROUNDS):  # Make N_ROUNDS the default
        self.n_rounds = n_rounds
        self.window = 50

    def create_plots(
        self,
        actions1: List[str],
        actions2: List[str],
        payoffs1: List[float],
        payoffs2: List[float],
        initial_probs: dict = None,
    ) -> None:
        """Create and display all game plots"""
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

        # Add initial probabilities
        if initial_probs:
            prob_UL = np.concatenate(([initial_probs["UL"]], prob_UL))
            prob_UR = np.concatenate(([initial_probs["UR"]], prob_UR))
            prob_DL = np.concatenate(([initial_probs["DL"]], prob_DL))
            prob_DR = np.concatenate(([initial_probs["DR"]], prob_DR))
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
