import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random


class BayesianCoinAgent:
    def __init__(self, prior_heads=1, prior_tails=1):
        """
        Initialize Bayesian agent with Beta prior
        prior_heads, prior_tails: parameters for Beta(alpha, beta) prior
        Beta(1,1) = uniform prior (fair coin assumption)
        """
        self.alpha = prior_heads  # prior belief in heads + observed heads
        self.beta = prior_tails  # prior belief in tails + observed tails
        self.predictions = []  # track predicted probabilities
        self.confidences = []  # track confidence levels
        self.observations = []  # track actual coin flips

    def predict_bias(self):
        """Predict the coin's bias (probability of heads)"""
        # Mean of Beta distribution
        predicted_prob = self.alpha / (self.alpha + self.beta)
        return predicted_prob

    def get_confidence(self):
        """
        Calculate confidence as inverse of variance
        Higher precision (lower variance) = higher confidence
        """
        # Variance of Beta distribution
        variance = (self.alpha * self.beta) / (
            (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        )
        # Convert to confidence (inverse relationship with variance)
        confidence = 1 / (1 + variance * 100)  # scaled for better visualization
        return confidence

    def update_belief(self, observation):
        """Update belief based on coin flip observation"""
        if observation == 1:  # heads
            self.alpha += 1
        else:  # tails
            self.beta += 1

        # Store current prediction and confidence
        self.predictions.append(self.predict_bias())
        self.confidences.append(self.get_confidence())
        self.observations.append(observation)


class CoinFlipGame:
    def __init__(self, true_coin_bias=0.5, num_flips=300):
        """
        Initialize the coin flip game
        true_coin_bias: actual probability of heads (0.5 = fair coin)
        num_flips: number of coin flips to simulate
        """
        self.true_bias = true_coin_bias
        self.num_flips = num_flips
        self.coin_flips = []

    def flip_coin(self):
        """Simulate a single coin flip"""
        return 1 if random.random() < self.true_bias else 0

    def run_simulation(self, agent):
        """Run the full simulation"""
        print(f"Starting simulation with {self.num_flips} flips")
        print(f"True coin bias: {self.true_bias}")
        print(f"Agent's prior belief: {agent.predict_bias():.3f}")
        print("-" * 50)

        for flip_num in range(self.num_flips):
            # Flip the coin
            result = self.flip_coin()
            self.coin_flips.append(result)

            # Agent updates its belief
            agent.update_belief(result)

            # Print progress every 50 flips
            if (flip_num + 1) % 50 == 0:
                current_pred = agent.predict_bias()
                current_conf = agent.get_confidence()
                heads_so_far = sum(self.coin_flips)
                print(
                    f"Flip {flip_num + 1:3d}: Predicted bias = {current_pred:.3f}, "
                    f"Confidence = {current_conf:.3f}, "
                    f"Actual rate = {heads_so_far/(flip_num+1):.3f}"
                )

        return agent

    def plot_results(self, agent):
        """Plot the results of the simulation"""
        flips = range(1, len(agent.predictions) + 1)

        # Calculate running average of actual flips
        running_avg = np.cumsum(self.coin_flips) / np.arange(
            1, len(self.coin_flips) + 1
        )

        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Predicted bias vs actual bias over time
        ax1.plot(
            flips, agent.predictions, "b-", label="Agent's Predicted Bias", linewidth=2
        )
        ax1.plot(flips, running_avg, "r-", label="Actual Running Average", linewidth=2)
        ax1.axhline(
            y=self.true_bias,
            color="g",
            linestyle="--",
            label=f"True Bias ({self.true_bias})",
        )
        ax1.set_xlabel("Number of Coin Flips")
        ax1.set_ylabel("Probability of Heads")
        ax1.set_title("Bayesian Agent's Bias Prediction vs Reality")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Confidence over time
        ax2.plot(flips, agent.confidences, "purple", linewidth=2)
        ax2.set_xlabel("Number of Coin Flips")
        ax2.set_ylabel("Agent's Confidence Level")
        ax2.set_title("Agent's Confidence in Predictions Over Time")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Final summary
        final_prediction = agent.predictions[-1]
        final_confidence = agent.confidences[-1]
        actual_rate = sum(self.coin_flips) / len(self.coin_flips)
        prediction_error = abs(final_prediction - self.true_bias)

        print("\n" + "=" * 60)
        print("FINAL RESULTS:")
        print("=" * 60)
        print(f"True coin bias:           {self.true_bias:.4f}")
        print(f"Agent's final prediction: {final_prediction:.4f}")
        print(f"Actual observed rate:     {actual_rate:.4f}")
        print(f"Prediction error:         {prediction_error:.4f}")
        print(f"Final confidence:         {final_confidence:.4f}")
        print(
            f"Agent accuracy: {'GOOD' if prediction_error < 0.05 else 'NEEDS MORE DATA'}"
        )


# Example usage and game setup
def run_game(coin_bias=0.6, num_flips=300, prior_belief_heads=1, prior_belief_tails=1):
    """
    Run the complete Bayesian coin flip game

    Parameters:
    coin_bias: True probability of heads (0.5 = fair, 0.6 = biased toward heads)
    num_flips: Number of coin flips to simulate
    prior_belief_heads: Prior belief parameter for heads (Beta distribution alpha)
    prior_belief_tails: Prior belief parameter for tails (Beta distribution beta)
    """

    # Create the game and agent
    game = CoinFlipGame(true_coin_bias=coin_bias, num_flips=num_flips)
    agent = BayesianCoinAgent(
        prior_heads=prior_belief_heads, prior_tails=prior_belief_tails
    )

    # Run simulation
    agent = game.run_simulation(agent)

    # Plot results
    game.plot_results(agent)

    return game, agent


# Run the game with default parameters
if __name__ == "__main__":
    print("Bayesian Coin Flip Prediction Game")
    print("=" * 40)

    # Example 1: Biased coin (60% heads) with fair prior
    print("\nExample 1: Biased coin (60% heads) with fair prior")
    game1, agent1 = run_game(
        coin_bias=0.6, num_flips=300, prior_belief_heads=1, prior_belief_tails=1
    )

    # Example 2: Fair coin with fair prior
    print("\nExample 2: Fair coin (50% heads) with fair prior")
    game2, agent2 = run_game(
        coin_bias=0.5, num_flips=300, prior_belief_heads=1, prior_belief_tails=1
    )

    # Example 3: Fair coin with strong prior of 0.5 (equivalent to 100 previous flips)
    print(
        "\nExample 3: Fair coin with strong prior (equivalent to 100 previous fair flips)"
    )
    game3, agent3 = run_game(
        coin_bias=0.3,
        num_flips=300,
        prior_belief_heads=50,  # 50 heads out of 100 prior observations
        prior_belief_tails=50,  # 50 tails out of 100 prior observations
    )
