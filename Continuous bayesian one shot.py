import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import minimize_scalar
import seaborn as sns


class BayesianGameAgent:
    def __init__(self, name, own_payoffs, prior_alpha=2, prior_beta=2, rationality=5.0):
        """
        Initialize a Bayesian learning agent for 2x2 games.

        Parameters:
        - name: Agent identifier
        - own_payoffs: 2x2 array of agent's own payoffs [[CC, CD], [DC, DD]]
        - prior_alpha, prior_beta: Beta distribution parameters for opponent's cooperation preference
        - rationality: How rational the agent is (higher = more optimal play)
        """
        self.name = name
        self.own_payoffs = np.array(own_payoffs)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.current_alpha = prior_alpha
        self.current_beta = prior_beta
        self.rationality = rationality
        self.history = []
        self.belief_history = []

    def get_current_belief_distribution(self):
        """Return current Beta distribution parameters"""
        return beta(self.current_alpha, self.current_beta)

    def get_belief_stats(self):
        """Get mean and std of current belief about opponent's cooperation preference"""
        dist = self.get_current_belief_distribution()
        return dist.mean(), dist.std()

    def opponent_cooperation_probability(self, theta, my_action):
        """
        Model how likely opponent is to cooperate given their type theta and my action.
        theta=1 means pure cooperator, theta=0 means pure defector
        """
        if my_action == "C":
            # If I cooperate, cooperation likelihood depends on opponent's type
            base_prob = theta
        else:
            # If I defect, even cooperators are less likely to cooperate
            base_prob = theta * 0.3  # Cooperators still might cooperate sometimes

        return np.clip(base_prob, 0.05, 0.95)  # Keep probabilities bounded

    def expected_payoff(self, my_action, opponent_coop_prob):
        """Calculate expected payoff given my action and opponent's cooperation probability"""
        if my_action == "C":
            return (
                opponent_coop_prob * self.own_payoffs[0, 0]
                + (1 - opponent_coop_prob) * self.own_payoffs[0, 1]
            )
        else:
            return (
                opponent_coop_prob * self.own_payoffs[1, 0]
                + (1 - opponent_coop_prob) * self.own_payoffs[1, 1]
            )

    def choose_action(self, n_samples=1000):
        """
        Choose action by integrating over belief distribution about opponent's type.
        Uses Monte Carlo sampling for the integral.
        """
        # Sample theta values from current belief distribution
        belief_dist = self.get_current_belief_distribution()
        theta_samples = belief_dist.rvs(n_samples)

        # Calculate expected payoffs for each action
        coop_payoffs = []
        defect_payoffs = []

        for theta in theta_samples:
            opp_coop_prob = self.opponent_cooperation_probability(theta, "C")
            coop_payoffs.append(self.expected_payoff("C", opp_coop_prob))

            opp_coop_prob = self.opponent_cooperation_probability(theta, "D")
            defect_payoffs.append(self.expected_payoff("D", opp_coop_prob))

        avg_coop_payoff = np.mean(coop_payoffs)
        avg_defect_payoff = np.mean(defect_payoffs)

        # Use logistic choice based on rationality parameter
        payoff_diff = self.rationality * (avg_coop_payoff - avg_defect_payoff)
        coop_probability = 1 / (1 + np.exp(-payoff_diff))

        # Choose action
        action = "C" if np.random.random() < coop_probability else "D"

        # Store decision info
        decision_info = {
            "action": action,
            "coop_probability": coop_probability,
            "expected_coop_payoff": avg_coop_payoff,
            "expected_defect_payoff": avg_defect_payoff,
        }

        return action, decision_info

    def update_beliefs(self, my_action, opponent_action):
        """Update beliefs about opponent using Bayesian updating"""

        # Likelihood function: P(opponent_action | theta, my_action)
        def likelihood(theta):
            expected_coop_prob = self.opponent_cooperation_probability(theta, my_action)
            if opponent_action == "C":
                return expected_coop_prob
            else:
                return 1 - expected_coop_prob

        # Bayesian update using conjugate prior properties
        # This is an approximation - for exact updating we'd need numerical integration

        # Sample from current prior
        n_samples = 10000
        current_dist = self.get_current_belief_distribution()
        theta_samples = current_dist.rvs(n_samples)

        # Calculate likelihood for each sample
        likelihoods = np.array([likelihood(theta) for theta in theta_samples])

        # Weight samples by likelihood
        weights = likelihoods / np.sum(likelihoods)

        # Estimate new distribution parameters using method of moments
        weighted_mean = np.sum(weights * theta_samples)
        weighted_var = np.sum(weights * (theta_samples - weighted_mean) ** 2)

        # Convert back to Beta parameters
        if weighted_var > 0 and weighted_mean > 0 and weighted_mean < 1:
            common_factor = weighted_mean * (1 - weighted_mean) / weighted_var - 1
            new_alpha = weighted_mean * common_factor
            new_beta = (1 - weighted_mean) * common_factor

            # Smooth updating to avoid extreme jumps
            learning_rate = 0.7
            self.current_alpha = (
                learning_rate * new_alpha + (1 - learning_rate) * self.current_alpha
            )
            self.current_beta = (
                learning_rate * new_beta + (1 - learning_rate) * self.current_beta
            )

        # Store history
        self.history.append((my_action, opponent_action))
        mean_belief, std_belief = self.get_belief_stats()
        self.belief_history.append((mean_belief, std_belief))

    def plot_belief_evolution(self):
        """Plot how beliefs about opponent evolved over time"""
        if not self.belief_history:
            return

        means = [b[0] for b in self.belief_history]
        stds = [b[1] for b in self.belief_history]
        rounds = range(1, len(means) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, means, "b-", linewidth=2, label="Mean belief (θ)")
        plt.fill_between(
            rounds,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.3,
            label="±1 std",
        )
        plt.xlabel("Round")
        plt.ylabel("Belief about opponent cooperation preference")
        plt.title(f"{self.name} - Evolution of Beliefs")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.show()

    def plot_current_belief_distribution(self):
        """Plot current belief distribution"""
        dist = self.get_current_belief_distribution()
        x = np.linspace(0, 1, 100)
        y = dist.pdf(x)

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, "b-", linewidth=2)
        plt.fill_between(x, y, alpha=0.3)
        plt.xlabel("Opponent cooperation preference (θ)")
        plt.ylabel("Probability density")
        plt.title(f"{self.name} - Current Belief Distribution")
        plt.grid(True, alpha=0.3)
        plt.show()


class GameSimulator:
    def __init__(self, agent1, agent2):
        self.agent1 = agent1
        self.agent2 = agent2
        self.game_history = []

    def play_round(self):
        """Play one round of the game"""
        # Both agents choose actions
        action1, info1 = self.agent1.choose_action()
        action2, info2 = self.agent2.choose_action()

        # Update beliefs
        self.agent1.update_beliefs(action1, action2)
        self.agent2.update_beliefs(action2, action1)

        # Calculate payoffs
        payoff_map = {
            ("C", "C"): (0, 0),
            ("C", "D"): (0, 1),
            ("D", "C"): (1, 0),
            ("D", "D"): (1, 1),
        }
        i, j = payoff_map[(action1, action2)]
        payoff1 = self.agent1.own_payoffs[i, j]
        payoff2 = self.agent2.own_payoffs[i, j]

        # Store round info
        round_info = {
            "agent1_action": action1,
            "agent2_action": action2,
            "agent1_payoff": payoff1,
            "agent2_payoff": payoff2,
            "agent1_info": info1,
            "agent2_info": info2,
        }

        self.game_history.append(round_info)
        return round_info

    def simulate(self, n_rounds):
        """Simulate n rounds of the game"""
        print(f"Starting simulation: {n_rounds} rounds")
        print("=" * 50)

        for round_num in range(n_rounds):
            round_info = self.play_round()

            if round_num < 10 or round_num % 10 == 0:
                print(f"Round {round_num + 1}:")
                print(
                    f"  Actions: {self.agent1.name}={round_info['agent1_action']}, "
                    f"{self.agent2.name}={round_info['agent2_action']}"
                )
                print(
                    f"  Payoffs: {self.agent1.name}={round_info['agent1_payoff']}, "
                    f"{self.agent2.name}={round_info['agent2_payoff']}"
                )

                mean1, std1 = self.agent1.get_belief_stats()
                mean2, std2 = self.agent2.get_belief_stats()
                print(
                    f"  {self.agent1.name} believes {self.agent2.name} cooperativeness: "
                    f"{mean1:.3f} ± {std1:.3f}"
                )
                print(
                    f"  {self.agent2.name} believes {self.agent1.name} cooperativeness: "
                    f"{mean2:.3f} ± {std2:.3f}"
                )
                print()

    def plot_game_summary(self):
        """Plot summary of the game"""
        if not self.game_history:
            return

        # Extract data
        actions1 = [r["agent1_action"] for r in self.game_history]
        actions2 = [r["agent2_action"] for r in self.game_history]
        payoffs1 = [r["agent1_payoff"] for r in self.game_history]
        payoffs2 = [r["agent2_payoff"] for r in self.game_history]

        cooperation1 = [1 if a == "C" else 0 for a in actions1]
        cooperation2 = [1 if a == "C" else 0 for a in actions2]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Cooperation rates over time
        window = max(1, len(cooperation1) // 20)
        coop_rate1 = np.convolve(cooperation1, np.ones(window) / window, mode="valid")
        coop_rate2 = np.convolve(cooperation2, np.ones(window) / window, mode="valid")

        axes[0, 0].plot(range(len(coop_rate1)), coop_rate1, label=self.agent1.name)
        axes[0, 0].plot(range(len(coop_rate2)), coop_rate2, label=self.agent2.name)
        axes[0, 0].set_title("Cooperation Rate Over Time")
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].set_ylabel("Cooperation Rate")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Cumulative payoffs
        cum_payoffs1 = np.cumsum(payoffs1)
        cum_payoffs2 = np.cumsum(payoffs2)

        axes[0, 1].plot(cum_payoffs1, label=self.agent1.name)
        axes[0, 1].plot(cum_payoffs2, label=self.agent2.name)
        axes[0, 1].set_title("Cumulative Payoffs")
        axes[0, 1].set_xlabel("Round")
        axes[0, 1].set_ylabel("Cumulative Payoff")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Action frequency heatmap
        action_counts = {"CC": 0, "CD": 0, "DC": 0, "DD": 0}
        for r in self.game_history:
            key = r["agent1_action"] + r["agent2_action"]
            action_counts[key] += 1

        heatmap_data = np.array(
            [
                [action_counts["CC"], action_counts["CD"]],
                [action_counts["DC"], action_counts["DD"]],
            ]
        )

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt="d",
            xticklabels=[f"{self.agent2.name} C", f"{self.agent2.name} D"],
            yticklabels=[f"{self.agent1.name} C", f"{self.agent1.name} D"],
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("Action Combination Frequencies")

        # Final belief distributions
        dist1 = self.agent1.get_current_belief_distribution()
        dist2 = self.agent2.get_current_belief_distribution()

        x = np.linspace(0, 1, 100)
        axes[1, 1].plot(
            x, dist1.pdf(x), label=f"{self.agent1.name} belief about {self.agent2.name}"
        )
        axes[1, 1].plot(
            x, dist2.pdf(x), label=f"{self.agent2.name} belief about {self.agent1.name}"
        )
        axes[1, 1].set_title("Final Belief Distributions")
        axes[1, 1].set_xlabel("Opponent cooperation preference")
        axes[1, 1].set_ylabel("Probability density")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Example usage and configuration
def create_example_simulation():
    """Create an example simulation with configurable parameters"""

    # Define payoff matrices
    # Format: [[CC, CD], [DC, DD]] where first letter is own action, second is opponent's

    # Classic Prisoner's Dilemma
    prisoner_payoffs = [
        [3, 0],
        [5, 1],
    ]  # Temptation=5, Reward=3, Punishment=1, Sucker=0

    # Coordination Game
    coordination_payoffs = [[3, 0], [0, 2]]  # Prefer to match opponent's action

    # Create agents with different priors and rationality
    agent1 = BayesianGameAgent(
        name="Alice",
        own_payoffs=prisoner_payoffs,
        prior_alpha=1,  # Skeptical prior (expects opponent to be selfish)
        prior_beta=3,
        rationality=3.0,  # Moderately rational
    )

    agent2 = BayesianGameAgent(
        name="Bob",
        own_payoffs=prisoner_payoffs,
        prior_alpha=3,  # Optimistic prior (expects opponent to be cooperative)
        prior_beta=1,
        rationality=5.0,  # Highly rational
    )

    return GameSimulator(agent1, agent2)


# Run example simulation
if __name__ == "__main__":
    # Create and run simulation
    sim = create_example_simulation()

    print("Initial beliefs:")
    mean1, std1 = sim.agent1.get_belief_stats()
    mean2, std2 = sim.agent2.get_belief_stats()
    print(f"Alice expects Bob's cooperativeness: {mean1:.3f} ± {std1:.3f}")
    print(f"Bob expects Alice's cooperativeness: {mean2:.3f} ± {std2:.3f}")
    print()

    # Run simulation
    sim.simulate(50)

    # Plot results
    sim.plot_game_summary()
    sim.agent1.plot_belief_evolution()
    sim.agent2.plot_belief_evolution()
    sim.agent1.plot_current_belief_distribution()
    sim.agent2.plot_current_belief_distribution()

    print("\nFinal Results:")
    print(
        f"Total payoffs - Alice: {sum(r['agent1_payoff'] for r in sim.game_history)}, "
        f"Bob: {sum(r['agent2_payoff'] for r in sim.game_history)}"
    )

    final_mean1, final_std1 = sim.agent1.get_belief_stats()
    final_mean2, final_std2 = sim.agent2.get_belief_stats()
    print(f"Alice's final belief about Bob: {final_mean1:.3f} ± {final_std1:.3f}")
    print(f"Bob's final belief about Alice: {final_mean2:.3f} ± {final_std2:.3f}")
