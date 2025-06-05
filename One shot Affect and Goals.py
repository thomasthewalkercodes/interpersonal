import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn as sns
from dataclasses import dataclass


@dataclass
class Goal:
    """Represents an interpersonal goal"""

    value: float  # Target value (-1 to 1)
    strength: float  # How important/rigid this goal is (0 to 1)
    confidence: float  # How certain we are about this goal (0 to 1)
    adaptability: float  # How easily this goal changes (0 to 1)


class InterpersonalAgent:
    def __init__(self, name, initial_alpha=5, initial_beta=5, security_threshold=0.1):
        self.name = name

        # Bayesian priors for warmth expectations
        self.alpha = initial_alpha  # warm interaction successes
        self.beta = initial_beta  # cold interaction failures

        # Derived parameters
        self.sample_size = self.alpha + self.beta  # self-esteem (conviction strength)
        self.security_ratio = self.alpha / (
            self.alpha + self.beta
        )  # comfortable warmth ratio
        self.security_threshold = security_threshold

        # Current affect state
        self.arousal = 0.0
        self.valence = 0.0

        # Goal: what % of interactions should be warm
        self.warmth_goal = self.security_ratio

        # History tracking
        self.history = {
            "alpha": [self.alpha],
            "beta": [self.beta],
            "sample_size": [self.sample_size],
            "security_ratio": [self.security_ratio],
            "warmth_goal": [self.warmth_goal],
            "arousal": [self.arousal],
            "valence": [self.valence],
            "behavior": [],
            "received_behavior": [],
        }

        # Interpersonal goals (trait and state)
        self.agency_trait = Goal(0.0, 0.5, 0.8, 0.3)  # Baseline agency
        self.communion_trait = Goal(0.0, 0.5, 0.8, 0.3)  # Baseline communion

        self.agency_state = Goal(0.0, 0.5, 0.5, 0.7)  # Current agency
        self.communion_state = Goal(0.0, 0.5, 0.5, 0.7)  # Current communion

        # Self-protective parameters
        self.security = self.alpha / (self.alpha + self.beta)
        self.self_esteem = self.alpha + self.beta  # Sample size as confidence

        # Affect system
        self.valence = 0.0  # -1 to 1
        self.arousal = 0.0  # 0 to 1
        self.affect_decay = 0.1  # How quickly affect returns to baseline

    def generate_behavior(self):
        """Generate behavior based on current warmth goal + some noise"""
        # Higher warmth goal = more likely to be warm
        warmth_probability = self.warmth_goal + np.random.normal(0, 0.1)
        warmth_probability = np.clip(warmth_probability, 0, 1)

        behavior = 1 if np.random.random() < warmth_probability else 0  # 1=warm, 0=cold
        self.history["behavior"].append(behavior)
        return behavior

    def receive_behavior(self, other_behavior):
        """Process received behavior and update internal state"""
        self.history["received_behavior"].append(other_behavior)

        # Calculate prediction error / arousal
        expected_warmth = self.alpha / (self.alpha + self.beta)
        observed_behavior = other_behavior

        # Arousal = surprise weighted by conviction (sample_size)
        surprise = abs(observed_behavior - expected_warmth)
        self.arousal = surprise * np.sqrt(
            self.sample_size
        )  # More conviction = more arousal when wrong

        # Valence = positive if behavior matches our goal
        behavior_matches_goal = (
            abs(other_behavior - self.warmth_goal) < 0.3
        )  # tolerance
        self.valence = 1.0 if behavior_matches_goal else -1.0

        # Update Bayesian priors
        if other_behavior == 1:  # received warmth
            self.alpha += 1
        else:  # received coldness
            self.beta += 1

        # Update derived parameters
        old_sample_size = self.sample_size
        self.sample_size = self.alpha + self.beta
        self.security_ratio = self.alpha / (self.alpha + self.beta)

        # Goal updating based on affect
        if self.valence < 0:  # negative valence
            # Low self-esteem (small sample size) = more likely to change goals
            change_probability = 1 / (1 + old_sample_size * 0.1)

            if np.random.random() < change_probability:
                # Move goal toward observed behavior
                adaptation_rate = 0.1 + (
                    self.arousal * 0.05
                )  # higher arousal = faster adaptation
                self.warmth_goal = (
                    1 - adaptation_rate
                ) * self.warmth_goal + adaptation_rate * other_behavior

        # High arousal increases conviction (sample_size effect)
        if self.arousal > 1.0:
            conviction_boost = self.arousal * 0.5
            self.sample_size += conviction_boost

            # When arousal is high, become more tolerant of deviations (lower ratio requirements)
            if self.arousal > 1.5:
                tolerance_increase = (self.arousal - 1.5) * 0.1
                # This makes the agent more flexible in what they'll accept
                self.security_threshold += tolerance_increase * 0.1

        # Store history
        self.history["alpha"].append(self.alpha)
        self.history["beta"].append(self.beta)
        self.history["sample_size"].append(self.sample_size)
        self.history["security_ratio"].append(self.security_ratio)
        self.history["warmth_goal"].append(self.warmth_goal)
        self.history["arousal"].append(self.arousal)
        self.history["valence"].append(self.valence)

    def update_goals_from_affect(self):
        """Update goals based on current affect state"""
        # Negative valence increases goal adaptability
        if self.valence < 0:
            adaptation_rate = abs(self.valence) / (1.0 + self.self_esteem)

            # Update state goals
            self.agency_state.value += np.random.normal(0, adaptation_rate)
            self.communion_state.value += np.random.normal(0, adaptation_rate)

            # Clip values to valid range
            self.agency_state.value = np.clip(self.agency_state.value, -1, 1)
            self.communion_state.value = np.clip(self.communion_state.value, -1, 1)

        # High arousal increases goal strength but reduces adaptability
        if self.arousal > 0.5:
            strength_boost = (self.arousal - 0.5) * 0.2
            self.agency_state.strength += strength_boost
            self.communion_state.strength += strength_boost

            # High arousal makes goals more rigid
            self.agency_state.adaptability *= (1 - self.arousal * 0.1)
            self.communion_state.adaptability *= (1 - self.arousal * 0.1)

    def update_affect_from_goals(self, other_behavior):
        """Update affect based on goal achievement"""
        # Calculate goal-behavior match
        agency_match = 1 - abs(self.agency_state.value - other_behavior[0])
        communion_match = 1 - abs(self.communion_state.value - other_behavior[1])

        # Update valence based on goal achievement
        goal_achievement = (agency_match + communion_match) / 2
        target_valence = (goal_achievement - 0.5) * 2  # Scale to -1 to 1

        # More confident goals have stronger impact on valence
        self.valence += (target_valence - self.valence) * (
            self.agency_state.confidence + self.communion_state.confidence
        ) / 2

        # Update arousal based on goal difficulty
        expected_success = (self.security + self.self_esteem / 20) / 2
        goal_difficulty = 1 - expected_success

        # Higher difficulty and importance = more arousal
        self.arousal = goal_difficulty * (
            self.agency_state.strength + self.communion_state.strength
        ) / 2

        # Apply affect decay
        self.valence *= (1 - self.affect_decay)
        self.arousal *= (1 - self.affect_decay)

    def get_current_state(self):
        return {
            "name": self.name,
            "alpha": self.alpha,
            "beta": self.beta,
            "sample_size": self.sample_size,
            "security_ratio": self.security_ratio,
            "warmth_goal": self.warmth_goal,
            "arousal": self.arousal,
            "valence": self.valence,
        }


class SelfSystem:
    def __init__(self):
        self.security = 0.5  # Ratio of warm interactions comfortable with
        self.confidence = 10.0  # Sample size / prior strength
        self.history = []

    def update_from_affect(self, valence: float, arousal: float):
        """Update self-concept based on affect"""
        if valence < 0:
            # More likely to change goals when negative
            change_prob = 1.0 / (1.0 + self.confidence)
            if np.random.random() < change_prob:
                self.security += -0.1 * valence

        # Arousal influences goal strength
        self.confidence += arousal * 0.1


def run_simulation(agent1, agent2, n_interactions=100):
    """Run interaction simulation between two agents"""

    for i in range(n_interactions):
        # Both agents generate behavior
        behavior1 = agent1.generate_behavior()
        behavior2 = agent2.generate_behavior()

        # Both agents receive the other's behavior
        agent1.receive_behavior(behavior2)
        agent2.receive_behavior(behavior1)

        # Print occasional updates
        if i % 20 == 0:
            print(f"Interaction {i}:")
            print(
                f"  {agent1.name}: goal={agent1.warmth_goal:.3f}, arousal={agent1.arousal:.3f}, valence={agent1.valence:.3f}"
            )
            print(
                f"  {agent2.name}: goal={agent2.warmth_goal:.3f}, arousal={agent2.arousal:.3f}, valence={agent2.valence:.3f}"
            )
            print()


def plot_simulation_results(agent1, agent2):
    """Plot the simulation results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    interactions = range(len(agent1.history["warmth_goal"]))

    # Plot 1: Warmth Goals over time
    axes[0, 0].plot(
        interactions,
        agent1.history["warmth_goal"],
        label=f"{agent1.name} Goal",
        linewidth=2,
    )
    axes[0, 0].plot(
        interactions,
        agent2.history["warmth_goal"],
        label=f"{agent2.name} Goal",
        linewidth=2,
    )
    axes[0, 0].set_title("Warmth Goals Over Time")
    axes[0, 0].set_ylabel("Warmth Goal (0-1)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Sample Size (Self-Esteem/Conviction)
    axes[0, 1].plot(
        interactions,
        agent1.history["sample_size"],
        label=f"{agent1.name} Conviction",
        linewidth=2,
    )
    axes[0, 1].plot(
        interactions,
        agent2.history["sample_size"],
        label=f"{agent2.name} Conviction",
        linewidth=2,
    )
    axes[0, 1].set_title("Sample Size (Conviction/Self-Esteem)")
    axes[0, 1].set_ylabel("Sample Size")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Security Ratio
    axes[0, 2].plot(
        interactions,
        agent1.history["security_ratio"],
        label=f"{agent1.name} Security",
        linewidth=2,
    )
    axes[0, 2].plot(
        interactions,
        agent2.history["security_ratio"],
        label=f"{agent2.name} Security",
        linewidth=2,
    )
    axes[0, 2].set_title("Security Ratio (α/(α+β))")
    axes[0, 2].set_ylabel("Security Ratio")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Arousal
    axes[1, 0].plot(
        interactions[1:],
        agent1.history["arousal"][1:],
        label=f"{agent1.name} Arousal",
        linewidth=2,
    )
    axes[1, 0].plot(
        interactions[1:],
        agent2.history["arousal"][1:],
        label=f"{agent2.name} Arousal",
        linewidth=2,
    )
    axes[1, 0].set_title("Arousal Over Time")
    axes[1, 0].set_ylabel("Arousal")
    axes[1, 0].set_xlabel("Interactions")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Valence
    axes[1, 1].plot(
        interactions[1:],
        agent1.history["valence"][1:],
        label=f"{agent1.name} Valence",
        linewidth=2,
    )
    axes[1, 1].plot(
        interactions[1:],
        agent2.history["valence"][1:],
        label=f"{agent2.name} Valence",
        linewidth=2,
    )
    axes[1, 1].set_title("Valence Over Time")
    axes[1, 1].set_ylabel("Valence")
    axes[1, 1].set_xlabel("Interactions")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Behavior patterns
    if len(agent1.history["behavior"]) > 0:
        # Moving average of behaviors
        window = 10
        agent1_behavior_avg = np.convolve(
            agent1.history["behavior"], np.ones(window) / window, mode="valid"
        )
        agent2_behavior_avg = np.convolve(
            agent2.history["behavior"], np.ones(window) / window, mode="valid"
        )

        axes[1, 2].plot(
            range(len(agent1_behavior_avg)),
            agent1_behavior_avg,
            label=f"{agent1.name} Behavior",
            linewidth=2,
        )
        axes[1, 2].plot(
            range(len(agent2_behavior_avg)),
            agent2_behavior_avg,
            label=f"{agent2.name} Behavior",
            linewidth=2,
        )
        axes[1, 2].set_title("Actual Behavior (Moving Average)")
        axes[1, 2].set_ylabel("Warmth (0=Cold, 1=Warm)")
        axes[1, 2].set_xlabel("Interactions")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("=== Interpersonal Bayesian Simulation ===\n")

    # Create agents with different starting conditions
    print("Creating agents...")

    # Agent 1: Secure, high self-esteem (large sample size)
    agent1 = InterpersonalAgent("Secure Sam", initial_alpha=15, initial_beta=5)

    # Agent 2: Insecure, low self-esteem (small sample size)
    agent2 = InterpersonalAgent("Anxious Anna", initial_alpha=3, initial_beta=7)

    print(f"Initial states:")
    print(
        f"  {agent1.name}: goal={agent1.warmth_goal:.3f}, sample_size={agent1.sample_size}"
    )
    print(
        f"  {agent2.name}: goal={agent2.warmth_goal:.3f}, sample_size={agent2.sample_size}"
    )
    print()

    # Run simulation
    print("Running simulation...")
    run_simulation(agent1, agent2, n_interactions=100)

    # Plot results
    print("Plotting results...")
    plot_simulation_results(agent1, agent2)

    # Final states
    print("Final states:")
    print(
        f"  {agent1.name}: goal={agent1.warmth_goal:.3f}, conviction={agent1.sample_size:.1f}"
    )
    print(
        f"  {agent2.name}: goal={agent2.warmth_goal:.3f}, conviction={agent2.sample_size:.1f}"
    )
