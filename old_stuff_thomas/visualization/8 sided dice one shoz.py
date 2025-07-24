import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict, deque
import seaborn as sns


class QLearningAgent:
    def __init__(
        self,
        agent_id,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3,
        epsilon_decay=0.995,
    ):
        """
        Q-Learning agent for dice prediction

        Parameters:
        agent_id: identifier for the agent
        learning_rate: alpha for Q-learning updates
        discount_factor: gamma for future reward discounting
        epsilon: exploration rate for epsilon-greedy policy
        epsilon_decay: decay rate for epsilon over time
        """
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.05

        # Q-table: state -> action -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))

        # History tracking
        self.state_history = deque(maxlen=3)  # Last 3 dice outcomes
        self.prediction_history = []
        self.score_history = []
        self.other_agent_predictions = []  # Track other agent's predictions

        # Model of other agent: what number will they choose given recent history
        self.other_agent_model = defaultdict(
            lambda: defaultdict(int)
        )  # state -> predicted_number -> count
        self.other_agent_prediction_probs = (
            {}
        )  # state -> probability distribution over 1-8

    def get_state(self):
        """Convert recent dice history to state string"""
        if len(self.state_history) < 3:
            return "start"
        return "_".join(map(str, self.state_history))

    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(1, 8)
        else:
            # Exploit: best known action
            q_values = self.q_table[state]
            if not q_values:
                return random.randint(1, 8)
            return max(q_values.items(), key=lambda x: x[1])[0]

    def update_q_table(self, state, action, reward, next_state):
        """Q-learning update rule"""
        current_q = self.q_table[state][action]

        # Find max Q-value for next state
        next_q_values = self.q_table[next_state]
        max_next_q = max(next_q_values.values()) if next_q_values else 0

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q

    def update_other_agent_model(self, state, other_agent_prediction):
        """Update model of what the other agent predicts"""
        self.other_agent_model[state][other_agent_prediction] += 1

        # Update probability distribution
        total_observations = sum(self.other_agent_model[state].values())
        self.other_agent_prediction_probs[state] = {}
        for number in range(1, 9):
            count = self.other_agent_model[state][number]
            self.other_agent_prediction_probs[state][number] = (
                count / total_observations if total_observations > 0 else 1 / 8
            )

    def predict_other_agent(self, state):
        """Predict what the other agent will choose"""
        if state not in self.other_agent_prediction_probs:
            return random.randint(1, 8)  # No data, random guess

        probs = self.other_agent_prediction_probs[state]
        numbers = list(range(1, 9))
        probabilities = [probs.get(num, 0) for num in numbers]

        # If all probabilities are 0, use uniform
        if sum(probabilities) == 0:
            return random.randint(1, 8)

        # Weighted random choice based on learned probabilities
        return np.random.choice(numbers, p=np.array(probabilities) / sum(probabilities))

    def update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


class DiceGame:
    def __init__(self, dice_bias=None):
        """
        Initialize 8-sided dice game
        dice_bias: list of 8 probabilities (must sum to 1), or None for fair dice
        """
        if dice_bias is None:
            self.dice_probabilities = [1 / 8] * 8  # Fair dice
        else:
            assert len(dice_bias) == 8 and abs(sum(dice_bias) - 1.0) < 1e-10
            self.dice_probabilities = dice_bias

        self.roll_history = []

    def roll_dice(self):
        """Roll the 8-sided dice"""
        result = np.random.choice(range(1, 9), p=self.dice_probabilities)
        self.roll_history.append(result)
        return result

    def calculate_score(self, prediction, actual):
        """Calculate score based on distance from actual result"""
        distance = abs(prediction - actual)
        max_score = 10
        return max(0, max_score - 2 * distance)  # 10, 8, 6, 4, 2, 0, 0, 0


class MultiAgentDiceGame:
    def __init__(self, num_rounds=1000, dice_bias=None):
        """
        Initialize multi-agent dice prediction game
        """
        self.num_rounds = num_rounds
        self.dice_game = DiceGame(dice_bias)

        # Create two agents
        self.agent1 = QLearningAgent("Agent1")
        self.agent2 = QLearningAgent("Agent2")
        self.agents = [self.agent1, self.agent2]

        # Game statistics
        self.round_results = []

    def run_game(self):
        """Run the complete multi-agent game"""
        print(f"Starting Multi-Agent Dice Game with {self.num_rounds} rounds")
        print(
            "Dice probabilities:",
            [f"{p:.3f}" for p in self.dice_game.dice_probabilities],
        )
        print("-" * 60)

        for round_num in range(self.num_rounds):
            # Roll the dice
            dice_result = self.dice_game.roll_dice()

            # Get current state for both agents
            state1 = self.agent1.get_state()
            state2 = self.agent2.get_state()

            # Agents make predictions
            prediction1 = self.agent1.choose_action(state1)
            prediction2 = self.agent2.choose_action(state2)

            # Calculate scores
            score1 = self.dice_game.calculate_score(prediction1, dice_result)
            score2 = self.dice_game.calculate_score(prediction2, dice_result)

            # Update agent histories
            self.agent1.state_history.append(dice_result)
            self.agent2.state_history.append(dice_result)
            self.agent1.prediction_history.append(prediction1)
            self.agent2.prediction_history.append(prediction2)
            self.agent1.score_history.append(score1)
            self.agent2.score_history.append(score2)

            # Agents observe each other's predictions and update their models
            self.agent1.other_agent_predictions.append(prediction2)
            self.agent2.other_agent_predictions.append(prediction1)
            self.agent1.update_other_agent_model(state1, prediction2)
            self.agent2.update_other_agent_model(state2, prediction1)

            # Q-learning updates
            next_state1 = self.agent1.get_state()
            next_state2 = self.agent2.get_state()

            self.agent1.update_q_table(state1, prediction1, score1, next_state1)
            self.agent2.update_q_table(state2, prediction2, score2, next_state2)

            # Update exploration rates
            self.agent1.update_epsilon()
            self.agent2.update_epsilon()

            # Store round results
            self.round_results.append(
                {
                    "round": round_num,
                    "dice_result": dice_result,
                    "agent1_prediction": prediction1,
                    "agent2_prediction": prediction2,
                    "agent1_score": score1,
                    "agent2_score": score2,
                }
            )

            # Print progress
            if (round_num + 1) % 200 == 0:
                avg_score1 = np.mean(self.agent1.score_history[-200:])
                avg_score2 = np.mean(self.agent2.score_history[-200:])
                print(
                    f"Round {round_num + 1:4d}: Agent1 avg score = {avg_score1:.2f}, "
                    f"Agent2 avg score = {avg_score2:.2f}, "
                    f"Epsilon = {self.agent1.epsilon:.3f}"
                )

        print("\nGame completed!")
        return self.agents

    def plot_results(self):
        """Plot comprehensive results of the game"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Score evolution over time
        window = 50
        rounds = range(len(self.agent1.score_history))

        score1_smooth = np.convolve(
            self.agent1.score_history, np.ones(window) / window, mode="valid"
        )
        score2_smooth = np.convolve(
            self.agent2.score_history, np.ones(window) / window, mode="valid"
        )

        axes[0, 0].plot(
            range(window - 1, len(self.agent1.score_history)),
            score1_smooth,
            "b-",
            label="Agent 1",
            linewidth=2,
        )
        axes[0, 0].plot(
            range(window - 1, len(self.agent2.score_history)),
            score2_smooth,
            "r-",
            label="Agent 2",
            linewidth=2,
        )
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].set_ylabel("Average Score")
        axes[0, 0].set_title("Learning Progress: Moving Average Score")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Prediction distribution
        pred_counts1 = np.bincount(self.agent1.prediction_history, minlength=9)[1:9]
        pred_counts2 = np.bincount(self.agent2.prediction_history, minlength=9)[1:9]
        dice_counts = np.bincount(self.dice_game.roll_history, minlength=9)[1:9]

        x = np.arange(1, 9)
        width = 0.25

        axes[0, 1].bar(
            x - width,
            pred_counts1 / len(self.agent1.prediction_history),
            width,
            label="Agent 1 Predictions",
            alpha=0.8,
        )
        axes[0, 1].bar(
            x,
            pred_counts2 / len(self.agent2.prediction_history),
            width,
            label="Agent 2 Predictions",
            alpha=0.8,
        )
        axes[0, 1].bar(
            x + width,
            dice_counts / len(self.dice_game.roll_history),
            width,
            label="Actual Dice Results",
            alpha=0.8,
        )

        axes[0, 1].set_xlabel("Dice Number")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Prediction vs Reality Distribution")
        axes[0, 1].legend()
        axes[0, 1].set_xticks(x)

        # Plot 3: Agent 1's model of Agent 2's behavior over time
        self.plot_agent_model_evolution(
            axes[1, 0], self.agent1, "Agent 1's Model of Agent 2"
        )

        # Plot 4: Agent 2's model of Agent 1's behavior over time
        self.plot_agent_model_evolution(
            axes[1, 1], self.agent2, "Agent 2's Model of Agent 1"
        )

        plt.tight_layout()
        plt.show()

        # Print final statistics
        self.print_final_stats()

    def plot_agent_model_evolution(self, ax, agent, title):
        """Plot how an agent's model of the other agent evolves over time"""
        # Sample key states and show how predictions evolved
        if not agent.other_agent_prediction_probs:
            ax.text(
                0.5,
                0.5,
                "No model data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            return

        # Get the most common states
        state_counts = defaultdict(int)
        for state in agent.other_agent_model.keys():
            state_counts[state] = sum(agent.other_agent_model[state].values())

        # Select top 3 most observed states
        top_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        if not top_states:
            ax.text(
                0.5,
                0.5,
                "No model data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            return

        # For each top state, show the probability distribution
        colors = ["blue", "red", "green"]
        x = np.arange(1, 9)
        width = 0.25

        for i, (state, count) in enumerate(top_states):
            if state in agent.other_agent_prediction_probs:
                probs = [
                    agent.other_agent_prediction_probs[state].get(num, 0)
                    for num in range(1, 9)
                ]
                ax.bar(
                    x + i * width - width,
                    probs,
                    width,
                    label=f"State: {state} (n={count})",
                    color=colors[i],
                    alpha=0.7,
                )

        ax.set_xlabel("Predicted Number Choice")
        ax.set_ylabel("Probability")
        ax.set_title(title)
        ax.legend()
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)

    def print_final_stats(self):
        """Print final game statistics"""
        total_score1 = sum(self.agent1.score_history)
        total_score2 = sum(self.agent2.score_history)
        avg_score1 = np.mean(self.agent1.score_history)
        avg_score2 = np.mean(self.agent2.score_history)

        print("\n" + "=" * 60)
        print("FINAL GAME STATISTICS")
        print("=" * 60)
        print(f"Agent 1 - Total Score: {total_score1}, Average: {avg_score1:.3f}")
        print(f"Agent 2 - Total Score: {total_score2}, Average: {avg_score2:.3f}")
        print(
            f"Winner: {'Agent 1' if total_score1 > total_score2 else 'Agent 2' if total_score2 > total_score1 else 'Tie'}"
        )

        # Show learning effectiveness
        early_avg1 = np.mean(self.agent1.score_history[:100])
        late_avg1 = np.mean(self.agent1.score_history[-100:])
        early_avg2 = np.mean(self.agent2.score_history[:100])
        late_avg2 = np.mean(self.agent2.score_history[-100:])

        print(f"\nLearning Progress:")
        print(
            f"Agent 1: Early average = {early_avg1:.3f}, Late average = {late_avg1:.3f}, Improvement = {late_avg1-early_avg1:.3f}"
        )
        print(
            f"Agent 2: Early average = {early_avg2:.3f}, Late average = {late_avg2:.3f}, Improvement = {late_avg2-early_avg2:.3f}"
        )


# Example usage
def run_experiment(num_rounds=1000, dice_bias=None):
    """
    Run a complete experiment

    Parameters:
    num_rounds: Number of rounds to play
    dice_bias: Bias for the dice (None = fair), e.g., [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2] for biased toward 1 and 8
    """
    game = MultiAgentDiceGame(num_rounds=num_rounds, dice_bias=dice_bias)
    agents = game.run_game()
    game.plot_results()
    return game, agents


if __name__ == "__main__":
    print("Multi-Agent Q-Learning Dice Prediction Game")
    print("=" * 50)

    # Experiment 1: Fair dice
    print("\nExperiment 1: Fair 8-sided dice")
    game1, agents1 = run_experiment(num_rounds=1000)

    # Experiment 2: Biased dice (favors 1 and 8)
    print("\nExperiment 2: Biased dice (favors numbers 1 and 8)")
    biased_probs = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
    game2, agents2 = run_experiment(num_rounds=1000, dice_bias=biased_probs)
