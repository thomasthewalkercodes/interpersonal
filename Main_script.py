# import matplotlib
# pip install matplotlib
# Import required packages
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from Configurations import QLearningConfig  # Updated import
from agent_thinking.Q_learning import QLearningAgent, GameEnvironment
from agent_thinking.prior_handling import (
    PriorConfig,
)  # Import PriorConfig here if needed

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now import your local modules
from Nash.nash_calculator import solve_2x2_nash
from agent_thinking.Q_learning import (
    QLearningAgent,
    GameEnvironment,
)
from agent_thinking.prior_handling import PriorConfig


def main():
    # Example usage
    A1 = np.array([[1, 0], [0, 1]])
    A2 = np.array([[0, 1], [1, 0]])

    # First: Get and display Nash equilibrium
    print("\n=== Nash Equilibrium Analysis ===")
    nash_result = solve_2x2_nash(A1, A2)

    # Get initial strategies
    p_init = nash_result["mixed_strategy"]["p_star"]
    q_init = nash_result["mixed_strategy"]["q_star"]

    print("\nNash Equilibrium Strategies:")
    print(f"Player 1 (Up probability): {p_init:.4f}")
    print(f"Player 2 (Left probability): {q_init:.4f}")
    print("\n" + "=" * 30 + "\n")

    # Second: Configure and run Q-learning simulation
    print("=== Starting Q-Learning Simulation ===")

    # Configure agents
    config1 = QLearningConfig(
        alpha=0.1,
        beta=2.0,
        gamma=0.88,
        rho=0.88,
        lambda_val=2.25,
        ema_weight=0.1,
        prior_weight=0.2,
    )
    config2 = QLearningConfig(
        alpha=0.1,
        beta=2.0,
        gamma=0.88,
        rho=0.88,
        lambda_val=2.25,
        ema_weight=0.1,
        prior_weight=0.2,
    )

    # Create agents with Nash equilibrium as initial strategy
    agent1 = QLearningAgent(config1, is_player1=True, initial_strategy=p_init)
    agent2 = QLearningAgent(config2, is_player1=False, initial_strategy=q_init)

    # Create game environment
    game = GameEnvironment(A1, A2, agent1, agent2)

    # Calculate and display initial joint probabilities
    print("\n=== Initial Nash Equilibrium Joint Probabilities ===")
    prob_UL = p_init * q_init
    prob_UR = p_init * (1 - q_init)
    prob_DL = (1 - p_init) * q_init
    prob_DR = (1 - p_init) * (1 - q_init)

    print(f"Up-Left probability:    {prob_UL:.4f}")
    print(f"Up-Right probability:   {prob_UR:.4f}")
    print(f"Down-Left probability:  {prob_DL:.4f}")
    print(f"Down-Right probability: {prob_DR:.4f}")
    print("\n" + "=" * 30 + "\n")

    # Add initial probabilities to the action lists
    actions1 = ["Up" if np.random.random() < p_init else "Down"]
    actions2 = ["Left" if np.random.random() < q_init else "Right"]

    # Get initial payoffs based on Nash strategies
    idx1 = 0 if actions1[0] == "Up" else 1
    idx2 = 0 if actions2[0] == "Left" else 1
    payoffs1 = [A1[idx1, idx2]]
    payoffs2 = [A2[idx1, idx2]]

    print("=== Starting Q-Learning Simulation ===")

    # Run simulation and collect data
    n_rounds = 200

    print(f"\nRunning {n_rounds} rounds of interaction...")

    for round_num in range(1, n_rounds):
        action1, action2, payoff1, payoff2 = game.step()
        actions1.append(action1)
        actions2.append(action2)
        payoffs1.append(payoff1)
        payoffs2.append(payoff2)

        # Print progress every 200 rounds
        if round_num % 200 == 0:
            print(
                f"Round {round_num}: Actions({action1}, {action2}), Payoffs({payoff1}, {payoff2})"
            )

    print("\n=== Simulation Complete ===")
    print(f"Average Payoff Player 1: {np.mean(payoffs1):.4f}")
    print(f"Average Payoff Player 2: {np.mean(payoffs2):.4f}")
    # Calculate initial joint probabilities
    initial_probs = {
        "UL": p_init * q_init,
        "UR": p_init * (1 - q_init),
        "DL": (1 - p_init) * q_init,
        "DR": (1 - p_init) * (1 - q_init),
    }

    # Create and show visualization
    print("\n=== Generating Visualizations ===")
    from visualization.game_plots import GameVisualizer

    visualizer = GameVisualizer(n_rounds)
    visualizer.create_plots(actions1, actions2, payoffs1, payoffs2, initial_probs)


if __name__ == "__main__":
    main()
