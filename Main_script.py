# import matplotlib
# pip install matplotlib
# Import required packages
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now import your local modules
from Nash.nash_calculator import solve_2x2_nash
from agent_thinking.Q_learning import (
    QLearningConfig,
    QLearningAgent,
    GameEnvironment,
)


def main():
    # Example usage
    A1 = np.array([[3, 1], [0, 2]])
    A2 = np.array([[2, 1], [0, 3]])

    # First: Get and display Nash equilibrium
    print("\n=== Nash Equilibrium Analysis ===")
    nash_result = solve_2x2_nash(A1, A2)

    # Get initial strategies
    p_init = nash_result["mixed_strategy"]["p_star"]
    q_init = nash_result["mixed_strategy"]["q_star"]

    print(f"\nNash Equilibrium Strategies:")
    print(f"Player 1 (Up probability): {p_init:.4f}")
    print(f"Player 2 (Left probability): {q_init:.4f}")
    print("\n" + "=" * 30 + "\n")

    # Second: Configure and run Q-learning simulation
    print("=== Starting Q-Learning Simulation ===")

    # Configure agents
    config1 = QLearningConfig(
        alpha=0.1,
        beta=1.0,
        gamma=0.88,
        rho=0.88,
        lambda_val=2.25,
        ema_weight=0.1,
        novelty_weight=0.1,
        novelty_decay=0.5,
        prior_weight=0.3,
        rand_explore=0.1,
    )
    config2 = QLearningConfig(
        alpha=0.1,
        beta=1.0,
        gamma=0.88,
        rho=0.88,
        lambda_val=2.25,
        ema_weight=0.1,
        novelty_weight=0.1,
        novelty_decay=0.5,
        prior_weight=0.3,
        rand_explore=0.1,
    )

    # Create agents with Nash equilibrium as initial strategy
    agent1 = QLearningAgent(config1, p_init)
    agent2 = QLearningAgent(config2, q_init)

    # Create game environment
    game = GameEnvironment(A1, A2, agent1, agent2)

    # Run simulation and collect data
    n_rounds = 1000
    actions1, actions2 = [], []
    payoffs1, payoffs2 = [], []

    print(f"\nRunning {n_rounds} rounds of interaction...")

    for round_num in range(n_rounds):
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

    # Create and show visualization
    print("\n=== Generating Visualizations ===")
    from visualization.game_plots import GameVisualizer

    visualizer = GameVisualizer(n_rounds)
    visualizer.create_plots(actions1, actions2, payoffs1, payoffs2)


if __name__ == "__main__":
    main()
