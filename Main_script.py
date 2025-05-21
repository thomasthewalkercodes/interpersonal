# import matplotlib
# pip install matplotlib
# Import required packages
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from Configurations import (
    QLearningConfig,
    config1,
    config2,
    N_ROUNDS,
    A1,
    A2,
)  # Updated import
from agent_thinking.Q_learning import QLearningAgent, GameEnvironment
from agent_thinking.prior_handling import (
    PriorConfig,
)
from Nash.nash_calculator import solve_2x2_nash

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


def main():
    # First: Get and display Nash equilibrium
    print("\n=== Nash Equilibrium Analysis ===")
    nash_result = solve_2x2_nash(A1, A2)

    # Add error checking for Nash equilibrium calculation
    if nash_result is None or "mixed_strategy" not in nash_result:
        print("Error: Failed to calculate Nash equilibrium")
        return

    # Get initial strategies with error checking
    p_init = nash_result.get("mixed_strategy", {}).get("p_star")
    q_init = nash_result.get("mixed_strategy", {}).get("q_star")

    if p_init is None or q_init is None:
        print("Error: Invalid Nash equilibrium probabilities")
        return

    print("\nNash Equilibrium Strategies:")
    print(f"Player 1 (Up probability): {p_init:.4f}")
    print(f"Player 2 (Left probability): {q_init:.4f}")
    print("\n" + "=" * 30 + "\n")

    # Second: Configure and run Q-learning simulation
    print("=== Starting Q-Learning Simulation ===")

    # Create agents with Nash equilibrium as initial strategy
    agent1 = QLearningAgent(config1, is_player1=True, initial_strategy=p_init)
    agent2 = QLearningAgent(config2, is_player1=False, initial_strategy=q_init)

    # Create game environment
    game = GameEnvironment(A1, A2, agent1, agent2)

    # Calculate and display initial joint probabilities
    print("\n=== Initial Nash Equilibrium Joint Probabilities ===")
    prob_UL = p_init * q_init  # Both play first action (Up, Left)
    prob_UR = p_init * (1 - q_init)  # P1 plays Up, P2 plays Right
    prob_DL = (1 - p_init) * q_init  # P1 plays Down, P2 plays Left
    prob_DR = (1 - p_init) * (1 - q_init)  # Both play second action (Down, Right)

    # Store initial probabilities immediately in the correct format
    initial_probs = {"UL": prob_UL, "UR": prob_UR, "DL": prob_DL, "DR": prob_DR}

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
    print(f"\nRunning {N_ROUNDS} rounds of interaction...")

    for round_num in range(1, N_ROUNDS):
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

    visualizer = GameVisualizer(N_ROUNDS)
    visualizer.create_plots(actions1, actions2, payoffs1, payoffs2, initial_probs)


if __name__ == "__main__":
    main()
