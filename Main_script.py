# import matplotlib
# pip install matplotlib
# Import required packages
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import nashpy as nash
from Configurations import (
    QLearningConfig,
    config1,
    config2,
    N_ROUNDS,
    A1,
    A2,
)
from agent_thinking.Q_learning import QLearningAgent, GameEnvironment

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


def main():
    try:
        # Create game and find all equilibria
        game = nash.Game(A1, A2)
        equilibria = list(game.support_enumeration())

        if not equilibria:
            print("Error: No Nash equilibria found")
            return

        # Display all equilibria
        print("\n=== Nash Equilibrium Analysis ===")
        for i, (p, q) in enumerate(equilibria, 1):
            print(f"\nEquilibrium {i}:")
            print(f"Player 1 strategy (Up, Down): ({p[0]:.4f}, {p[1]:.4f})")
            print(f"Player 2 strategy (Left, Right): ({q[0]:.4f}, {q[1]:.4f})")

            # Identify if pure or mixed
            is_pure = all(x in [0.0, 1.0] for x in np.concatenate([p, q]))
            print(f"Type: {'Pure' if is_pure else 'Mixed'} Strategy")

        print("\n" + "=" * 30 + "\n")

        # Use first equilibrium for simulation
        p_init, q_init = equilibria[0]

        # Initialize Q-learning simulation with payoff matrices
        agent1 = QLearningAgent(config1, is_player1=True, payoff_matrices=(A1, A2))
        agent2 = QLearningAgent(config2, is_player1=False, payoff_matrices=(A1, A2))
        game = GameEnvironment(A1, A2, agent1, agent2)

        # Store initial joint probabilities
        initial_probs = {
            "UL": p_init[0] * q_init[0],
            "UR": p_init[0] * q_init[1],
            "DL": p_init[1] * q_init[0],
            "DR": p_init[1] * q_init[1],
        }

        # Initialize history tracking
        actions1 = ["Up" if np.random.random() < p_init[0] else "Down"]
        actions2 = ["Left" if np.random.random() < q_init[0] else "Right"]
        idx1, idx2 = (
            0 if actions1[0] == "Up" else 1,
            0 if actions2[0] == "Left" else 1,
        )
        payoffs1 = [A1[idx1, idx2]]
        payoffs2 = [A2[idx1, idx2]]

        print("=== Starting Q-Learning Simulation ===")
        print(f"Running {N_ROUNDS} rounds...\n")

        # Run simulation
        for round_num in range(1, N_ROUNDS):
            action1, action2, payoff1, payoff2 = game.step()
            actions1.append(action1)
            actions2.append(action2)
            payoffs1.append(payoff1)
            payoffs2.append(payoff2)

            if round_num % 200 == 0:
                print(
                    f"Round {round_num}: Actions({action1}, {action2}), Payoffs({payoff1}, {payoff2})"
                )

        # Display results
        print("\n=== Simulation Complete ===")
        print(f"Average Payoff Player 1: {np.mean(payoffs1):.4f}")
        print(f"Average Payoff Player 2: {np.mean(payoffs2):.4f}")

        # Generate visualizations
        from visualization.game_plots import GameVisualizer

        visualizer = GameVisualizer(N_ROUNDS)
        visualizer.create_plots(actions1, actions2, payoffs1, payoffs2, initial_probs)

    except ImportError:
        print(
            "Error: Required packages not installed. Please run 'pip install nashpy matplotlib'"
        )
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
