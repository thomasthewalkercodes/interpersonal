# import matplotlib
# pip install matplotlib
# pip install seaborn pandas
# Import required packages
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import nashpy as nash
from Configurations import (
    QLearningConfig,
    GameConfig,
    test_config,
    N_ROUNDS,
    TestConfiguration,  # Add this import
    A1,
    A2,
)
from agent_thinking.Q_learning import (
    QLearningAgent,
    ContinuousQLearningAgent,
    GameEnvironment,
)
from visualization.game_plots import GameVisualizer
from visualization.map_plots import CircularGameVisualizer  # Add this import
import pandas as pd

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


def calculate_joint_probs(actions1, actions2):
    """Calculate joint probabilities for the last 100 rounds"""
    last_100_1 = actions1[-100:]
    last_100_2 = actions2[-100:]

    up = sum(1 for a in last_100_1 if a == "Up") / 100
    left = sum(1 for a in last_100_2 if a == "Left") / 100

    return {
        "UL": up * left,
        "UR": up * (1 - left),
        "DL": (1 - up) * left,
        "DR": (1 - up) * (1 - left),
    }


def run_simulation_batch(test_config: TestConfiguration):
    """Run multiple simulations with different configurations"""
    results = []

    # Get configurations as list of dictionaries
    configs = test_config.generate_configs()

    for config in configs:
        for _ in range(test_config.n_repetitions):
            # Create QLearningConfig objects with the current configuration
            q_config = QLearningConfig(
                alpha=config["alpha"],
                beta=config["beta"],
                gamma=config["gamma"],
                rho=config["rho"],
                lambda_val=config["lambda_val"],
                ema_weight=config["ema_weight"],
                prior_weight=config["prior_weight"],
            )

            # Initialize agents with proper configuration
            agent1 = QLearningAgent(q_config, is_player1=True, payoff_matrices=(A1, A2))
            agent2 = QLearningAgent(
                q_config, is_player1=False, payoff_matrices=(A1, A2)
            )

            # Run simulation
            game = GameEnvironment(A1, A2, agent1, agent2)
            actions1, actions2 = [], []
            payoffs1, payoffs2 = [], []

            for _ in range(test_config.n_rounds):
                action1, action2, payoff1, payoff2 = game.step()
                actions1.append(action1)
                actions2.append(action2)
                payoffs1.append(payoff1)
                payoffs2.append(payoff2)

            # Create result object
            result = SimulationResult(
                config=config,
                payoffs1=payoffs1,
                payoffs2=payoffs2,
                joint_probs=calculate_joint_probs(actions1, actions2),
                final_strategies=(
                    sum(1 for a in actions1[-100:] if a == "Up") / 100,
                    sum(1 for a in actions2[-100:] if a == "Left") / 100,
                ),
            )
            results.append(result)

    return results


def main():
    try:
        game_config = GameConfig()  # Create new game config instance

        # Convert base_config dictionary to QLearningConfig object
        q_config = QLearningConfig(
            alpha=test_config.base_config["alpha"],
            beta=test_config.base_config["beta"],
            gamma=test_config.base_config["gamma"],
            rho=test_config.base_config["rho"],
            lambda_val=test_config.base_config["lambda_val"],
            ema_weight=test_config.base_config["ema_weight"],
            prior_weight=test_config.base_config["prior_weight"],
        )

        if game_config.game_type == "matrix":
            # Create matrix game agents
            agent1 = QLearningAgent(q_config, True, (A1, A2))
            agent2 = QLearningAgent(q_config, False, (A1, A2))
        else:
            # Create continuous game agents
            agent1 = ContinuousQLearningAgent(q_config, True)
            agent2 = ContinuousQLearningAgent(q_config, False)

        # Initialize game environment
        game = GameEnvironment(game_config, agent1, agent2)

        # Initialize history tracking
        actions1, actions2 = [], []
        payoffs1, payoffs2 = [], []

        print("=== Starting Simulation ===")
        print(f"Game Type: {game_config.game_type}")
        print(f"Running {N_ROUNDS} rounds...\n")

        # Run simulation
        for round_num in range(N_ROUNDS):
            action1, action2, payoff1, payoff2 = game.step()
            actions1.append(action1)
            actions2.append(action2)
            payoffs1.append(payoff1)
            payoffs2.append(payoff2)

            if round_num % 200 == 0:
                print(f"Round {round_num}: Payoffs({payoff1:.2f}, {payoff2:.2f})")

        # Display results
        print("\n=== Simulation Complete ===")
        print(f"Average Payoff Player 1: {np.mean(payoffs1):.4f}")
        print(f"Average Payoff Player 2: {np.mean(payoffs2):.4f}")

        # Create visualization
        if game_config.game_type == "circular":
            visualizer = CircularGameVisualizer(N_ROUNDS)
            visualizer.create_analysis_plots(actions1, actions2, payoffs1, payoffs2)
        else:
            visualizer = GameVisualizer(N_ROUNDS)
            visualizer.create_plots(
                actions1, actions2, payoffs1, payoffs2, game_config.game_type
            )

    except ImportError:
        print("Error: Required packages not installed.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
