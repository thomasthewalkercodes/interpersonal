import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


class SimulationPlotter:
    def __init__(self):
        self.colors = {"agent1": "#2E86AB", "agent2": "#A23B72"}

    def create_all_plots(self, results, output_dir):
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Creating plots in: {plots_dir}")

        try:
            self.plot_rewards(results, plots_dir)
            print("Plots created successfully!")
        except Exception as e:
            print(f"Error creating plots: {e}")
            import traceback

            traceback.print_exc()

    def plot_rewards(self, results, output_dir):
        training_results = results.get("training_results", {})
        episode_rewards = training_results.get("episode_rewards", {})

        agent1_rewards = episode_rewards.get("agent1", [])
        agent2_rewards = episode_rewards.get("agent2", [])

        if not agent1_rewards or not agent2_rewards:
            print("No reward data to plot")
            return

        plt.figure(figsize=(12, 6))
        episodes = range(len(agent1_rewards))

        plt.plot(
            episodes,
            agent1_rewards,
            color=self.colors["agent1"],
            alpha=0.6,
            label="Agent 1",
        )
        plt.plot(
            episodes,
            agent2_rewards,
            color=self.colors["agent2"],
            alpha=0.6,
            label="Agent 2",
        )

        plt.title("Episode Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(
            os.path.join(output_dir, "episode_rewards.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print("Created reward plot")
