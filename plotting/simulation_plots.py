"""
Simple plotting module for SAC simulations.
Basic version that works without complex dependencies.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


class SimulationPlotter:
    """Simple plotter for SAC simulation results."""

    def __init__(self):
        plt.style.use("default")
        self.colors = {"agent1": "#2E86AB", "agent2": "#A23B72"}

    def create_all_plots(self, results, output_dir):
        """Create all standard plots for a simulation run."""

        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        print(f"Creating plots in: {plots_dir}")

        try:
            # 1. Reward plots
            self.plot_rewards(results, plots_dir)

            # 2. Training progress (if available)
            self.plot_training_progress(results, plots_dir)

            # 3. Action patterns (if available)
            self.plot_actions(results, plots_dir)

            print("All plots created successfully!")

        except Exception as e:
            print(f"Error creating plots: {e}")
            import traceback

            traceback.print_exc()

    def plot_rewards(self, results, output_dir):
        """Plot episode rewards over time."""

        training_results = results.get("training_results", {})
        episode_rewards = training_results.get("episode_rewards", {})

        agent1_rewards = episode_rewards.get("agent1", [])
        agent2_rewards = episode_rewards.get("agent2", [])

        if not agent1_rewards or not agent2_rewards:
            print("No reward data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("Episode Rewards Analysis", fontsize=16, fontweight="bold")

        episodes = range(len(agent1_rewards))

        # Raw rewards
        ax1.plot(
            episodes,
            agent1_rewards,
            color=self.colors["agent1"],
            alpha=0.6,
            label="Agent 1",
        )
        ax1.plot(
            episodes,
            agent2_rewards,
            color=self.colors["agent2"],
            alpha=0.6,
            label="Agent 2",
        )
        ax1.set_title("Raw Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Smoothed rewards
        window = min(20, len(agent1_rewards) // 10)
        if window > 1:
            smooth1 = pd.Series(agent1_rewards).rolling(window=window).mean()
            smooth2 = pd.Series(agent2_rewards).rolling(window=window).mean()

            ax2.plot(
                episodes,
                smooth1,
                color=self.colors["agent1"],
                linewidth=2,
                label="Agent 1 (smoothed)",
            )
            ax2.plot(
                episodes,
                smooth2,
                color=self.colors["agent2"],
                linewidth=2,
                label="Agent 2 (smoothed)",
            )
            ax2.set_title(f"Smoothed Rewards (window={window})")
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Average Reward")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "episode_rewards.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print("Created reward plots")

    def plot_training_progress(self, results, output_dir):
        """Plot training metrics if available."""

        training_results = results.get("training_results", {})
        training_metrics = training_results.get("training_metrics", {})

        agent1_metrics = training_metrics.get("agent1", [])
        agent2_metrics = training_metrics.get("agent2", [])

        if not agent1_metrics or not agent2_metrics:
            print("No training metrics to plot")
            return

        # Convert to DataFrame for easier handling
        try:
            df1 = pd.DataFrame(agent1_metrics)
            df2 = pd.DataFrame(agent2_metrics)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Training Progress", fontsize=16, fontweight="bold")

            # Actor loss
            if "actor_loss" in df1.columns:
                axes[0, 0].plot(
                    df1["actor_loss"], color=self.colors["agent1"], label="Agent 1"
                )
                axes[0, 0].plot(
                    df2["actor_loss"], color=self.colors["agent2"], label="Agent 2"
                )
                axes[0, 0].set_title("Actor Loss")
                axes[0, 0].set_ylabel("Loss")
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # Critic loss
            if "critic_loss" in df1.columns:
                axes[0, 1].plot(
                    df1["critic_loss"], color=self.colors["agent1"], label="Agent 1"
                )
                axes[0, 1].plot(
                    df2["critic_loss"], color=self.colors["agent2"], label="Agent 2"
                )
                axes[0, 1].set_title("Critic Loss")
                axes[0, 1].set_ylabel("Loss")
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

            # Alpha (temperature)
            if "alpha" in df1.columns:
                axes[1, 0].plot(
                    df1["alpha"], color=self.colors["agent1"], label="Agent 1"
                )
                axes[1, 0].plot(
                    df2["alpha"], color=self.colors["agent2"], label="Agent 2"
                )
                axes[1, 0].set_title("Temperature (Alpha)")
                axes[1, 0].set_ylabel("Alpha")
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Alpha loss
            if "alpha_loss" in df1.columns:
                axes[1, 1].plot(
                    df1["alpha_loss"], color=self.colors["agent1"], label="Agent 1"
                )
                axes[1, 1].plot(
                    df2["alpha_loss"], color=self.colors["agent2"], label="Agent 2"
                )
                axes[1, 1].set_title("Alpha Loss")
                axes[1, 1].set_ylabel("Loss")
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "training_progress.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            print("Created training progress plots")

        except Exception as e:
            print(f"Error creating training plots: {e}")

    def plot_actions(self, results, output_dir):
        """Plot action patterns if available."""

        env_history = results.get("environment_history", {})

        if not env_history:
            print("No environment history to plot")
            return

        actions1 = env_history.get("agent1_actions", [])
        actions2 = env_history.get("agent2_actions", [])

        if not actions1 or not actions2:
            print("No action data to plot")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle("Action Patterns", fontsize=16, fontweight="bold")

        steps = range(len(actions1))

        # Time series of actions
        ax1.plot(
            steps,
            actions1,
            color=self.colors["agent1"],
            label="Agent 1 Warmth",
            linewidth=2,
        )
        ax1.plot(
            steps,
            actions2,
            color=self.colors["agent2"],
            label="Agent 2 Warmth",
            linewidth=2,
        )
        ax1.set_title("Warmth Levels Over Time")
        ax1.set_ylabel("Warmth Level")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Action histograms
        ax2.hist(
            actions1,
            bins=20,
            alpha=0.7,
            color=self.colors["agent1"],
            label="Agent 1",
            density=True,
        )
        ax2.hist(
            actions2,
            bins=20,
            alpha=0.7,
            color=self.colors["agent2"],
            label="Agent 2",
            density=True,
        )
        ax2.set_title("Warmth Distribution")
        ax2.set_xlabel("Warmth Level")
        ax2.set_ylabel("Density")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Correlation scatter
        ax3.scatter(actions1, actions2, alpha=0.6, color="purple")
        ax3.set_title("Action Correlation")
        ax3.set_xlabel("Agent 1 Warmth")
        ax3.set_ylabel("Agent 2 Warmth")
        ax3.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(actions1, actions2)[0, 1]
        ax3.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax3.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "action_patterns.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print("Created action pattern plots")

    def create_summary_dashboard(self, results, output_dir):
        """Create a summary dashboard."""

        config = results.get("config", {})
        final_eval = results.get("final_evaluation", {})

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f'Simulation Summary: {results.get("run_name", "Unknown")}',
            fontsize=16,
            fontweight="bold",
        )

        # Performance comparison
        if final_eval:
            agent1_perf = final_eval.get("agent1_avg_reward", 0)
            agent2_perf = final_eval.get("agent2_avg_reward", 0)

            bars = ax1.bar(
                ["Agent 1", "Agent 2"],
                [agent1_perf, agent2_perf],
                color=[self.colors["agent1"], self.colors["agent2"]],
            )
            ax1.set_title("Final Performance")
            ax1.set_ylabel("Average Reward")

            # Add value labels
            for bar, value in zip(bars, [agent1_perf, agent2_perf]):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

        # Configuration info
        ax2.text(
            0.1,
            0.9,
            f"Agent 1: {config.get('agent1_type', 'Unknown')}",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.text(
            0.1,
            0.8,
            f"Agent 2: {config.get('agent2_type', 'Unknown')}",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.text(
            0.1,
            0.7,
            f"Episodes: {config.get('episodes', 'Unknown')}",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.text(
            0.1,
            0.6,
            f"Payoff α: {config.get('payoff_alpha', 'Unknown')}",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.text(
            0.1,
            0.5,
            f"Payoff β: {config.get('payoff_beta', 'Unknown')}",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("Configuration")
        ax2.axis("off")

        # Simple learning curve
        episode_rewards = results.get("training_results", {}).get("episode_rewards", {})
        if episode_rewards:
            agent1_rewards = episode_rewards.get("agent1", [])
            agent2_rewards = episode_rewards.get("agent2", [])

            if agent1_rewards and agent2_rewards:
                episodes = range(len(agent1_rewards))
                window = max(10, len(agent1_rewards) // 20)

                smooth1 = pd.Series(agent1_rewards).rolling(window=window).mean()
                smooth2 = pd.Series(agent2_rewards).rolling(window=window).mean()

                ax3.plot(
                    episodes,
                    smooth1,
                    color=self.colors["agent1"],
                    linewidth=2,
                    label="Agent 1",
                )
                ax3.plot(
                    episodes,
                    smooth2,
                    color=self.colors["agent2"],
                    linewidth=2,
                    label="Agent 2",
                )
                ax3.set_title("Learning Progress")
                ax3.set_xlabel("Episode")
                ax3.set_ylabel("Smoothed Reward")
                ax3.legend()
                ax3.grid(True, alpha=0.3)

        # Performance statistics
        if episode_rewards:
            agent1_rewards = episode_rewards.get("agent1", [])
            agent2_rewards = episode_rewards.get("agent2", [])

            if agent1_rewards and agent2_rewards:
                stats_text = f"""
                Agent 1 Stats:
                Mean: {np.mean(agent1_rewards):.2f}
                Std: {np.std(agent1_rewards):.2f}
                
                Agent 2 Stats:
                Mean: {np.mean(agent2_rewards):.2f}
                Std: {np.std(agent2_rewards):.2f}
                """

                ax4.text(
                    0.1,
                    0.9,
                    stats_text,
                    transform=ax4.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    fontfamily="monospace",
                )
                ax4.set_title("Performance Statistics")
                ax4.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "summary_dashboard.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print("Created summary dashboard")


# Test function
def test_plotter():
    """Quick test of the plotter."""

    # Create fake results
    fake_results = {
        "run_name": "test_run",
        "config": {
            "agent1_type": "cooperative",
            "agent2_type": "competitive",
            "episodes": 100,
            "payoff_alpha": 4.0,
            "payoff_beta": 10.0,
        },
        "training_results": {
            "episode_rewards": {
                "agent1": [10 + i + np.random.randn() for i in range(100)],
                "agent2": [15 + i * 0.5 + np.random.randn() for i in range(100)],
            }
        },
        "final_evaluation": {"agent1_avg_reward": 105.5, "agent2_avg_reward": 108.2},
    }

    # Test plotting
    plotter = SimulationPlotter()

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        plotter.create_all_plots(fake_results, temp_dir)
        print(f"Test plots created in: {temp_dir}")


if __name__ == "__main__":
    test_plotter()
