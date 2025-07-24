"""
Plotting module for SAC agent simulation results.

This module creates various visualizations to analyze the training progress,
agent behavior, and interaction dynamics in interpersonal simulations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import os
from matplotlib.gridspec import GridSpec
import json


class SimulationPlotter:
    """
    Main plotting class for SAC simulation results.

    Creates comprehensive visualizations of agent training progress,
    interaction dynamics, and behavioral patterns.
    """

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize plotter with style preferences.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = {
            "agent1": "#2E86AB",
            "agent2": "#A23B72",
            "cooperative": "#28A745",
            "competitive": "#DC3545",
            "adaptive": "#FFC107",
            "cautious": "#6F42C1",
            "base": "#17A2B8",
        }

    def create_all_plots(self, results: Dict[str, Any], output_dir: str):
        """
        Create all standard plots for a simulation run.

        Args:
            results: Results dictionary from simulation
            output_dir: Directory to save plots
        """
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Extract data
        training_results = results["training_results"]
        config = results["config"]

        # 1. Training progress plots
        self.plot_training_progress(training_results, plots_dir)

        # 2. Reward dynamics
        self.plot_reward_dynamics(training_results, plots_dir)

        # 3. Action patterns
        if "environment_history" in results:
            self.plot_action_patterns(results["environment_history"], plots_dir)

        # 4. Learning curves
        self.plot_learning_curves(training_results, plots_dir)

        # 5. Agent comparison
        self.plot_agent_comparison(results, plots_dir)

        # 6. Summary dashboard
        self.create_summary_dashboard(results, plots_dir)

        print(f"All plots saved to: {plots_dir}")

    def plot_training_progress(self, training_results: Dict[str, Any], output_dir: str):
        """Plot training metrics over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Progress", fontsize=16, fontweight="bold")

        # Extract metrics
        agent1_metrics = training_results.get("training_metrics", {}).get("agent1", [])
        agent2_metrics = training_results.get("training_metrics", {}).get("agent2", [])

        if not agent1_metrics or not agent2_metrics:
            print("Warning: No training metrics found")
            plt.close(fig)
            return

        # Convert to arrays for plotting
        metrics_df1 = pd.DataFrame(agent1_metrics)
        metrics_df2 = pd.DataFrame(agent2_metrics)

        # Plot actor loss
        axes[0, 0].plot(
            metrics_df1["actor_loss"], label="Agent 1", color=self.colors["agent1"]
        )
        axes[0, 0].plot(
            metrics_df2["actor_loss"], label="Agent 2", color=self.colors["agent2"]
        )
        axes[0, 0].set_title("Actor Loss")
        axes[0, 0].set_xlabel("Training Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot critic loss
        axes[0, 1].plot(
            metrics_df1["critic_loss"], label="Agent 1", color=self.colors["agent1"]
        )
        axes[0, 1].plot(
            metrics_df2["critic_loss"], label="Agent 2", color=self.colors["agent2"]
        )
        axes[0, 1].set_title("Critic Loss")
        axes[0, 1].set_xlabel("Training Step")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot alpha (temperature) evolution
        axes[1, 0].plot(
            metrics_df1["alpha"], label="Agent 1", color=self.colors["agent1"]
        )
        axes[1, 0].plot(
            metrics_df2["alpha"], label="Agent 2", color=self.colors["agent2"]
        )
        axes[1, 0].set_title("Temperature Parameter (Alpha)")
        axes[1, 0].set_xlabel("Training Step")
        axes[1, 0].set_ylabel("Alpha")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot alpha loss
        axes[1, 1].plot(
            metrics_df1["alpha_loss"], label="Agent 1", color=self.colors["agent1"]
        )
        axes[1, 1].plot(
            metrics_df2["alpha_loss"], label="Agent 2", color=self.colors["agent2"]
        )
        axes[1, 1].set_title("Alpha Loss")
        axes[1, 1].set_xlabel("Training Step")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "training_progress.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_reward_dynamics(self, training_results: Dict[str, Any], output_dir: str):
        """Plot reward dynamics and learning curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Reward Dynamics", fontsize=16, fontweight="bold")

        # Episode rewards
        episode_rewards = training_results.get("episode_rewards", {})
        agent1_rewards = episode_rewards.get("agent1", [])
        agent2_rewards = episode_rewards.get("agent2", [])

        if not agent1_rewards or not agent2_rewards:
            print("Warning: No episode rewards found")
            plt.close(fig)
            return

        episodes = range(len(agent1_rewards))

        # Raw episode rewards
        axes[0, 0].plot(
            episodes,
            agent1_rewards,
            alpha=0.6,
            color=self.colors["agent1"],
            label="Agent 1",
        )
        axes[0, 0].plot(
            episodes,
            agent2_rewards,
            alpha=0.6,
            color=self.colors["agent2"],
            label="Agent 2",
        )
        axes[0, 0].set_title("Episode Rewards (Raw)")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Smoothed rewards (moving average)
        window = min(50, len(agent1_rewards) // 10)
        if window > 1:
            smooth1 = pd.Series(agent1_rewards).rolling(window=window).mean()
            smooth2 = pd.Series(agent2_rewards).rolling(window=window).mean()

            axes[0, 1].plot(
                episodes,
                smooth1,
                color=self.colors["agent1"],
                label="Agent 1",
                linewidth=2,
            )
            axes[0, 1].plot(
                episodes,
                smooth2,
                color=self.colors["agent2"],
                label="Agent 2",
                linewidth=2,
            )
            axes[0, 1].set_title(f"Smoothed Rewards (Window={window})")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Average Reward")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Reward distribution
        axes[1, 0].hist(
            agent1_rewards,
            bins=30,
            alpha=0.7,
            color=self.colors["agent1"],
            label="Agent 1",
        )
        axes[1, 0].hist(
            agent2_rewards,
            bins=30,
            alpha=0.7,
            color=self.colors["agent2"],
            label="Agent 2",
        )
        axes[1, 0].set_title("Reward Distribution")
        axes[1, 0].set_xlabel("Total Episode Reward")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Cumulative rewards
        cumulative1 = np.cumsum(agent1_rewards)
        cumulative2 = np.cumsum(agent2_rewards)

        axes[1, 1].plot(
            episodes,
            cumulative1,
            color=self.colors["agent1"],
            label="Agent 1",
            linewidth=2,
        )
        axes[1, 1].plot(
            episodes,
            cumulative2,
            color=self.colors["agent2"],
            label="Agent 2",
            linewidth=2,
        )
        axes[1, 1].set_title("Cumulative Rewards")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Cumulative Reward")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "reward_dynamics.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_action_patterns(self, env_history: Dict[str, List], output_dir: str):
        """Plot action patterns and interaction dynamics."""
        if not env_history or "agent1_actions" not in env_history:
            print("Warning: No environment history found")
            return

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)
        fig.suptitle(
            "Action Patterns and Interaction Dynamics", fontsize=16, fontweight="bold"
        )

        actions1 = env_history["agent1_actions"]
        actions2 = env_history["agent2_actions"]
        rewards = env_history["rewards"]

        if not actions1 or not actions2:
            print("Warning: Empty action history")
            plt.close(fig)
            return

        steps = range(len(actions1))

        # Time series of actions
        ax1 = fig.add_subplot(gs[0, :2])
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
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Warmth Level")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Action correlation
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.scatter(actions1, actions2, alpha=0.6, color="purple")
        ax2.set_title("Action Correlation")
        ax2.set_xlabel("Agent 1 Warmth")
        ax2.set_ylabel("Agent 2 Warmth")
        ax2.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(actions1, actions2)[0, 1]
        ax2.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Reward time series
        ax3 = fig.add_subplot(gs[1, :])
        rewards1 = [r[0] for r in rewards]
        rewards2 = [r[1] for r in rewards]

        ax3.plot(
            steps,
            rewards1,
            color=self.colors["agent1"],
            label="Agent 1 Rewards",
            linewidth=2,
        )
        ax3.plot(
            steps,
            rewards2,
            color=self.colors["agent2"],
            label="Agent 2 Rewards",
            linewidth=2,
        )
        ax3.set_title("Rewards Over Time")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Reward")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Action histograms
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.hist(
            actions1, bins=20, alpha=0.7, color=self.colors["agent1"], label="Agent 1"
        )
        ax4.set_title("Agent 1 Warmth Distribution")
        ax4.set_xlabel("Warmth Level")
        ax4.set_ylabel("Frequency")
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(
            actions2, bins=20, alpha=0.7, color=self.colors["agent2"], label="Agent 2"
        )
        ax5.set_title("Agent 2 Warmth Distribution")
        ax5.set_xlabel("Warmth Level")
        ax5.set_ylabel("Frequency")
        ax5.grid(True, alpha=0.3)

        # Interaction heatmap
        ax6 = fig.add_subplot(gs[2, 2])
        try:
            # Create 2D histogram
            hist, xedges, yedges = np.histogram2d(actions1, actions2, bins=15)
            im = ax6.imshow(hist.T, origin="lower", extent=[0, 1, 0, 1], cmap="YlOrRd")
            ax6.set_title("Interaction Heatmap")
            ax6.set_xlabel("Agent 1 Warmth")
            ax6.set_ylabel("Agent 2 Warmth")
            plt.colorbar(im, ax=ax6, label="Frequency")
        except Exception as e:
            print(f"Warning: Could not create heatmap: {e}")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "action_patterns.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_learning_curves(self, training_results: Dict[str, Any], output_dir: str):
        """Plot learning curves and convergence analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Learning Curves and Convergence", fontsize=16, fontweight="bold")

        episode_rewards = training_results.get("episode_rewards", {})
        agent1_rewards = episode_rewards.get("agent1", [])
        agent2_rewards = episode_rewards.get("agent2", [])

        if not agent1_rewards or not agent2_rewards:
            plt.close(fig)
            return

        episodes = range(len(agent1_rewards))

        # Learning curves with trend lines
        window = max(10, len(agent1_rewards) // 20)

        # Agent 1 learning curve
        smooth1 = pd.Series(agent1_rewards).rolling(window=window, center=True).mean()
        axes[0, 0].plot(
            episodes, agent1_rewards, alpha=0.3, color=self.colors["agent1"]
        )
        axes[0, 0].plot(
            episodes,
            smooth1,
            color=self.colors["agent1"],
            linewidth=2,
            label="Moving Average",
        )
        axes[0, 0].set_title("Agent 1 Learning Curve")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Episode Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Agent 2 learning curve
        smooth2 = pd.Series(agent2_rewards).rolling(window=window, center=True).mean()
        axes[0, 1].plot(
            episodes, agent2_rewards, alpha=0.3, color=self.colors["agent2"]
        )
        axes[0, 1].plot(
            episodes,
            smooth2,
            color=self.colors["agent2"],
            linewidth=2,
            label="Moving Average",
        )
        axes[0, 1].set_title("Agent 2 Learning Curve")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Episode Reward")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Convergence analysis - difference in performance
        performance_diff = np.array(agent1_rewards) - np.array(agent2_rewards)
        smooth_diff = (
            pd.Series(performance_diff).rolling(window=window, center=True).mean()
        )

        axes[1, 0].plot(episodes, performance_diff, alpha=0.3, color="gray")
        axes[1, 0].plot(episodes, smooth_diff, color="black", linewidth=2)
        axes[1, 0].axhline(y=0, color="red", linestyle="--", alpha=0.7)
        axes[1, 0].set_title("Performance Difference (Agent 1 - Agent 2)")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Reward Difference")
        axes[1, 0].grid(True, alpha=0.3)

        # Learning stability (variance over time)
        variance_window = max(20, len(agent1_rewards) // 10)
        variance1 = pd.Series(agent1_rewards).rolling(window=variance_window).var()
        variance2 = pd.Series(agent2_rewards).rolling(window=variance_window).var()

        axes[1, 1].plot(
            episodes,
            variance1,
            color=self.colors["agent1"],
            label="Agent 1",
            linewidth=2,
        )
        axes[1, 1].plot(
            episodes,
            variance2,
            color=self.colors["agent2"],
            label="Agent 2",
            linewidth=2,
        )
        axes[1, 1].set_title("Learning Stability (Reward Variance)")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Reward Variance")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "learning_curves.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_agent_comparison(self, results: Dict[str, Any], output_dir: str):
        """Create comparison plots between the two agents."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Agent Comparison Analysis", fontsize=16, fontweight="bold")

        config = results["config"]
        final_eval = results.get("final_evaluation", {})
        episode_rewards = results["training_results"].get("episode_rewards", {})

        # Performance comparison
        agent1_avg = final_eval.get("agent1_avg_reward", 0)
        agent2_avg = final_eval.get("agent2_avg_reward", 0)

        axes[0, 0].bar(
            [
                "Agent 1\n({})".format(config["agent1_type"]),
                "Agent 2\n({})".format(config["agent2_type"]),
            ],
            [agent1_avg, agent2_avg],
            color=[self.colors["agent1"], self.colors["agent2"]],
        )
        axes[0, 0].set_title("Final Average Performance")
        axes[0, 0].set_ylabel("Average Reward")
        axes[0, 0].grid(True, alpha=0.3)

        # Learning speed comparison (early vs late performance)
        if episode_rewards.get("agent1") and episode_rewards.get("agent2"):
            early_episodes = slice(0, min(100, len(episode_rewards["agent1"]) // 4))
            late_episodes = slice(-min(100, len(episode_rewards["agent1"]) // 4), None)

            early1 = np.mean(episode_rewards["agent1"][early_episodes])
            early2 = np.mean(episode_rewards["agent2"][early_episodes])
            late1 = np.mean(episode_rewards["agent1"][late_episodes])
            late2 = np.mean(episode_rewards["agent2"][late_episodes])

            x = np.arange(2)
            width = 0.35

            axes[0, 1].bar(
                x - width / 2,
                [early1, late1],
                width,
                label="Agent 1",
                color=self.colors["agent1"],
            )
            axes[0, 1].bar(
                x + width / 2,
                [early2, late2],
                width,
                label="Agent 2",
                color=self.colors["agent2"],
            )
            axes[0, 1].set_title("Learning Progress: Early vs Late")
            axes[0, 1].set_ylabel("Average Reward")
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(["Early Episodes", "Late Episodes"])
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Reward statistics comparison
        if episode_rewards.get("agent1") and episode_rewards.get("agent2"):
            stats_data = {
                "Agent": ["Agent 1", "Agent 1", "Agent 2", "Agent 2"],
                "Metric": ["Mean", "Std", "Mean", "Std"],
                "Value": [
                    np.mean(episode_rewards["agent1"]),
                    np.std(episode_rewards["agent1"]),
                    np.mean(episode_rewards["agent2"]),
                    np.std(episode_rewards["agent2"]),
                ],
            }

            # Create grouped bar chart
            df_stats = pd.DataFrame(stats_data)
            df_pivot = df_stats.pivot(index="Metric", columns="Agent", values="Value")

            df_pivot.plot(
                kind="bar",
                ax=axes[1, 0],
                color=[self.colors["agent1"], self.colors["agent2"]],
            )
            axes[1, 0].set_title("Performance Statistics")
            axes[1, 0].set_ylabel("Value")
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis="x", rotation=0)
            axes[1, 0].grid(True, alpha=0.3)

        # Configuration comparison
        agent_types = [config["agent1_type"], config["agent2_type"]]
        type_colors = [self.colors.get(t, "gray") for t in agent_types]

        axes[1, 1].pie(
            [1, 1],
            labels=agent_types,
            colors=type_colors,
            autopct="%1.0f%%",
            startangle=90,
        )
        axes[1, 1].set_title("Agent Type Configuration")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "agent_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def create_summary_dashboard(self, results: Dict[str, Any], output_dir: str):
        """Create a comprehensive summary dashboard."""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle(
            f'Simulation Summary: {results["run_name"]}', fontsize=18, fontweight="bold"
        )

        config = results["config"]
        training_results = results["training_results"]
        final_eval = results.get("final_evaluation", {})

        # Key metrics summary (text)
        ax_summary = fig.add_subplot(gs[0, 0])
        ax_summary.axis("off")

        summary_text = f"""
        SIMULATION SUMMARY
        
        Agent 1: {config['agent1_type']}
        Agent 2: {config['agent2_type']}
        
        Episodes: {config['episodes']}
        Steps/Episode: {config['steps_per_episode']}
        
        Final Performance:
        Agent 1: {final_eval.get('agent1_avg_reward', 0):.3f}
        Agent 2: {final_eval.get('agent2_avg_reward', 0):.3f}
        
        Payoff Parameters:
        Alpha: {config['payoff_alpha']} 
        Beta: {config['payoff_beta']}
        """

        ax_summary.text(
            0.05,
            0.95,
            summary_text,
            transform=ax_summary.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        # Quick performance comparison
        ax_perf = fig.add_subplot(gs[0, 1])
        if final_eval:
            performance = [
                final_eval.get("agent1_avg_reward", 0),
                final_eval.get("agent2_avg_reward", 0),
            ]
            bars = ax_perf.bar(
                ["Agent 1", "Agent 2"],
                performance,
                color=[self.colors["agent1"], self.colors["agent2"]],
            )
            ax_perf.set_title("Final Performance")
            ax_perf.set_ylabel("Avg Reward")

            # Add value labels on bars
            for bar, value in zip(bars, performance):
                ax_perf.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

        # Learning progress mini-plot
        ax_learning = fig.add_subplot(gs[0, 2:])
        episode_rewards = training_results.get("episode_rewards", {})
        if episode_rewards.get("agent1") and episode_rewards.get("agent2"):
            episodes = range(len(episode_rewards["agent1"]))
            window = max(10, len(episode_rewards["agent1"]) // 20)

            smooth1 = pd.Series(episode_rewards["agent1"]).rolling(window=window).mean()
            smooth2 = pd.Series(episode_rewards["agent2"]).rolling(window=window).mean()

            ax_learning.plot(
                episodes,
                smooth1,
                color=self.colors["agent1"],
                label="Agent 1",
                linewidth=2,
            )
            ax_learning.plot(
                episodes,
                smooth2,
                color=self.colors["agent2"],
                label="Agent 2",
                linewidth=2,
            )
            ax_learning.set_title("Learning Progress")
            ax_learning.set_xlabel("Episode")
            ax_learning.set_ylabel("Smoothed Reward")
            ax_learning.legend()
            ax_learning.grid(True, alpha=0.3)

        # Action patterns from last episode
        if "environment_history" in results and results["environment_history"]:
            env_history = results["environment_history"]

            # Recent actions trajectory
            ax_traj = fig.add_subplot(gs[1, :2])
            if env_history.get("agent1_actions") and env_history.get("agent2_actions"):
                recent_steps = slice(-100, None)  # Last 100 steps
                recent_actions1 = env_history["agent1_actions"][recent_steps]
                recent_actions2 = env_history["agent2_actions"][recent_steps]
                recent_steps_range = range(len(recent_actions1))

                ax_traj.plot(
                    recent_steps_range,
                    recent_actions1,
                    color=self.colors["agent1"],
                    label="Agent 1",
                    linewidth=2,
                )
                ax_traj.plot(
                    recent_steps_range,
                    recent_actions2,
                    color=self.colors["agent2"],
                    label="Agent 2",
                    linewidth=2,
                )
                ax_traj.set_title("Recent Action Trajectory (Last 100 Steps)")
                ax_traj.set_xlabel("Step")
                ax_traj.set_ylabel("Warmth Level")
                ax_traj.legend()
                ax_traj.grid(True, alpha=0.3)

            # Action space exploration
            ax_explore = fig.add_subplot(gs[1, 2:])
            if env_history.get("agent1_actions") and env_history.get("agent2_actions"):
                ax_explore.scatter(
                    env_history["agent1_actions"],
                    env_history["agent2_actions"],
                    alpha=0.6,
                    c=range(len(env_history["agent1_actions"])),
                    cmap="viridis",
                    s=20,
                )
                ax_explore.set_title("Action Space Exploration")
                ax_explore.set_xlabel("Agent 1 Warmth")
                ax_explore.set_ylabel("Agent 2 Warmth")
                cbar = plt.colorbar(ax_explore.collections[0], ax=ax_explore)
                cbar.set_label("Time Step")

        # Training metrics overview
        if training_results.get("training_metrics"):
            ax_metrics = fig.add_subplot(gs[2, :])

            agent1_metrics = training_results["training_metrics"].get("agent1", [])
            agent2_metrics = training_results["training_metrics"].get("agent2", [])

            if agent1_metrics and agent2_metrics:
                # Plot multiple metrics on same axis with different scales
                metrics_df1 = pd.DataFrame(agent1_metrics)
                metrics_df2 = pd.DataFrame(agent2_metrics)

                # Normalize metrics for comparison
                for col in ["actor_loss", "critic_loss"]:
                    if col in metrics_df1.columns:
                        norm1 = (metrics_df1[col] - metrics_df1[col].min()) / (
                            metrics_df1[col].max() - metrics_df1[col].min() + 1e-8
                        )
                        norm2 = (metrics_df2[col] - metrics_df2[col].min()) / (
                            metrics_df2[col].max() - metrics_df2[col].min() + 1e-8
                        )

                        ax_metrics.plot(norm1, label=f"Agent 1 {col}", alpha=0.7)
                        ax_metrics.plot(norm2, label=f"Agent 2 {col}", alpha=0.7)

                ax_metrics.set_title("Normalized Training Metrics")
                ax_metrics.set_xlabel("Training Step")
                ax_metrics.set_ylabel("Normalized Value")
                ax_metrics.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                ax_metrics.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "summary_dashboard.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def create_comparison_plots(
        self, comparison_results: Dict[str, List[Dict[str, Any]]], output_dir: str
    ):
        """Create plots comparing different agent type combinations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Agent Type Comparison Study", fontsize=16, fontweight="bold")

        # Collect data for comparison
        comparison_data = []

        for pair_key, runs in comparison_results.items():
            agent1_type, agent2_type = pair_key.split("_vs_")

            for run_idx, run_result in enumerate(runs):
                final_eval = run_result.get("final_evaluation", {})

                comparison_data.append(
                    {
                        "pair": pair_key,
                        "agent1_type": agent1_type,
                        "agent2_type": agent2_type,
                        "run": run_idx,
                        "agent1_reward": final_eval.get("agent1_avg_reward", 0),
                        "agent2_reward": final_eval.get("agent2_avg_reward", 0),
                        "total_reward": final_eval.get("agent1_avg_reward", 0)
                        + final_eval.get("agent2_avg_reward", 0),
                    }
                )

        df = pd.DataFrame(comparison_data)

        if df.empty:
            print("Warning: No comparison data to plot")
            plt.close(fig)
            return

        # Average performance by agent type
        avg_performance = df.groupby("pair")[
            ["agent1_reward", "agent2_reward", "total_reward"]
        ].mean()

        # Individual agent performance
        axes[0, 0].bar(
            range(len(avg_performance)),
            avg_performance["agent1_reward"],
            alpha=0.7,
            label="Agent 1",
            color=self.colors["agent1"],
        )
        axes[0, 0].bar(
            range(len(avg_performance)),
            avg_performance["agent2_reward"],
            alpha=0.7,
            label="Agent 2",
            color=self.colors["agent2"],
        )
        axes[0, 0].set_title("Average Performance by Pair")
        axes[0, 0].set_xlabel("Agent Pair")
        axes[0, 0].set_ylabel("Average Reward")
        axes[0, 0].set_xticks(range(len(avg_performance)))
        axes[0, 0].set_xticklabels(
            [p.replace("_vs_", "\nvs\n") for p in avg_performance.index], rotation=45
        )
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Total system performance
        axes[0, 1].bar(
            range(len(avg_performance)),
            avg_performance["total_reward"],
            color="green",
            alpha=0.7,
        )
        axes[0, 1].set_title("Total System Performance")
        axes[0, 1].set_xlabel("Agent Pair")
        axes[0, 1].set_ylabel("Combined Average Reward")
        axes[0, 1].set_xticks(range(len(avg_performance)))
        axes[0, 1].set_xticklabels(
            [p.replace("_vs_", "\nvs\n") for p in avg_performance.index], rotation=45
        )
        axes[0, 1].grid(True, alpha=0.3)

        # Performance variance (stability)
        performance_std = df.groupby("pair")[["agent1_reward", "agent2_reward"]].std()

        axes[1, 0].bar(
            range(len(performance_std)),
            performance_std["agent1_reward"],
            alpha=0.7,
            label="Agent 1",
            color=self.colors["agent1"],
        )
        axes[1, 0].bar(
            range(len(performance_std)),
            performance_std["agent2_reward"],
            alpha=0.7,
            label="Agent 2",
            color=self.colors["agent2"],
        )
        axes[1, 0].set_title("Performance Variability (Std Dev)")
        axes[1, 0].set_xlabel("Agent Pair")
        axes[1, 0].set_ylabel("Standard Deviation")
        axes[1, 0].set_xticks(range(len(performance_std)))
        axes[1, 0].set_xticklabels(
            [p.replace("_vs_", "\nvs\n") for p in performance_std.index], rotation=45
        )
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Heatmap of pairwise performance
        try:
            # Create matrix for heatmap
            unique_types = sorted(
                set(df["agent1_type"].tolist() + df["agent2_type"].tolist())
            )
            performance_matrix = np.zeros((len(unique_types), len(unique_types)))

            for _, row in avg_performance.iterrows():
                pair = row.name
                agent1_type, agent2_type = pair.split("_vs_")
                i = unique_types.index(agent1_type)
                j = unique_types.index(agent2_type)

                # Use total reward for the matrix
                performance_matrix[i, j] = row["total_reward"]
                if i != j:  # Make symmetric for different pairs
                    performance_matrix[j, i] = row["total_reward"]

            im = axes[1, 1].imshow(performance_matrix, cmap="RdYlGn", aspect="equal")
            axes[1, 1].set_title("Pairwise Performance Heatmap")
            axes[1, 1].set_xticks(range(len(unique_types)))
            axes[1, 1].set_yticks(range(len(unique_types)))
            axes[1, 1].set_xticklabels(unique_types, rotation=45)
            axes[1, 1].set_yticklabels(unique_types)

            # Add text annotations
            for i in range(len(unique_types)):
                for j in range(len(unique_types)):
                    text = axes[1, 1].text(
                        j,
                        i,
                        f"{performance_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                    )

            plt.colorbar(im, ax=axes[1, 1], label="Total Reward")

        except Exception as e:
            print(f"Warning: Could not create performance heatmap: {e}")
            axes[1, 1].text(
                0.5,
                0.5,
                "Heatmap\nUnavailable",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=12,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "summary_dashboard.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save comparison summary as CSV
        df.to_csv(os.path.join(output_dir, "comparison_summary.csv"), index=False)

    def plot_payoff_landscape(
        self, alpha: float = 4.0, beta: float = 10.0, output_dir: str = "./plots"
    ):
        """Create a detailed payoff landscape visualization."""
        from payoff_functions.gaussian_payoff import calculate_warmth_payoff

        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f"Payoff Landscape (α={alpha}, β={beta})", fontsize=16, fontweight="bold"
        )

        # Create grid for payoff calculation
        resolution = 100
        w1_range = np.linspace(0, 1, resolution)
        w2_range = np.linspace(0, 1, resolution)
        W1, W2 = np.meshgrid(w1_range, w2_range)

        # Calculate payoffs for agent 1
        payoff_matrix = np.zeros_like(W1)
        for i in range(resolution):
            for j in range(resolution):
                payoff_matrix[i, j] = calculate_warmth_payoff(
                    W1[i, j], W2[i, j], alpha, beta
                )

        # Payoff heatmap
        im1 = axes[0].imshow(
            payoff_matrix, extent=[0, 1, 0, 1], origin="lower", cmap="RdYlBu_r"
        )
        axes[0].set_title("Payoff Landscape for Agent 1")
        axes[0].set_xlabel("Agent 1 Warmth")
        axes[0].set_ylabel("Agent 2 Warmth")
        plt.colorbar(im1, ax=axes[0], label="Payoff")

        # Add contour lines
        contours = axes[0].contour(
            W1, W2, payoff_matrix, levels=10, colors="black", alpha=0.4, linewidths=0.5
        )
        axes[0].clabel(contours, inline=True, fontsize=8)

        # 3D surface plot
        from mpl_toolkits.mplot3d import Axes3D

        ax_3d = fig.add_subplot(122, projection="3d")

        # Subsample for 3D plot (for performance)
        step = 5
        W1_sub = W1[::step, ::step]
        W2_sub = W2[::step, ::step]
        payoff_sub = payoff_matrix[::step, ::step]

        surface = ax_3d.plot_surface(
            W1_sub, W2_sub, payoff_sub, cmap="RdYlBu_r", alpha=0.8
        )
        ax_3d.set_title("3D Payoff Surface")
        ax_3d.set_xlabel("Agent 1 Warmth")
        ax_3d.set_ylabel("Agent 2 Warmth")
        ax_3d.set_zlabel("Payoff")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "payoff_landscape.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_specific_results(results_path: str, plot_types: Optional[List[str]] = None):
    """
    Utility function to generate specific plots from saved results.

    Args:
        results_path: Path to results.json file
        plot_types: List of plot types to generate (None for all)
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    output_dir = os.path.dirname(results_path)
    plotter = SimulationPlotter()

    available_plots = {
        "training": plotter.plot_training_progress,
        "rewards": plotter.plot_reward_dynamics,
        "actions": plotter.plot_action_patterns,
        "learning": plotter.plot_learning_curves,
        "comparison": plotter.plot_agent_comparison,
        "dashboard": plotter.create_summary_dashboard,
    }

    if plot_types is None:
        plot_types = list(available_plots.keys())

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for plot_type in plot_types:
        if plot_type in available_plots:
            try:
                if plot_type == "actions" and "environment_history" in results:
                    available_plots[plot_type](
                        results["environment_history"], plots_dir
                    )
                elif plot_type in ["comparison", "dashboard"]:
                    available_plots[plot_type](results, plots_dir)
                else:
                    available_plots[plot_type](results["training_results"], plots_dir)
                print(f"Generated {plot_type} plot")
            except Exception as e:
                print(f"Error generating {plot_type} plot: {e}")
        else:
            print(f"Unknown plot type: {plot_type}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        plot_types = sys.argv[2:] if len(sys.argv) > 2 else None
        plot_specific_results(results_file, plot_types)
    else:
        print("Usage: python simulation_plots.py <results.json> [plot_types...]")
        print(
            "Available plot types: training, rewards, actions, learning, comparison, dashboard"
        )
