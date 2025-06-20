"""
Comprehensive logging system for SAC interpersonal behavior simulation.
Tracks detailed metrics about agent behavior, relationship dynamics, and learning patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import seaborn as sns

# Set style for better plots
plt.style.use("default")
sns.set_palette("husl")


class ComprehensiveLogger:
    """
    Comprehensive logging system that tracks:
    1. Agent behavior patterns (warmth, consistency, adaptation)
    2. Relationship dynamics (trust, satisfaction, cooperation levels)
    3. Learning metrics (convergence, exploration, stability)
    4. Interaction patterns (action sequences, response patterns)
    5. Performance analytics (rewards, efficiency, mutual benefit)
    """

    def __init__(self, experiment_name: str, log_dir: str = "./logs"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.start_time = datetime.now()

        # Create logging directory
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Initialize logging containers
        self.episode_logs = []
        self.step_logs = []
        self.training_logs = {"agent1": [], "agent2": []}
        self.relationship_logs = []
        self.behavior_patterns = {
            "agent1": defaultdict(list),
            "agent2": defaultdict(list),
        }

        # Metrics tracking
        self.metrics = {
            "episode_rewards": {"agent1": [], "agent2": []},
            "episode_actions": {"agent1": [], "agent2": []},
            "trust_evolution": {"agent1": [], "agent2": []},
            "satisfaction_evolution": {"agent1": [], "agent2": []},
            "cooperation_levels": [],
            "mutual_benefit_score": [],
            "relationship_stability": [],
            "learning_progress": {"agent1": [], "agent2": []},
        }

        # Behavioral analysis
        self.action_sequences = {
            "agent1": deque(maxlen=1000),
            "agent2": deque(maxlen=1000),
        }
        self.response_patterns = []
        self.adaptation_tracking = {"agent1": [], "agent2": []}

        print(f"[LOGGER] Initialized comprehensive logging for {experiment_name}")
        print(f"[LOGGER] Logs will be saved to: {self.experiment_dir}")

    def log_episode_start(self, episode: int):
        """Log the start of a new episode."""
        self.current_episode = episode
        self.episode_start_time = datetime.now()
        self.current_episode_data = {
            "episode": episode,
            "start_time": self.episode_start_time,
            "steps": [],
            "agent1_actions": [],
            "agent2_actions": [],
            "rewards": {"agent1": [], "agent2": []},
            "states": {"agent1": [], "agent2": []},
        }

    def log_step(
        self,
        step: int,
        agent1_state: np.ndarray,
        agent2_state: np.ndarray,
        agent1_action: float,
        agent2_action: float,
        agent1_reward: float,
        agent2_reward: float,
        environment_stats: Dict = None,
    ):
        """Log detailed information for each step within an episode."""

        # Store step data
        step_data = {
            "episode": self.current_episode,
            "step": step,
            "timestamp": datetime.now(),
            "agent1_action": agent1_action,
            "agent2_action": agent2_action,
            "agent1_reward": agent1_reward,
            "agent2_reward": agent2_reward,
            "agent1_state_summary": self._summarize_state(agent1_state),
            "agent2_state_summary": self._summarize_state(agent2_state),
        }

        # Add environment stats if available
        if environment_stats:
            step_data.update(environment_stats)

        self.step_logs.append(step_data)

        # Update current episode data
        self.current_episode_data["steps"].append(step)
        self.current_episode_data["agent1_actions"].append(agent1_action)
        self.current_episode_data["agent2_actions"].append(agent2_action)
        self.current_episode_data["rewards"]["agent1"].append(agent1_reward)
        self.current_episode_data["rewards"]["agent2"].append(agent2_reward)

        # Track action sequences
        self.action_sequences["agent1"].append(agent1_action)
        self.action_sequences["agent2"].append(agent2_action)

        # Analyze response patterns (how agents respond to each other)
        if len(self.action_sequences["agent1"]) > 1:
            response_pattern = {
                "episode": self.current_episode,
                "step": step,
                "agent1_prev_action": self.action_sequences["agent1"][-2],
                "agent1_curr_action": agent1_action,
                "agent2_prev_action": self.action_sequences["agent2"][-2],
                "agent2_curr_action": agent2_action,
                "agent1_response_to_agent2": agent1_action
                - self.action_sequences["agent2"][-2],
                "agent2_response_to_agent1": agent2_action
                - self.action_sequences["agent1"][-2],
            }
            self.response_patterns.append(response_pattern)

    def log_episode_end(self, final_states: Dict = None):
        """Log the end of an episode and compute episode-level metrics."""

        # Complete episode data
        self.current_episode_data["end_time"] = datetime.now()
        self.current_episode_data["duration"] = (
            self.current_episode_data["end_time"] - self.episode_start_time
        ).total_seconds()

        # Compute episode metrics
        episode_metrics = self._compute_episode_metrics()
        self.current_episode_data["metrics"] = episode_metrics

        # Add final states if available
        if final_states:
            self.current_episode_data["final_states"] = final_states

        # Store episode
        self.episode_logs.append(self.current_episode_data.copy())

        # Update aggregate metrics
        self._update_aggregate_metrics()

    def log_training_step(self, agent_id: str, training_metrics: Dict):
        """Log training step metrics for an agent."""
        training_data = {
            "episode": getattr(self, "current_episode", 0),
            "timestamp": datetime.now(),
            "agent_id": agent_id,
            **training_metrics,
        }
        self.training_logs[agent_id].append(training_data)

        # Track learning progress
        if "actor_loss" in training_metrics and "critic_loss" in training_metrics:
            progress_metric = {
                "episode": getattr(self, "current_episode", 0),
                "actor_loss": training_metrics["actor_loss"],
                "critic_loss": training_metrics["critic_loss"],
                "alpha": training_metrics.get("alpha", 0),
                "combined_loss": training_metrics["actor_loss"]
                + training_metrics["critic_loss"],
            }
            self.metrics["learning_progress"][agent_id].append(progress_metric)

    def _summarize_state(self, state: np.ndarray) -> Dict:
        """Create a summary of the agent's state vector."""
        return {
            "mean": float(np.mean(state)),
            "std": float(np.std(state)),
            "min": float(np.min(state)),
            "max": float(np.max(state)),
            "trust_estimate": float(state[0]) if len(state) > 0 else 0.0,
            "satisfaction_estimate": float(state[1]) if len(state) > 1 else 0.0,
        }

    def _compute_episode_metrics(self) -> Dict:
        """Compute comprehensive metrics for the completed episode."""
        actions1 = np.array(self.current_episode_data["agent1_actions"])
        actions2 = np.array(self.current_episode_data["agent2_actions"])
        rewards1 = np.array(self.current_episode_data["rewards"]["agent1"])
        rewards2 = np.array(self.current_episode_data["rewards"]["agent2"])

        metrics = {
            # Basic performance
            "total_reward_agent1": float(np.sum(rewards1)),
            "total_reward_agent2": float(np.sum(rewards2)),
            "avg_reward_agent1": float(np.mean(rewards1)),
            "avg_reward_agent2": float(np.mean(rewards2)),
            # Behavioral patterns
            "avg_warmth_agent1": float(np.mean(actions1)),
            "avg_warmth_agent2": float(np.mean(actions2)),
            "warmth_std_agent1": float(np.std(actions1)),
            "warmth_std_agent2": float(np.std(actions2)),
            # Cooperation analysis
            "mutual_cooperation_rate": float(np.mean((actions1 > 0) & (actions2 > 0))),
            "mutual_defection_rate": float(np.mean((actions1 < 0) & (actions2 < 0))),
            "exploitation_by_agent1": float(np.mean((actions1 < 0) & (actions2 > 0))),
            "exploitation_by_agent2": float(np.mean((actions1 > 0) & (actions2 < 0))),
            # Relationship dynamics
            "action_correlation": (
                float(np.corrcoef(actions1, actions2)[0, 1])
                if len(actions1) > 1
                else 0.0
            ),
            "reward_correlation": (
                float(np.corrcoef(rewards1, rewards2)[0, 1])
                if len(rewards1) > 1
                else 0.0
            ),
            # Adaptation and responsiveness
            "agent1_action_changes": (
                float(np.mean(np.abs(np.diff(actions1)))) if len(actions1) > 1 else 0.0
            ),
            "agent2_action_changes": (
                float(np.mean(np.abs(np.diff(actions2)))) if len(actions2) > 1 else 0.0
            ),
            # Mutual benefit
            "mutual_benefit_score": float(np.mean(rewards1 + rewards2)),
            "reward_fairness": 1.0
            - abs(np.mean(rewards1) - np.mean(rewards2))
            / (abs(np.mean(rewards1)) + abs(np.mean(rewards2)) + 1e-6),
        }

        return metrics

    def _update_aggregate_metrics(self):
        """Update aggregate metrics based on the latest episode."""
        episode_metrics = self.current_episode_data["metrics"]

        # Update reward tracking
        self.metrics["episode_rewards"]["agent1"].append(
            episode_metrics["total_reward_agent1"]
        )
        self.metrics["episode_rewards"]["agent2"].append(
            episode_metrics["total_reward_agent2"]
        )

        # Update behavioral tracking
        self.metrics["cooperation_levels"].append(
            episode_metrics["mutual_cooperation_rate"]
        )
        self.metrics["mutual_benefit_score"].append(
            episode_metrics["mutual_benefit_score"]
        )

        # Update relationship tracking
        if "final_states" in self.current_episode_data:
            final_states = self.current_episode_data["final_states"]
            if "agent1_trust" in final_states:
                self.metrics["trust_evolution"]["agent1"].append(
                    final_states["agent1_trust"]
                )
            if "agent2_trust" in final_states:
                self.metrics["trust_evolution"]["agent2"].append(
                    final_states["agent2_trust"]
                )
            if "agent1_satisfaction" in final_states:
                self.metrics["satisfaction_evolution"]["agent1"].append(
                    final_states["agent1_satisfaction"]
                )
            if "agent2_satisfaction" in final_states:
                self.metrics["satisfaction_evolution"]["agent2"].append(
                    final_states["agent2_satisfaction"]
                )

    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE ANALYSIS REPORT: {self.experiment_name}")
        print(f"{'='*80}")

        # Basic statistics
        total_episodes = len(self.episode_logs)
        total_steps = len(self.step_logs)

        print(f"\n BASIC STATISTICS")
        print(f"   Total Episodes: {total_episodes}")
        print(f"   Total Steps: {total_steps}")
        print(f"   Average Steps per Episode: {total_steps/max(1, total_episodes):.1f}")
        print(f"   Training Duration: {datetime.now() - self.start_time}")

        if total_episodes > 0:
            # Performance analysis
            self._analyze_performance()

            # Behavioral analysis
            self._analyze_behavior_patterns()

            # Relationship dynamics
            self._analyze_relationship_dynamics()

            # Learning progress
            self._analyze_learning_progress()

        # Save detailed logs
        self._save_logs()

        # Generate visualizations
        self._generate_visualizations()

        print(f"\nAll logs and visualizations saved to: {self.experiment_dir}")

    def _analyze_performance(self):
        """Analyze agent performance metrics."""
        print(f"\n PERFORMANCE ANALYSIS")

        rewards1 = self.metrics["episode_rewards"]["agent1"]
        rewards2 = self.metrics["episode_rewards"]["agent2"]

        print(
            f"   Agent 1 - Avg Reward: {np.mean(rewards1):.3f} ± {np.std(rewards1):.3f}"
        )
        print(
            f"   Agent 2 - Avg Reward: {np.mean(rewards2):.3f} ± {np.std(rewards2):.3f}"
        )
        print(
            f"   Total System Reward: {np.mean(np.array(rewards1) + np.array(rewards2)):.3f}"
        )

        # Performance trends
        if len(rewards1) > 10:
            early_performance = np.mean(rewards1[: len(rewards1) // 4])
            late_performance = np.mean(rewards1[-len(rewards1) // 4 :])
            print(f"   Agent 1 Improvement: {late_performance - early_performance:.3f}")

            early_performance = np.mean(rewards2[: len(rewards2) // 4])
            late_performance = np.mean(rewards2[-len(rewards2) // 4 :])
            print(f"   Agent 2 Improvement: {late_performance - early_performance:.3f}")

    def _analyze_behavior_patterns(self):
        """Analyze behavioral patterns of both agents."""
        print(f"\n BEHAVIORAL ANALYSIS")

        if self.episode_logs:
            # Cooperation patterns
            coop_rates = [
                ep["metrics"]["mutual_cooperation_rate"] for ep in self.episode_logs
            ]
            defect_rates = [
                ep["metrics"]["mutual_defection_rate"] for ep in self.episode_logs
            ]

            print(f"   Average Mutual Cooperation Rate: {np.mean(coop_rates):.3f}")
            print(f"   Average Mutual Defection Rate: {np.mean(defect_rates):.3f}")

            # Behavioral consistency
            warmth_std1 = [
                ep["metrics"]["warmth_std_agent1"] for ep in self.episode_logs
            ]
            warmth_std2 = [
                ep["metrics"]["warmth_std_agent2"] for ep in self.episode_logs
            ]

            print(
                f"   Agent 1 Behavioral Consistency: {1.0 - np.mean(warmth_std1):.3f}"
            )
            print(
                f"   Agent 2 Behavioral Consistency: {1.0 - np.mean(warmth_std2):.3f}"
            )

            # Adaptation patterns
            action_changes1 = [
                ep["metrics"]["agent1_action_changes"] for ep in self.episode_logs
            ]
            action_changes2 = [
                ep["metrics"]["agent2_action_changes"] for ep in self.episode_logs
            ]

            print(f"   Agent 1 Adaptability: {np.mean(action_changes1):.3f}")
            print(f"   Agent 2 Adaptability: {np.mean(action_changes2):.3f}")

    def _analyze_relationship_dynamics(self):
        """Analyze relationship and interaction dynamics."""
        print(f"\n RELATIONSHIP DYNAMICS")

        if self.episode_logs:
            # Action correlations
            correlations = [
                ep["metrics"]["action_correlation"]
                for ep in self.episode_logs
                if not np.isnan(ep["metrics"]["action_correlation"])
            ]
            if correlations:
                print(f"   Average Action Correlation: {np.mean(correlations):.3f}")

            # Mutual benefit
            mutual_benefits = [
                ep["metrics"]["mutual_benefit_score"] for ep in self.episode_logs
            ]
            print(f"   Average Mutual Benefit: {np.mean(mutual_benefits):.3f}")

            # Relationship stability (consistency of interactions)
            if len(mutual_benefits) > 5:
                stability = 1.0 - (
                    np.std(mutual_benefits) / (abs(np.mean(mutual_benefits)) + 1e-6)
                )
                print(f"   Relationship Stability: {stability:.3f}")

        # Trust and satisfaction evolution
        if self.metrics["trust_evolution"]["agent1"]:
            trust1_trend = np.polyfit(
                range(len(self.metrics["trust_evolution"]["agent1"])),
                self.metrics["trust_evolution"]["agent1"],
                1,
            )[0]
            print(f"   Agent 1 Trust Trend: {trust1_trend:.4f} per episode")

        if self.metrics["trust_evolution"]["agent2"]:
            trust2_trend = np.polyfit(
                range(len(self.metrics["trust_evolution"]["agent2"])),
                self.metrics["trust_evolution"]["agent2"],
                1,
            )[0]
            print(f"   Agent 2 Trust Trend: {trust2_trend:.4f} per episode")

    def _analyze_learning_progress(self):
        """Analyze learning progress and convergence."""
        print(f"\n LEARNING PROGRESS")

        for agent_id in ["agent1", "agent2"]:
            if self.metrics["learning_progress"][agent_id]:
                losses = [
                    m["combined_loss"]
                    for m in self.metrics["learning_progress"][agent_id]
                ]
                alphas = [
                    m["alpha"] for m in self.metrics["learning_progress"][agent_id]
                ]

                print(f"   {agent_id.title()}:")
                print(f"     Final Combined Loss: {losses[-1]:.4f}")
                print(f"     Final Temperature: {alphas[-1]:.4f}")

                if len(losses) > 10:
                    early_loss = np.mean(losses[: len(losses) // 4])
                    late_loss = np.mean(losses[-len(losses) // 4 :])
                    improvement = (early_loss - late_loss) / early_loss * 100
                    print(f"     Loss Improvement: {improvement:.1f}%")

    def _save_logs(self):
        """Save all logs to files."""
        # Save episode logs
        with open(os.path.join(self.experiment_dir, "episode_logs.json"), "w") as f:
            json.dump(self.episode_logs, f, indent=2, default=str)

        # Save training logs
        with open(os.path.join(self.experiment_dir, "training_logs.json"), "w") as f:
            json.dump(self.training_logs, f, indent=2, default=str)

        # Save metrics
        with open(os.path.join(self.experiment_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)

        # Save response patterns as CSV for easy analysis
        if self.response_patterns:
            df = pd.DataFrame(self.response_patterns)
            df.to_csv(
                os.path.join(self.experiment_dir, "response_patterns.csv"), index=False
            )

    def _generate_visualizations(self):
        """Generate comprehensive visualizations."""
        if not self.episode_logs:
            return

        # Create a comprehensive dashboard
        fig = plt.figure(figsize=(20, 24))

        # 1. Episode rewards over time
        plt.subplot(4, 3, 1)
        episodes = range(len(self.metrics["episode_rewards"]["agent1"]))
        plt.plot(
            episodes,
            self.metrics["episode_rewards"]["agent1"],
            label="Agent 1",
            alpha=0.7,
        )
        plt.plot(
            episodes,
            self.metrics["episode_rewards"]["agent2"],
            label="Agent 2",
            alpha=0.7,
        )

        # Add moving averages
        if len(episodes) > 20:
            window = min(50, len(episodes) // 10)
            avg1 = np.convolve(
                self.metrics["episode_rewards"]["agent1"],
                np.ones(window) / window,
                mode="valid",
            )
            avg2 = np.convolve(
                self.metrics["episode_rewards"]["agent2"],
                np.ones(window) / window,
                mode="valid",
            )
            avg_episodes = range(window - 1, len(episodes))
            plt.plot(avg_episodes, avg1, "--", linewidth=2, label="Agent 1 (avg)")
            plt.plot(avg_episodes, avg2, "--", linewidth=2, label="Agent 2 (avg)")

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Episode Rewards Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Cooperation levels over time
        plt.subplot(4, 3, 2)
        plt.plot(episodes, self.metrics["cooperation_levels"], color="green", alpha=0.7)
        if len(self.metrics["cooperation_levels"]) > 20:
            window = min(30, len(episodes) // 10)
            avg_coop = np.convolve(
                self.metrics["cooperation_levels"],
                np.ones(window) / window,
                mode="valid",
            )
            avg_episodes = range(window - 1, len(episodes))
            plt.plot(
                avg_episodes,
                avg_coop,
                "--",
                linewidth=2,
                color="darkgreen",
                label="Moving Avg",
            )
        plt.xlabel("Episode")
        plt.ylabel("Mutual Cooperation Rate")
        plt.title("Cooperation Evolution")
        plt.grid(True, alpha=0.3)

        # 3. Trust evolution (if available)
        plt.subplot(4, 3, 3)
        if self.metrics["trust_evolution"]["agent1"]:
            plt.plot(
                self.metrics["trust_evolution"]["agent1"],
                label="Agent 1 Trust",
                alpha=0.7,
            )
        if self.metrics["trust_evolution"]["agent2"]:
            plt.plot(
                self.metrics["trust_evolution"]["agent2"],
                label="Agent 2 Trust",
                alpha=0.7,
            )
        plt.xlabel("Episode")
        plt.ylabel("Trust Level")
        plt.title("Trust Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Action distribution histogram
        plt.subplot(4, 3, 4)
        all_actions1 = [
            action for ep in self.episode_logs for action in ep["agent1_actions"]
        ]
        all_actions2 = [
            action for ep in self.episode_logs for action in ep["agent2_actions"]
        ]
        plt.hist(all_actions1, bins=30, alpha=0.5, label="Agent 1", density=True)
        plt.hist(all_actions2, bins=30, alpha=0.5, label="Agent 2", density=True)
        plt.xlabel("Action Value (Warmth)")
        plt.ylabel("Density")
        plt.title("Action Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Learning curves (if available)
        plt.subplot(4, 3, 5)
        for agent_id in ["agent1", "agent2"]:
            if self.metrics["learning_progress"][agent_id]:
                losses = [
                    m["combined_loss"]
                    for m in self.metrics["learning_progress"][agent_id]
                ]
                plt.plot(losses, label=f"{agent_id} Loss", alpha=0.7)
        plt.xlabel("Training Step")
        plt.ylabel("Combined Loss")
        plt.title("Learning Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        # 6. Mutual benefit over time
        plt.subplot(4, 3, 6)
        plt.plot(
            episodes, self.metrics["mutual_benefit_score"], color="purple", alpha=0.7
        )
        if len(self.metrics["mutual_benefit_score"]) > 20:
            window = min(30, len(episodes) // 10)
            avg_benefit = np.convolve(
                self.metrics["mutual_benefit_score"],
                np.ones(window) / window,
                mode="valid",
            )
            avg_episodes = range(window - 1, len(episodes))
            plt.plot(
                avg_episodes,
                avg_benefit,
                "--",
                linewidth=2,
                color="darkviolet",
                label="Moving Avg",
            )
        plt.xlabel("Episode")
        plt.ylabel("Mutual Benefit Score")
        plt.title("Mutual Benefit Evolution")
        plt.grid(True, alpha=0.3)

        # 7. Action correlation heatmap
        plt.subplot(4, 3, 7)
        correlations = [
            ep["metrics"]["action_correlation"]
            for ep in self.episode_logs
            if not np.isnan(ep["metrics"]["action_correlation"])
        ]
        if correlations:
            plt.plot(correlations, color="red", alpha=0.7)
            plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            plt.xlabel("Episode")
            plt.ylabel("Action Correlation")
            plt.title("Action Correlation Over Time")
            plt.grid(True, alpha=0.3)

        # 8. Behavioral consistency
        plt.subplot(4, 3, 8)
        consistency1 = [
            1.0 - ep["metrics"]["warmth_std_agent1"] for ep in self.episode_logs
        ]
        consistency2 = [
            1.0 - ep["metrics"]["warmth_std_agent2"] for ep in self.episode_logs
        ]
        plt.plot(consistency1, label="Agent 1", alpha=0.7)
        plt.plot(consistency2, label="Agent 2", alpha=0.7)
        plt.xlabel("Episode")
        plt.ylabel("Behavioral Consistency")
        plt.title("Behavioral Consistency Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 9. Reward fairness
        plt.subplot(4, 3, 9)
        fairness = [ep["metrics"]["reward_fairness"] for ep in self.episode_logs]
        plt.plot(fairness, color="orange", alpha=0.7)
        plt.xlabel("Episode")
        plt.ylabel("Reward Fairness")
        plt.title("Reward Fairness Over Time")
        plt.grid(True, alpha=0.3)

        # 10. Temperature evolution (if available)
        plt.subplot(4, 3, 10)
        for agent_id in ["agent1", "agent2"]:
            if self.metrics["learning_progress"][agent_id]:
                alphas = [
                    m["alpha"] for m in self.metrics["learning_progress"][agent_id]
                ]
                plt.plot(alphas, label=f"{agent_id} Temperature", alpha=0.7)
        plt.xlabel("Training Step")
        plt.ylabel("Temperature (Alpha)")
        plt.title("Temperature Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 11. Response pattern analysis
        plt.subplot(4, 3, 11)
        if self.response_patterns:
            responses1 = [
                rp["agent1_response_to_agent2"] for rp in self.response_patterns
            ]
            responses2 = [
                rp["agent2_response_to_agent1"] for rp in self.response_patterns
            ]
            plt.scatter(
                responses1[: min(1000, len(responses1))],
                responses2[: min(1000, len(responses2))],
                alpha=0.5,
            )
            plt.xlabel("Agent 1 Response to Agent 2")
            plt.ylabel("Agent 2 Response to Agent 1")
            plt.title("Response Pattern Correlation")
            plt.grid(True, alpha=0.3)

        # 12. Final performance summary
        plt.subplot(4, 3, 12)
        if len(self.metrics["episode_rewards"]["agent1"]) > 50:
            final_rewards1 = self.metrics["episode_rewards"]["agent1"][-50:]
            final_rewards2 = self.metrics["episode_rewards"]["agent2"][-50:]

            labels = ["Agent 1", "Agent 2", "Combined"]
            means = [
                np.mean(final_rewards1),
                np.mean(final_rewards2),
                np.mean(np.array(final_rewards1) + np.array(final_rewards2)),
            ]
            stds = [
                np.std(final_rewards1),
                np.std(final_rewards2),
                np.std(np.array(final_rewards1) + np.array(final_rewards2)),
            ]

            plt.bar(labels, means, yerr=stds, capsize=5, alpha=0.7)
            plt.ylabel("Average Reward (Last 50 Episodes)")
            plt.title("Final Performance Summary")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.experiment_dir, "comprehensive_dashboard.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(
            f" Comprehensive dashboard saved to {self.experiment_dir}/comprehensive_dashboard.png"
        )


# Integration helper for your main training script
class LoggingTrainerWrapper:
    """Wrapper to integrate comprehensive logging with your existing SACTrainer."""

    def __init__(self, trainer, experiment_name: str):
        self.trainer = trainer
        self.logger = ComprehensiveLogger(experiment_name)

    def train_with_logging(self, save_dir: str = "./models"):
        """Train with comprehensive logging enabled."""
        print(
            f"Starting training with comprehensive logging: {self.logger.experiment_name}"
        )

        # Monkey patch the original trainer to add logging
        original_train = self.trainer.train

        def logged_train(save_dir: str = "./models"):
            """Enhanced training method with comprehensive logging."""
            import os

            os.makedirs(save_dir, exist_ok=True)

            for episode in range(self.trainer.episodes_per_training):
                # Log episode start
                self.logger.log_episode_start(episode)

                # Reset environment
                state1, state2 = self.trainer.environment.reset()

                episode_reward1 = 0
                episode_reward2 = 0

                for step in range(self.trainer.steps_per_episode):
                    # Select actions
                    action1 = self.trainer.agent1.select_action(state1, training=True)
                    action2 = self.trainer.agent2.select_action(state2, training=True)

                    # Execute actions and get rewards
                    next_state1, next_state2, reward1, reward2, done = (
                        self.trainer.environment.step(action1, action2)
                    )

                    # Get environment stats if available
                    env_stats = {}
                    if hasattr(self.trainer.environment, "get_interaction_stats"):
                        env_stats = self.trainer.environment.get_interaction_stats()

                    # Log the step
                    self.logger.log_step(
                        step=step,
                        agent1_state=state1,
                        agent2_state=state2,
                        agent1_action=action1,
                        agent2_action=action2,
                        agent1_reward=reward1,
                        agent2_reward=reward2,
                        environment_stats=env_stats,
                    )

                    # Store transitions
                    self.trainer.agent1.store_transition(
                        state1, action1, reward1, next_state1, done
                    )
                    self.trainer.agent2.store_transition(
                        state2, action2, reward2, next_state2, done
                    )

                    # Update episode rewards
                    episode_reward1 += reward1
                    episode_reward2 += reward2

                    # Train agents
                    if self.trainer.step_count % self.trainer.training_frequency == 0:
                        metrics1 = self.trainer.agent1.train_step()
                        metrics2 = self.trainer.agent2.train_step()

                        # Log training metrics
                        if metrics1:
                            self.logger.log_training_step("agent1", metrics1)
                            self.trainer.training_metrics["agent1"].append(metrics1)
                        if metrics2:
                            self.logger.log_training_step("agent2", metrics2)
                            self.trainer.training_metrics["agent2"].append(metrics2)

                    # Update states
                    state1 = next_state1
                    state2 = next_state2
                    self.trainer.step_count += 1

                    if done:
                        break

                # Store episode rewards in trainer
                self.trainer.episode_rewards["agent1"].append(episode_reward1)
                self.trainer.episode_rewards["agent2"].append(episode_reward2)

                # Get final environment stats and log episode end
                final_stats = {}
                if hasattr(self.trainer.environment, "get_interaction_stats"):
                    final_stats = self.trainer.environment.get_interaction_stats()

                self.logger.log_episode_end(final_stats)

                # Evaluation and progress reporting
                if episode % self.trainer.evaluation_frequency == 0:
                    eval_results = self.trainer.evaluate(num_episodes=10)
                    print(
                        f"Episode {episode}: Agent1 Avg Reward: {eval_results['agent1_avg_reward']:.2f}, "
                        f"Agent2 Avg Reward: {eval_results['agent2_avg_reward']:.2f}"
                    )

                # Save models
                if episode % self.trainer.save_frequency == 0 and episode > 0:
                    self.trainer.agent1.save_model(
                        f"{save_dir}/agent1_episode_{episode}.pth"
                    )
                    self.trainer.agent2.save_model(
                        f"{save_dir}/agent2_episode_{episode}.pth"
                    )

            # Final save
            self.trainer.agent1.save_model(f"{save_dir}/agent1_final.pth")
            self.trainer.agent2.save_model(f"{save_dir}/agent2_final.pth")

            # Generate comprehensive report
            self.logger.generate_comprehensive_report()

            # Return enhanced results
            return {
                "episode_rewards": self.trainer.episode_rewards,
                "training_metrics": self.trainer.training_metrics,
                "total_episodes": self.trainer.episodes_per_training,
                "total_steps": self.trainer.step_count,
                "comprehensive_logs": self.logger.experiment_dir,
            }

        return logged_train(save_dir)
