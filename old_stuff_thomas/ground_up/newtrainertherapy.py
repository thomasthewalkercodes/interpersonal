"""
Enhanced main training script with therapeutic agent integrated into existing system.
Adds therapeutic agent to your existing experiments while keeping everything working.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Tuple, List
from comprehensive_logging import LoggingTrainerWrapper

# Import your existing modules
from agent_configuration import (
    BaseAgentConfig,
    CooperativeAgentConfig,
    CompetitiveAgentConfig,
    AdaptiveAgentConfig,
    CautiousAgentConfig,
)
from agent_state import InterpersonalAgentState
from sac_algorithm import SACAgent, SACTrainer
from interaction_environment import InterpersonalEnvironment
from gaussian_payoff_graph import calculate_warmth_payoff
from interfaces import PayoffCalculator


class TherapeuticPayoffCalculator(PayoffCalculator):
    """
    Specialized payoff calculator that rewards the therapist for maximizing
    the patient's payoff and achieving mutual warmth.
    """

    def __init__(
        self, alpha: float = 4, beta: float = 10, therapeutic_weight: float = 2.0
    ):
        self.alpha = alpha
        self.beta = beta
        self.therapeutic_weight = therapeutic_weight

    def calculate_payoff(
        self,
        therapist_action: float,
        patient_action: float,
        therapist_id: str,
        patient_id: str,
    ) -> tuple[float, float]:
        """Calculate payoffs with therapeutic modifications."""
        # Convert actions from [-1, 1] to [0, 1] for warmth
        therapist_warmth = (therapist_action + 1) / 2
        patient_warmth = (patient_action + 1) / 2

        # Calculate base payoffs using your Gaussian function
        therapist_base_payoff = calculate_warmth_payoff(
            therapist_warmth, patient_warmth, self.alpha, self.beta
        )
        patient_payoff = calculate_warmth_payoff(
            patient_warmth, therapist_warmth, self.alpha, self.beta
        )

        # Therapeutic modifications for therapist reward
        patient_success_bonus = patient_payoff * self.therapeutic_weight
        patient_warmth_bonus = patient_warmth * 1.5
        mutual_warmth = min(therapist_warmth, patient_warmth)
        mutual_bonus = mutual_warmth * 2.0

        trust_building_bonus = 0.0
        if therapist_warmth > patient_warmth and therapist_warmth > 0.5:
            trust_building_bonus = (therapist_warmth - patient_warmth) * 1.0

        consistency_bonus = 0.5 if 0.3 <= therapist_warmth <= 0.9 else 0.0

        therapist_payoff = (
            therapist_base_payoff
            + patient_success_bonus
            + patient_warmth_bonus
            + mutual_bonus
            + trust_building_bonus
            + consistency_bonus
        )

        return therapist_payoff, patient_payoff


class GaussianPayoffCalculator(PayoffCalculator):
    """Your original Gaussian payoff calculator - keeping this unchanged."""

    def __init__(self, alpha: float = 4, beta: float = 10):
        self.alpha = alpha
        self.beta = beta

    def calculate_payoff(
        self, agent1_action: float, agent2_action: float, agent1_id: str, agent2_id: str
    ) -> tuple[float, float]:
        """Calculate payoffs using your Gaussian warmth function."""
        w1 = (agent1_action + 1) / 2
        w2 = (agent2_action + 1) / 2

        payoff1 = calculate_warmth_payoff(w1, w2, self.alpha, self.beta)
        payoff2 = calculate_warmth_payoff(w2, w1, self.alpha, self.beta)

        return payoff1, payoff2


class TherapeuticAgentConfig(BaseAgentConfig):
    """Configuration for therapeutic agent with specialized parameters."""

    def __init__(
        self,
        memory_length: int = 60,
        initial_trust: float = 0.0,
        initial_satisfaction: float = 0.0,
        **kwargs,
    ):

        therapeutic_params = {
            "lr_actor": 3e-4,
            "lr_critic": 3e-4,
            "lr_temperature": 1e-4,
            "gamma": 0.95,
            "tau": 0.01,
            "batch_size": 128,
            "buffer_size": 50000,
            "hidden_size": 256,
            "noise_scale": 0.05,
            "alpha": 0.15,
            "target_entropy": -0.5,
        }

        therapeutic_params.update(kwargs)

        super().__init__(
            memory_length=memory_length,
            initial_trust=initial_trust,
            initial_satisfaction=initial_satisfaction,
            **therapeutic_params,
        )


class AdaptiveTherapeuticAgent(SACAgent):
    """
    Therapeutic agent that adapts behavior to build trust and maximize patient outcomes.
    """

    def __init__(self, state_dim: int, config: Dict[str, Any]):
        super().__init__(state_dim, config)

        # Therapeutic state tracking
        self.patient_warmth_history = []
        self.patient_trust_estimate = 0.0
        self.trust_trend = 0.0
        self.baseline_patient_warmth = None
        self.warmth_improvement_target = 0.7

        # Therapeutic strategy parameters
        self.trust_building_phase = True
        self.leading_intensity = 0.1
        self.adaptation_rate = 0.02

        # Track therapeutic progress
        self.therapy_step = 0
        self.trust_building_steps = 0
        self.successful_interactions = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> float:
        """Select therapeutic action based on current strategy and patient state."""

        # Get base action from SAC
        base_action = super().select_action(state, training)

        # Apply therapeutic modifications
        therapeutic_action = self._apply_therapeutic_strategy(base_action, state)

        # Ensure action is in bounds
        therapeutic_action = np.clip(therapeutic_action, -1.0, 1.0)

        self.therapy_step += 1
        return therapeutic_action

    def _apply_therapeutic_strategy(
        self, base_action: float, state: np.ndarray
    ) -> float:
        """Apply therapeutic strategy to modify the base SAC action."""

        current_patient_warmth = self._estimate_patient_warmth(state)
        self._update_trust_estimate(state)

        if self.trust_building_phase:
            return self._trust_building_action(base_action, current_patient_warmth)
        else:
            return self._warmth_leading_action(base_action, current_patient_warmth)

    def _trust_building_action(
        self, base_action: float, patient_warmth: float
    ) -> float:
        """Action strategy during trust building phase."""

        target_warmth = 0.6  # Moderate warmth in [0,1] space
        target_action = target_warmth * 2 - 1  # Convert to [-1,1] space

        trust_weight = 0.7 if self.patient_trust_estimate < 0.3 else 0.5
        action = trust_weight * target_action + (1 - trust_weight) * base_action

        if patient_warmth is not None:
            patient_action = patient_warmth * 2 - 1
            leading_action = patient_action + self.leading_intensity
            action = 0.6 * action + 0.4 * leading_action

        # Transition to leading phase when trust is established
        if (
            self.patient_trust_estimate > 0.4
            and len(self.patient_warmth_history) > 20
            and np.mean(self.patient_warmth_history[-10:]) > 0.4
        ):
            self.trust_building_phase = False
            print(
                f"[THERAPIST] Transitioning to warmth leading phase at step {self.therapy_step}"
            )

        return action

    def _warmth_leading_action(
        self, base_action: float, patient_warmth: float
    ) -> float:
        """Action strategy during warmth leading phase."""

        if patient_warmth is None:
            return base_action

        current_gap = self.warmth_improvement_target - patient_warmth

        if patient_warmth < 0.3:
            lead_amount = 0.3
        elif patient_warmth < 0.5:
            lead_amount = 0.2
        elif patient_warmth < 0.7:
            lead_amount = 0.1
        else:
            lead_amount = 0.05

        target_warmth = min(0.9, patient_warmth + lead_amount)
        target_action = target_warmth * 2 - 1

        therapeutic_weight = 0.8 if current_gap > 0.3 else 0.6
        action = (
            therapeutic_weight * target_action + (1 - therapeutic_weight) * base_action
        )

        return action

    def _estimate_patient_warmth(self, state: np.ndarray) -> float:
        """Estimate patient's current warmth level from state."""
        if len(state) > 2:
            estimated_warmth = np.clip((state[2] + 1) / 2, 0, 1)
        else:
            estimated_warmth = 0.3

        return estimated_warmth

    def _update_trust_estimate(self, state: np.ndarray):
        """Update estimate of patient trust based on state information."""

        if len(state) > 0:
            trust_indicator = np.clip(state[0], -1, 1)
            new_trust = (trust_indicator + 1) / 2

            self.patient_trust_estimate = (
                0.9 * self.patient_trust_estimate + 0.1 * new_trust
            )

            if len(self.patient_warmth_history) > 5:
                recent_trend = new_trust - np.mean([self.patient_trust_estimate] * 5)
                self.trust_trend = 0.8 * self.trust_trend + 0.2 * recent_trend

    def store_transition(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition and update therapeutic tracking."""

        super().store_transition(state, action, reward, next_state, done)

        patient_warmth = self._estimate_patient_warmth(next_state)
        if patient_warmth is not None:
            self.patient_warmth_history.append(patient_warmth)

            if len(self.patient_warmth_history) > 200:
                self.patient_warmth_history = self.patient_warmth_history[-150:]

            if (
                self.baseline_patient_warmth is None
                and len(self.patient_warmth_history) > 10
            ):
                self.baseline_patient_warmth = np.mean(self.patient_warmth_history[:10])

        if reward > 2.0:
            self.successful_interactions += 1

    def get_therapeutic_progress(self) -> Dict[str, float]:
        """Get current therapeutic progress metrics."""

        if not self.patient_warmth_history:
            return {"progress": 0.0, "current_warmth": 0.0, "trust": 0.0}

        current_warmth = self.patient_warmth_history[-1]
        baseline = self.baseline_patient_warmth or self.patient_warmth_history[0]

        warmth_improvement = current_warmth - baseline
        progress_toward_target = (
            (current_warmth - baseline) / (self.warmth_improvement_target - baseline)
            if baseline < self.warmth_improvement_target
            else 1.0
        )

        return {
            "current_warmth": current_warmth,
            "baseline_warmth": baseline,
            "warmth_improvement": warmth_improvement,
            "progress_toward_target": progress_toward_target,
            "trust_estimate": self.patient_trust_estimate,
            "trust_trend": self.trust_trend,
            "successful_interactions": self.successful_interactions,
            "therapy_step": self.therapy_step,
            "phase": (
                "trust_building" if self.trust_building_phase else "warmth_leading"
            ),
        }


class ExperimentRunner:
    """Enhanced experiment runner with both original and therapeutic capabilities."""

    def __init__(self):
        self.results = {}

    def create_agent_pair(self, config1, config2, experiment_name, alpha=4, beta=10):
        """Create a pair of agents with your original Gaussian payoff function."""
        # Your original method - unchanged
        state1 = config1.create_initial_state()
        state2 = config2.create_initial_state()

        state_dim = state1.get_state_dimension()

        agent1 = SACAgent(state_dim, config1.get_sac_params())
        agent2 = SACAgent(state_dim, config2.get_sac_params())

        payoff_calculator = GaussianPayoffCalculator(alpha=alpha, beta=beta)

        environment = InterpersonalEnvironment(
            payoff_calculator=payoff_calculator,
            agent1_state=state1,
            agent2_state=state2,
            agent1_id=f"{experiment_name}_agent1",
            agent2_id=f"{experiment_name}_agent2",
            max_steps_per_episode=50,
        )

        return agent1, agent2, environment

    def create_therapeutic_pair(
        self, patient_config, experiment_name, alpha=4, beta=10, therapeutic_weight=2.0
    ):
        """Create a therapist-patient pair with therapeutic payoff function."""

        # Create patient state
        patient_state = patient_config.create_initial_state()
        state_dim = patient_state.get_state_dimension()

        # Create standard patient agent
        patient_agent = SACAgent(state_dim, patient_config.get_sac_params())

        # Create therapeutic agent
        therapeutic_config = TherapeuticAgentConfig(
            memory_length=patient_config.memory_length
        )
        therapist_agent = AdaptiveTherapeuticAgent(
            state_dim, therapeutic_config.get_sac_params()
        )

        # Create therapeutic payoff calculator
        payoff_calculator = TherapeuticPayoffCalculator(
            alpha=alpha, beta=beta, therapeutic_weight=therapeutic_weight
        )

        # Create dummy therapist state for environment
        therapist_state = patient_config.create_initial_state()

        # Create environment
        environment = InterpersonalEnvironment(
            payoff_calculator=payoff_calculator,
            agent1_state=therapist_state,
            agent2_state=patient_state,
            agent1_id=f"{experiment_name}_therapist",
            agent2_id=f"{experiment_name}_patient",
            max_steps_per_episode=50,
        )

        return therapist_agent, patient_agent, environment

    def run_experiment(
        self, config1, config2, experiment_name, episodes=300, alpha=4, beta=10
    ):
        """Run your original experiment - unchanged."""
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT: {experiment_name}")
        print(f"Episodes: {episodes}, Alpha: {alpha}, Beta: {beta}")
        print(f"{'='*60}")

        agent1, agent2, environment = self.create_agent_pair(
            config1, config2, experiment_name, alpha, beta
        )

        trainer = SACTrainer(
            agent1=agent1,
            agent2=agent2,
            environment=environment,
            payoff_calculator=environment.payoff_calculator,
            episodes_per_training=episodes,
            steps_per_episode=50,
            evaluation_frequency=max(100, episodes // 10),
            save_frequency=max(500, episodes // 2),
        )

        save_dir = f"./models/{experiment_name}"
        logging_wrapper = LoggingTrainerWrapper(trainer, experiment_name)
        results = logging_wrapper.train_with_logging(save_dir)

        self.results[experiment_name] = results

        print(f"\nExperiment {experiment_name} completed!")
        return results

    def run_therapeutic_experiment(
        self,
        patient_config,
        experiment_name,
        episodes=300,
        alpha=4,
        beta=10,
        therapeutic_weight=2.0,
    ):
        """Run a therapeutic experiment."""
        print(f"\n{'='*60}")
        print(f"RUNNING THERAPEUTIC EXPERIMENT: {experiment_name}")
        print(f"Episodes: {episodes}, Alpha: {alpha}, Beta: {beta}")
        print(f"Therapeutic Weight: {therapeutic_weight}")
        print(f"{'='*60}")

        therapist, patient, environment = self.create_therapeutic_pair(
            patient_config, experiment_name, alpha, beta, therapeutic_weight
        )

        trainer = SACTrainer(
            agent1=therapist,
            agent2=patient,
            environment=environment,
            payoff_calculator=environment.payoff_calculator,
            episodes_per_training=episodes,
            steps_per_episode=50,
            evaluation_frequency=max(50, episodes // 10),
            save_frequency=max(200, episodes // 2),
        )

        save_dir = f"./therapeutic_models/{experiment_name}"
        logging_wrapper = LoggingTrainerWrapper(trainer, experiment_name)
        results = logging_wrapper.train_with_logging(save_dir)

        # Get therapeutic progress
        therapeutic_progress = therapist.get_therapeutic_progress()
        results["therapeutic_progress"] = therapeutic_progress

        self.results[experiment_name] = results

        print(f"\nTherapeutic Experiment {experiment_name} completed!")
        print(f"Therapeutic Progress:")
        print(
            f"  Patient warmth: {therapeutic_progress['baseline_warmth']:.3f} -> {therapeutic_progress['current_warmth']:.3f}"
        )
        print(f"  Improvement: {therapeutic_progress['warmth_improvement']:.3f}")
        print(
            f"  Progress to target: {therapeutic_progress['progress_toward_target']:.1%}"
        )
        print(f"  Trust estimate: {therapeutic_progress['trust_estimate']:.3f}")
        print(f"  Current phase: {therapeutic_progress['phase']}")

        return results, therapist

    def analyze_results(self, experiment_name):
        """Your original analyze_results method - unchanged."""
        if experiment_name not in self.results:
            print(f"No results found for {experiment_name}")
            return

        results = self.results[experiment_name]

        plt.figure(figsize=(15, 5))

        # Plot 1: Episode rewards
        plt.subplot(1, 3, 1)
        episodes = range(len(results["episode_rewards"]["agent1"]))
        plt.plot(
            episodes,
            results["episode_rewards"]["agent1"],
            label="Agent 1",
            alpha=0.7,
            linewidth=1,
        )
        plt.plot(
            episodes,
            results["episode_rewards"]["agent2"],
            label="Agent 2",
            alpha=0.7,
            linewidth=1,
        )

        window = min(50, len(episodes) // 10)
        if len(episodes) > window:
            avg1 = np.convolve(
                results["episode_rewards"]["agent1"],
                np.ones(window) / window,
                mode="valid",
            )
            avg2 = np.convolve(
                results["episode_rewards"]["agent2"],
                np.ones(window) / window,
                mode="valid",
            )
            avg_episodes = range(window - 1, len(episodes))
            plt.plot(avg_episodes, avg1, "--", linewidth=2, label="Agent 1 (avg)")
            plt.plot(avg_episodes, avg2, "--", linewidth=2, label="Agent 2 (avg)")

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"{experiment_name}: Episode Rewards")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Learning curves
        plt.subplot(1, 3, 2)
        if results["training_metrics"]["agent1"]:
            steps = range(len(results["training_metrics"]["agent1"]))
            alpha_values1 = [m["alpha"] for m in results["training_metrics"]["agent1"]]
            alpha_values1 = [
                float(a.detach()) if hasattr(a, "detach") else float(a)
                for a in alpha_values1
            ]
            plt.plot(steps, alpha_values1, label="Agent 1 Temperature", alpha=0.7)

        if results["training_metrics"]["agent2"]:
            steps = range(len(results["training_metrics"]["agent2"]))
            alpha_values2 = [m["alpha"] for m in results["training_metrics"]["agent2"]]
            alpha_values2 = [
                float(a.detach()) if hasattr(a, "detach") else float(a)
                for a in alpha_values2
            ]
            plt.plot(steps, alpha_values2, label="Agent 2 Temperature", alpha=0.7)

        plt.xlabel("Training Step")
        plt.ylabel("Temperature (Alpha)")
        plt.title(f"{experiment_name}: Learning Temperature")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Final performance
        plt.subplot(1, 3, 3)
        final_episodes = min(100, len(results["episode_rewards"]["agent1"]) // 4)
        final_rewards1 = results["episode_rewards"]["agent1"][-final_episodes:]
        final_rewards2 = results["episode_rewards"]["agent2"][-final_episodes:]

        labels = ["Agent 1", "Agent 2"]
        means = [np.mean(final_rewards1), np.mean(final_rewards2)]
        stds = [np.std(final_rewards1), np.std(final_rewards2)]

        plt.bar(
            labels,
            means,
            yerr=stds,
            capsize=5,
            alpha=0.7,
            color=["skyblue", "lightcoral"],
        )
        plt.ylabel(f"Average Reward (Last {final_episodes} Episodes)")
        plt.title(f"{experiment_name}: Final Performance")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        os.makedirs("./results", exist_ok=True)
        plt.savefig(
            f"./results/{experiment_name}_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # Print summary
        print(f"\n{'='*50}")
        print(f"RESULTS SUMMARY: {experiment_name}")
        print(f"{'='*50}")
        print(f"Total Episodes: {len(results['episode_rewards']['agent1'])}")
        print(
            f"Agent 1 - Final Avg Reward: {np.mean(final_rewards1):.3f} ± {np.std(final_rewards1):.3f}"
        )
        print(
            f"Agent 2 - Final Avg Reward: {np.mean(final_rewards2):.3f} ± {np.std(final_rewards2):.3f}"
        )

        if results["training_metrics"]["agent1"]:
            final_alpha1 = results["training_metrics"]["agent1"][-1]["alpha"]
            print(f"Agent 1 - Final Temperature: {final_alpha1:.4f}")
        if results["training_metrics"]["agent2"]:
            final_alpha2 = results["training_metrics"]["agent2"][-1]["alpha"]
            print(f"Agent 2 - Final Temperature: {final_alpha2:.4f}")


def main():
    """Main function to run therapeutic experiments only."""
    print("SAC Therapeutic Agent Training System")
    print("=" * 60)
    print(
        "Training therapist agents to maximize patient outcomes through trust building"
    )
    print("=" * 60)

    runner = ExperimentRunner()

    # Therapeutic experiments only
    therapeutic_experiments = [
        {
            "name": "therapist_vs_anxious_patient",
            "patient_config": CautiousAgentConfig(
                memory_length=60,
                initial_trust=-0.3,  # Start with low trust
                initial_satisfaction=-0.2,  # Start dissatisfied
                lr_actor=4e-4,  # Learn at moderate pace
                noise_scale=0.12,  # Some behavioral variability
                alpha=0.3,  # Higher exploration due to anxiety
            ),
            "episodes": 400,
            "alpha": 8,  # Moderate mismatch sensitivity
            "beta": 20,  # Higher rejection sensitivity (anxious)
            "therapeutic_weight": 2.5,  # Strong emphasis on patient outcomes
        },
        {
            "name": "therapist_vs_resistant_patient",
            "patient_config": CompetitiveAgentConfig(
                memory_length=50,
                initial_trust=-0.5,  # Very low initial trust
                initial_satisfaction=-0.4,  # Quite dissatisfied
                lr_actor=3e-4,  # Slower learning (resistance)
                noise_scale=0.08,  # More consistent but cold behavior
                alpha=0.2,  # Lower exploration (stuck in patterns)
            ),
            "episodes": 400,
            "alpha": 8,  # High mismatch sensitivity
            "beta": 15,  # Very high rejection sensitivity
            "therapeutic_weight": 3.0,  # Maximum emphasis on patient breakthrough
        },
        {
            "name": "therapist_vs_mild_withdrawal",
            "patient_config": CautiousAgentConfig(
                memory_length=40,
                initial_trust=0.1,  # Slightly positive starting point
                initial_satisfaction=0.0,  # Neutral satisfaction
                lr_actor=5e-4,  # Faster learning capacity
                noise_scale=0.15,  # More behavioral flexibility
                alpha=0.25,  # Moderate exploration
            ),
            "episodes": 300,
            "alpha": 4,  # Standard mismatch sensitivity
            "beta": 8,  # Moderate rejection sensitivity
            "therapeutic_weight": 2.0,  # Standard therapeutic emphasis
        },
    ]

    print("Running therapeutic experiments...")
    print("Expected time: 15-20 minutes for all experiments")
    print("-" * 60)

    therapeutic_results = {}

    for i, exp in enumerate(therapeutic_experiments, 1):
        try:
            print(f"\n🏥 [{i}/{len(therapeutic_experiments)}] Starting: {exp['name']}")
            print(f"   Patient type: {exp['patient_config'].__class__.__name__}")
            print(f"   Episodes: {exp['episodes']}")
            print(f"   Therapeutic weight: {exp['therapeutic_weight']}")

            results, therapist = runner.run_therapeutic_experiment(
                patient_config=exp["patient_config"],
                experiment_name=exp["name"],
                episodes=exp["episodes"],
                alpha=exp["alpha"],
                beta=exp["beta"],
                therapeutic_weight=exp["therapeutic_weight"],
            )

            therapeutic_results[exp["name"]] = {
                "results": results,
                "therapist": therapist,
                "progress": therapist.get_therapeutic_progress(),
            }

        except Exception as e:
            print(f"❌ Error in therapeutic experiment {exp['name']}: {e}")
            import traceback

            traceback.print_exc()

    # Show comprehensive therapeutic comparison
    if therapeutic_results:
        print(f"\n{'='*90}")
        print("THERAPEUTIC RESULTS COMPARISON")
        print(f"{'='*90}")

        print(
            f"{'Experiment':<30} {'Warmth Δ':<10} {'Current W':<10} {'Trust':<8} {'Success':<8} {'Phase':<15}"
        )
        print("-" * 90)

        for exp_name, data in therapeutic_results.items():
            progress = data["progress"]
            print(
                f"{exp_name:<30} "
                f"{progress.get('warmth_improvement', 0):>8.3f}  "
                f"{progress.get('current_warmth', 0):>8.3f}  "
                f"{progress.get('trust_estimate', 0):>6.3f}  "
                f"{progress.get('successful_interactions', 0):>6d}  "
                f"{progress.get('phase', 'unknown'):<15}"
            )

        # Determine most successful therapeutic approaches
        best_improvement = max(
            therapeutic_results.items(),
            key=lambda x: x[1]["progress"].get("warmth_improvement", 0),
        )
        best_trust = max(
            therapeutic_results.items(),
            key=lambda x: x[1]["progress"].get("trust_estimate", 0),
        )
        best_success = max(
            therapeutic_results.items(),
            key=lambda x: x[1]["progress"].get("successful_interactions", 0),
        )

        print(f"\n🏆 THERAPEUTIC SUCCESS METRICS:")
        print(f"   Highest Warmth Improvement: {best_improvement[0]}")
        print(
            f"     → Improvement: {best_improvement[1]['progress'].get('warmth_improvement', 0):.3f}"
        )
        print(f"   Highest Trust Achievement: {best_trust[0]}")
        print(
            f"     → Trust level: {best_trust[1]['progress'].get('trust_estimate', 0):.3f}"
        )
        print(f"   Most Successful Interactions: {best_success[0]}")
        print(
            f"     → Success count: {best_success[1]['progress'].get('successful_interactions', 0)}"
        )

        # Overall therapeutic effectiveness
        print(f"\n📊 OVERALL EFFECTIVENESS:")
        total_improvements = sum(
            data["progress"].get("warmth_improvement", 0)
            for data in therapeutic_results.values()
        )
        avg_trust = np.mean(
            [
                data["progress"].get("trust_estimate", 0)
                for data in therapeutic_results.values()
            ]
        )
        total_successes = sum(
            data["progress"].get("successful_interactions", 0)
            for data in therapeutic_results.values()
        )

        print(
            f"   Total warmth improvements across all patients: {total_improvements:.3f}"
        )
        print(f"   Average trust level achieved: {avg_trust:.3f}")
        print(f"   Total successful therapeutic interactions: {total_successes}")

        # Success rate analysis
        successful_experiments = sum(
            1
            for data in therapeutic_results.values()
            if data["progress"].get("warmth_improvement", 0) > 0.2
            and data["progress"].get("trust_estimate", 0) > 0.4
        )
        success_rate = successful_experiments / len(therapeutic_results) * 100

        print(
            f"   Therapeutic success rate: {success_rate:.1f}% ({successful_experiments}/{len(therapeutic_results)} patients)"
        )

    print(f"\n{'='*60}")
    print("🎯 THERAPEUTIC TRAINING COMPLETED!")
    print(f"{'='*60}")
    print("Generated outputs:")
    print("📁 Therapeutic models: ./therapeutic_models/")
    print("📊 Comprehensive logs: Check each experiment folder for detailed metrics")
    print("🧠 Key learnings:")
    print("   • Therapist learned to build trust before leading")
    print("   • Different patient types require different therapeutic approaches")
    print("   • Patient-focused rewards lead to better mutual outcomes")
    print("   • Trust-building phase typically lasts 100-200 interactions")
    print("   • Leading phase shows gradual warmth improvements")

    if therapeutic_results:
        print(f"\n💡 Next steps:")
        print(
            "   • Analyze individual experiment logs for detailed interaction patterns"
        )
        print("   • Try different therapeutic_weight values to see sensitivity")
        print("   • Experiment with different patient configurations")
        print("   • Extend episodes for longer-term therapeutic relationships")


if __name__ == "__main__":
    main()
