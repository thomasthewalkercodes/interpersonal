# parameter_sweep.py (place in project root)
"""
Comprehensive parameter sweep and testing script for continuous Bayesian agents.
Allows testing different combinations of:
- Agent personalities
- Payoff functions
- Learning parameters
- Action selection methods
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from typing import Dict, List, Any, Tuple

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from continuous_bayesian_agent import create_continuous_bayesian_agent
from continuous_simulation import run_multiple_continuous_simulations
from payoff_functions import (
    gaussian_matching_payoff,
    create_personality_payoff_function,
    symmetric_gaussian_payoff,
    interpersonal_gaussian_payoff,
)


class ParameterSweep:
    """
    Class for running comprehensive parameter sweeps and analysis
    """

    def __init__(self, output_dir: str = "sweep_results"):
        self.output_dir = output_dir
        self.results_df = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)

        print(f"Parameter Sweep initialized. Results will be saved to: {output_dir}")

    def define_agent_configurations(self) -> Dict[str, Dict[str, Any]]:
        """
        Define different agent personality configurations
        """

        agent_configs = {
            # Basic personalities
            "secure": {
                "prior_mean": 0.7,
                "prior_confidence": 2.0,
                "prior_variance": 0.05,
                "lambda_loss": 1.0,
                "temperature": 1.0,
            },
            "anxious": {
                "prior_mean": 0.4,
                "prior_confidence": 3.0,
                "prior_variance": 0.1,
                "lambda_loss": 2.5,
                "temperature": 1.2,
            },
            "avoidant": {
                "prior_mean": 0.3,
                "prior_confidence": 4.0,
                "prior_variance": 0.03,
                "lambda_loss": 1.5,
                "temperature": 0.8,
            },
            "neutral": {
                "prior_mean": 0.5,
                "prior_confidence": 1.0,
                "prior_variance": 0.1,
                "lambda_loss": 1.0,
                "temperature": 1.0,
            },
            # Extreme personalities for testing
            "very_optimistic": {
                "prior_mean": 0.9,
                "prior_confidence": 5.0,
                "prior_variance": 0.02,
                "lambda_loss": 0.8,
                "temperature": 0.6,
            },
            "very_pessimistic": {
                "prior_mean": 0.1,
                "prior_confidence": 5.0,
                "prior_variance": 0.02,
                "lambda_loss": 3.0,
                "temperature": 1.5,
            },
            # High uncertainty personality
            "uncertain": {
                "prior_mean": 0.5,
                "prior_confidence": 0.5,  # Very uncertain
                "prior_variance": 0.2,  # Expects high variance
                "lambda_loss": 2.0,
                "temperature": 2.0,  # High exploration
            },
            # Rigid personality
            "rigid": {
                "prior_mean": 0.6,
                "prior_confidence": 10.0,  # Very confident
                "prior_variance": 0.01,  # Expects consistency
                "lambda_loss": 1.2,
                "temperature": 0.3,  # Low exploration
            },
        }

        return agent_configs

    def define_payoff_functions(self) -> Dict[str, Any]:
        """
        Define different payoff function types
        """

        payoff_functions = {
            # Your main Gaussian matching function
            "gaussian_matching": lambda my_action, opp_action: gaussian_matching_payoff(
                my_action,
                opp_action,
                peak_payoff=10.0,
                mismatch_penalty=-5.0,
                matching_bonus=3.0,
                falloff_rate=2.0,
            ),
            # Symmetric version
            "symmetric_gaussian": lambda my_action, opp_action: symmetric_gaussian_payoff(
                my_action, opp_action
            ),
            # Personality-based payoff functions
            "secure_payoff": create_personality_payoff_function("secure"),
            "anxious_payoff": create_personality_payoff_function("anxious"),
            "avoidant_payoff": create_personality_payoff_function("avoidant"),
            # Custom variations for testing
            "high_mismatch_penalty": lambda my_action, opp_action: gaussian_matching_payoff(
                my_action,
                opp_action,
                peak_payoff=10.0,
                mismatch_penalty=-10.0,  # Higher penalty
                matching_bonus=2.0,
                falloff_rate=3.0,
            ),
            "low_mismatch_penalty": lambda my_action, opp_action: gaussian_matching_payoff(
                my_action,
                opp_action,
                peak_payoff=8.0,
                mismatch_penalty=-2.0,  # Lower penalty
                matching_bonus=4.0,
                falloff_rate=1.0,
            ),
            # Interpersonal preference function
            "interpersonal_pref": lambda my_action, opp_action: interpersonal_gaussian_payoff(
                my_action,
                opp_action,
                my_ideal_warmth=0.7,
                ideal_opponent_warmth=0.6,
                tolerance=0.2,
            ),
        }

        return payoff_functions

    def run_single_parameter_sweep(
        self,
        sweep_params: Dict[str, Any],
        n_simulations: int = 20,
        n_rounds: int = 300,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run parameter sweep for given parameters
        """

        agent_configs = self.define_agent_configurations()
        payoff_functions = self.define_payoff_functions()

        # Extract sweep parameters
        agent1_types = sweep_params.get("agent1_types", list(agent_configs.keys())[:3])
        agent2_types = sweep_params.get("agent2_types", list(agent_configs.keys())[:3])
        payoff1_types = sweep_params.get(
            "payoff1_types", list(payoff_functions.keys())[:3]
        )
        payoff2_types = sweep_params.get(
            "payoff2_types", list(payoff_functions.keys())[:3]
        )
        action_methods = sweep_params.get("action_methods", ["thompson", "ucb"])

        # Generate all combinations
        combinations = list(
            product(
                agent1_types, agent2_types, payoff1_types, payoff2_types, action_methods
            )
        )

        print(f"Running parameter sweep: {len(combinations)} combinations")
        print(f"Each combination: {n_simulations} simulations × {n_rounds} rounds")

        results = []

        for i, (a1_type, a2_type, p1_type, p2_type, method) in enumerate(combinations):

            if verbose:
                print(
                    f"[{i+1}/{len(combinations)}] Testing: {a1_type} vs {a2_type} | "
                    f"{p1_type} vs {p2_type} | {method}"
                )

            try:
                # Get configurations
                a1_config = agent_configs[a1_type].copy()
                a2_config = agent_configs[a2_type].copy()
                a1_config["agent_id"] = f"{a1_type}_agent"
                a2_config["agent_id"] = f"{a2_type}_agent"

                p1_func = payoff_functions[p1_type]
                p2_func = payoff_functions[p2_type]

                # Run multiple simulations
                sim_results = run_multiple_continuous_simulations(
                    agent1_params=a1_config,
                    agent2_params=a2_config,
                    payoff_function1=p1_func,
                    payoff_function2=p2_func,
                    n_simulations=n_simulations,
                    n_rounds=n_rounds,
                    action_selection_method=method,
                    verbose=False,
                )

                # Calculate summary statistics
                summary = {
                    "agent1_type": a1_type,
                    "agent2_type": a2_type,
                    "payoff1_type": p1_type,
                    "payoff2_type": p2_type,
                    "action_method": method,
                    "n_simulations": n_simulations,
                    "n_rounds": n_rounds,
                    # Action statistics
                    "mean_final_warmth": (
                        sim_results["agent1_mean_action"].mean()
                        + sim_results["agent2_mean_action"].mean()
                    )
                    / 2,
                    "std_final_warmth": np.std(
                        [
                            sim_results["agent1_mean_action"].mean(),
                            sim_results["agent2_mean_action"].mean(),
                        ]
                    ),
                    "mean_action_difference": sim_results[
                        "mean_action_difference"
                    ].mean(),
                    "std_action_difference": sim_results[
                        "mean_action_difference"
                    ].std(),
                    # Convergence statistics
                    "convergence_rate": np.mean(
                        sim_results["final_action_difference"] < 0.2
                    ),
                    "high_cooperation_rate": np.mean(
                        sim_results["agent1_mean_action"] > 0.6
                    )
                    * np.mean(sim_results["agent2_mean_action"] > 0.6),
                    # Payoff statistics
                    "mean_total_payoff": (
                        sim_results["agent1_total_payoff"].mean()
                        + sim_results["agent2_total_payoff"].mean()
                    ),
                    "payoff_balance": abs(
                        sim_results["agent1_total_payoff"].mean()
                        - sim_results["agent2_total_payoff"].mean()
                    ),
                    # Learning statistics
                    "mean_belief_accuracy_1": 1
                    - abs(
                        sim_results["agent1_final_belief"]
                        - sim_results["agent2_mean_action"]
                    ).mean(),
                    "mean_belief_accuracy_2": 1
                    - abs(
                        sim_results["agent2_final_belief"]
                        - sim_results["agent1_mean_action"]
                    ).mean(),
                    "action_correlation": sim_results["action_correlation"].mean(),
                }

                results.append(summary)

            except Exception as e:
                print(f"  ✗ Error in combination {i+1}: {e}")
                continue

        results_df = pd.DataFrame(results)
        return results_df

    def run_parameter_sensitivity_analysis(
        self,
        base_personality: str = "secure",
        parameter_ranges: Dict[str, List[float]] = None,
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis for specific parameters
        """

        if parameter_ranges is None:
            parameter_ranges = {
                "prior_mean": [0.2, 0.4, 0.6, 0.8],
                "prior_confidence": [0.5, 1.0, 2.0, 4.0],
                "lambda_loss": [0.5, 1.0, 1.5, 2.0, 3.0],
            }

        base_config = self.define_agent_configurations()[base_personality]
        payoff_func = gaussian_matching_payoff

        results = []

        for param_name, param_values in parameter_ranges.items():
            print(f"\nTesting sensitivity of {param_name}...")

            for param_value in param_values:
                print(f"  {param_name} = {param_value}")

                # Create modified configuration
                config1 = base_config.copy()
                config2 = base_config.copy()
                config1[param_name] = param_value
                config1["agent_id"] = f"test_agent_1"
                config2["agent_id"] = f"test_agent_2"

                try:
                    sim_results = run_multiple_continuous_simulations(
                        agent1_params=config1,
                        agent2_params=config2,
                        payoff_function1=payoff_func,
                        payoff_function2=payoff_func,
                        n_simulations=15,
                        n_rounds=200,
                        verbose=False,
                    )

                    summary = {
                        "parameter": param_name,
                        "parameter_value": param_value,
                        "base_personality": base_personality,
                        "convergence_rate": np.mean(
                            sim_results["final_action_difference"] < 0.2
                        ),
                        "mean_final_warmth": (
                            sim_results["agent1_mean_action"].mean()
                            + sim_results["agent2_mean_action"].mean()
                        )
                        / 2,
                        "mean_total_payoff": (
                            sim_results["agent1_total_payoff"].mean()
                            + sim_results["agent2_total_payoff"].mean()
                        ),
                        "mean_action_difference": sim_results[
                            "mean_action_difference"
                        ].mean(),
                        "belief_accuracy": (
                            1
                            - abs(
                                sim_results["agent1_final_belief"]
                                - sim_results["agent2_mean_action"]
                            ).mean()
                            + 1
                            - abs(
                                sim_results["agent2_final_belief"]
                                - sim_results["agent1_mean_action"]
                            ).mean()
                        )
                        / 2,
                    }

                    results.append(summary)

                except Exception as e:
                    print(f"    ✗ Error: {e}")
                    continue

        return pd.DataFrame(results)

    def create_visualizations(
        self, results_df: pd.DataFrame, analysis_type: str = "sweep"
    ):
        """
        Create comprehensive visualizations of results
        """

        print(f"Creating visualizations for {analysis_type} analysis...")

        if analysis_type == "sweep":
            self._create_sweep_visualizations(results_df)
        elif analysis_type == "sensitivity":
            self._create_sensitivity_visualizations(results_df)

    def _create_sweep_visualizations(self, df: pd.DataFrame):
        """Create visualizations for parameter sweep results"""

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")

        # 1. Heatmap of convergence rates by agent combinations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Convergence rate heatmap
        pivot_conv = (
            df.groupby(["agent1_type", "agent2_type"])["convergence_rate"]
            .mean()
            .unstack()
        )
        sns.heatmap(pivot_conv, annot=True, cmap="viridis", ax=axes[0, 0], fmt=".2f")
        axes[0, 0].set_title("Convergence Rate by Agent Combination")
        axes[0, 0].set_xlabel("Agent 2 Type")
        axes[0, 0].set_ylabel("Agent 1 Type")

        # Final warmth level heatmap
        pivot_warmth = (
            df.groupby(["agent1_type", "agent2_type"])["mean_final_warmth"]
            .mean()
            .unstack()
        )
        sns.heatmap(pivot_warmth, annot=True, cmap="plasma", ax=axes[0, 1], fmt=".2f")
        axes[0, 1].set_title("Final Warmth Level by Agent Combination")
        axes[0, 1].set_xlabel("Agent 2 Type")
        axes[0, 1].set_ylabel("Agent 1 Type")

        # Action method comparison
        method_stats = (
            df.groupby("action_method")
            .agg(
                {
                    "convergence_rate": ["mean", "std"],
                    "mean_final_warmth": ["mean", "std"],
                    "mean_total_payoff": ["mean", "std"],
                }
            )
            .round(3)
        )

        methods = df["action_method"].unique()
        conv_means = [
            df[df["action_method"] == m]["convergence_rate"].mean() for m in methods
        ]
        conv_stds = [
            df[df["action_method"] == m]["convergence_rate"].std() for m in methods
        ]

        axes[1, 0].bar(methods, conv_means, yerr=conv_stds, capsize=5)
        axes[1, 0].set_title("Convergence Rate by Action Selection Method")
        axes[1, 0].set_ylabel("Convergence Rate")

        # Payoff function comparison
        payoff_stats = (
            df.groupby(["payoff1_type", "payoff2_type"])["mean_total_payoff"]
            .mean()
            .unstack()
        )
        sns.heatmap(payoff_stats, annot=True, cmap="coolwarm", ax=axes[1, 1], fmt=".1f")
        axes[1, 1].set_title("Total Payoff by Payoff Function Combination")
        axes[1, 1].set_xlabel("Agent 2 Payoff Function")
        axes[1, 1].set_ylabel("Agent 1 Payoff Function")

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/plots/parameter_sweep_overview.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # 2. Detailed analysis plots
        self._create_detailed_analysis_plots(df)

    def _create_sensitivity_visualizations(self, df: pd.DataFrame):
        """Create visualizations for sensitivity analysis"""

        parameters = df["parameter"].unique()
        n_params = len(parameters)

        fig, axes = plt.subplots(n_params, 2, figsize=(12, 4 * n_params))
        if n_params == 1:
            axes = axes.reshape(1, -1)

        for i, param in enumerate(parameters):
            param_data = df[df["parameter"] == param]

            # Convergence rate vs parameter value
            axes[i, 0].plot(
                param_data["parameter_value"],
                param_data["convergence_rate"],
                "o-",
                linewidth=2,
                markersize=8,
            )
            axes[i, 0].set_xlabel(f"{param}")
            axes[i, 0].set_ylabel("Convergence Rate")
            axes[i, 0].set_title(f"Convergence Rate vs {param}")
            axes[i, 0].grid(True, alpha=0.3)

            # Final warmth vs parameter value
            axes[i, 1].plot(
                param_data["parameter_value"],
                param_data["mean_final_warmth"],
                "s-",
                linewidth=2,
                markersize=8,
                color="orange",
            )
            axes[i, 1].set_xlabel(f"{param}")
            axes[i, 1].set_ylabel("Final Warmth Level")
            axes[i, 1].set_title(f"Final Warmth vs {param}")
            axes[i, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/plots/sensitivity_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def _create_detailed_analysis_plots(self, df: pd.DataFrame):
        """Create additional detailed analysis plots"""

        # Correlation analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Convergence vs Final Warmth
        axes[0].scatter(df["mean_final_warmth"], df["convergence_rate"], alpha=0.6)
        axes[0].set_xlabel("Mean Final Warmth")
        axes[0].set_ylabel("Convergence Rate")
        axes[0].set_title("Convergence Rate vs Final Warmth Level")

        # Action Difference vs Convergence
        axes[1].scatter(
            df["mean_action_difference"],
            df["convergence_rate"],
            alpha=0.6,
            color="orange",
        )
        axes[1].set_xlabel("Mean Action Difference")
        axes[1].set_ylabel("Convergence Rate")
        axes[1].set_title("Convergence Rate vs Action Difference")

        # Payoff vs Convergence
        axes[2].scatter(
            df["mean_total_payoff"], df["convergence_rate"], alpha=0.6, color="green"
        )
        axes[2].set_xlabel("Mean Total Payoff")
        axes[2].set_ylabel("Convergence Rate")
        axes[2].set_title("Convergence Rate vs Total Payoff")

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/plots/detailed_correlations.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def save_results(self, results_df: pd.DataFrame, filename: str):
        """Save results to CSV file"""

        filepath = f"{self.output_dir}/{filename}"
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")

    def generate_summary_report(self, results_df: pd.DataFrame) -> str:
        """Generate a summary report of the results"""

        report = []
        report.append("PARAMETER SWEEP SUMMARY REPORT")
        report.append("=" * 50)
        report.append("")

        # Basic statistics
        report.append(f"Total combinations tested: {len(results_df)}")
        report.append(f"Agent types: {sorted(results_df['agent1_type'].unique())}")
        report.append(
            f"Payoff functions: {sorted(results_df['payoff1_type'].unique())}"
        )
        report.append(f"Action methods: {sorted(results_df['action_method'].unique())}")
        report.append("")

        # Key findings
        report.append("KEY FINDINGS:")
        report.append("-" * 20)

        # Best combinations
        best_convergence = results_df.loc[results_df["convergence_rate"].idxmax()]
        report.append(f"Best convergence: {best_convergence['convergence_rate']:.3f}")
        report.append(
            f"  Combination: {best_convergence['agent1_type']} vs {best_convergence['agent2_type']}"
        )
        report.append(
            f"  Payoffs: {best_convergence['payoff1_type']} vs {best_convergence['payoff2_type']}"
        )
        report.append(f"  Method: {best_convergence['action_method']}")
        report.append("")

        # Highest cooperation
        best_cooperation = results_df.loc[results_df["high_cooperation_rate"].idxmax()]
        report.append(
            f"Highest cooperation: {best_cooperation['high_cooperation_rate']:.3f}"
        )
        report.append(
            f"  Combination: {best_cooperation['agent1_type']} vs {best_cooperation['agent2_type']}"
        )
        report.append("")

        # Method comparison
        report.append("ACTION METHOD COMPARISON:")
        method_comparison = (
            results_df.groupby("action_method")
            .agg(
                {
                    "convergence_rate": "mean",
                    "mean_final_warmth": "mean",
                    "mean_total_payoff": "mean",
                }
            )
            .round(3)
        )
        report.append(method_comparison.to_string())
        report.append("")

        # Agent type analysis
        report.append("AGENT TYPE PERFORMANCE:")
        agent_performance = (
            results_df.groupby("agent1_type")
            .agg({"convergence_rate": "mean", "mean_final_warmth": "mean"})
            .round(3)
        )
        report.append(agent_performance.to_string())

        report_text = "\n".join(report)

        # Save report
        with open(f"{self.output_dir}/summary_report.txt", "w") as f:
            f.write(report_text)

        return report_text


def main():
    """
    Main function to run parameter sweeps with user-configurable options
    """

    print("Continuous Bayesian Learning - Parameter Sweep Tool")
    print("=" * 60)

    # Initialize sweep
    sweep = ParameterSweep()

    # Configuration options
    print("\nAvailable configurations:")
    print("Agent types:", list(sweep.define_agent_configurations().keys()))
    print("Payoff functions:", list(sweep.define_payoff_functions().keys()))
    print()

    # Example 1: Quick test sweep
    print("Running quick test sweep...")
    quick_params = {
        "agent1_types": ["secure", "anxious"],
        "agent2_types": ["secure", "anxious"],
        "payoff1_types": ["gaussian_matching", "symmetric_gaussian"],
        "payoff2_types": ["gaussian_matching", "symmetric_gaussian"],
        "action_methods": ["thompson", "ucb"],
    }

    results_quick = sweep.run_single_parameter_sweep(
        sweep_params=quick_params,
        n_simulations=10,  # Reduced for speed
        n_rounds=200,
        verbose=True,
    )

    # Save and visualize quick results
    sweep.save_results(results_quick, "quick_sweep_results.csv")
    sweep.create_visualizations(results_quick, "sweep")

    # Example 2: Sensitivity analysis
    print("\nRunning sensitivity analysis...")
    sensitivity_results = sweep.run_parameter_sensitivity_analysis(
        base_personality="secure",
        parameter_ranges={
            "prior_mean": [0.3, 0.5, 0.7, 0.9],
            "lambda_loss": [0.5, 1.0, 2.0, 3.0],
        },
    )

    sweep.save_results(sensitivity_results, "sensitivity_results.csv")
    sweep.create_visualizations(sensitivity_results, "sensitivity")

    # Generate summary report
    report = sweep.generate_summary_report(results_quick)
    print("\n" + report)

    print(f"\nAll results saved to: {sweep.output_dir}/")
    print("✓ Parameter sweep completed successfully!")


if __name__ == "__main__":
    main()
