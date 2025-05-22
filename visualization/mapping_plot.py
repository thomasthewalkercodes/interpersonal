import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """Store results from a single simulation run"""

    config: Dict[str, float]
    payoffs1: List[float]
    payoffs2: List[float]
    joint_probs: Dict[str, float]
    final_strategies: Tuple[float, float]


class ResultMapper:
    """Analyze and visualize results from multiple simulation runs"""

    def __init__(self, results: List[SimulationResult]):
        self.results = results
        # Get the tested variables from the first result's config
        # (these are the ones specified in variable_ranges)
        self.tested_variables = sorted(
            set(
                key
                for key in self.results[0].config.keys()
                if len(set(res.config[key] for res in self.results)) > 1
            )
        )

    def create_heatmap(self, var1: str, var2: str, metric: str = "mean_payoff1"):
        """Create heatmap comparing two variables"""
        # Get unique values for each variable
        vals1 = sorted(set(res.config[var1] for res in self.results))
        vals2 = sorted(set(res.config[var2] for res in self.results))

        # Create matrix for heatmap
        matrix = np.zeros((len(vals1), len(vals2)))  # Fixed closing parenthesis

        # Fill matrix with average payoffs
        for i, v1 in enumerate(vals1):
            for j, v2 in enumerate(vals2):
                relevant_results = [
                    r
                    for r in self.results
                    if r.config[var1] == v1 and r.config[var2] == v2
                ]
                if relevant_results:
                    if metric.startswith("mean_payoff"):
                        payoffs = [
                            np.mean(r.payoffs1 if metric.endswith("1") else r.payoffs2)
                            for r in relevant_results
                        ]
                        matrix[i, j] = np.mean(payoffs)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            xticklabels=vals2,
            yticklabels=vals1,
            annot=True,
            fmt=".2f",
            cmap="viridis",
        )
        plt.xlabel(var2)
        plt.ylabel(var1)
        plt.title(f"Average {metric} for different {var1} and {var2} values")
        plt.show()

    def plot_payoff_distributions(self, var1: str, var2: str):
        """Create boxplots showing payoff distributions"""
        data = []
        for result in self.results:
            data.extend(
                [
                    (result.config[var1], result.config[var2], p, "Player 1")
                    for p in result.payoffs1
                ]
            )
            data.extend(
                [
                    (result.config[var1], result.config[var2], p, "Player 2")
                    for p in result.payoffs2
                ]
            )

        df = pd.DataFrame(data, columns=[var1, var2, "Payoff", "Player"])
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=var1, y="Payoff", hue="Player", data=df)
        plt.title(f"Payoff Distributions for Different {var1} Values")
        plt.show()

    def create_faceted_heatmap(self, value_col="payoff_diff"):
        """Create single heatmap showing all variable combinations"""
        variables = self.tested_variables

        # Create DataFrame with all results
        data = []
        for result in self.results:
            # Create a combination string for the x-axis
            var_combo = "\n".join(
                [f"{var}={result.config[var]}" for var in variables[:-1]]
            )
            row = {
                "variable_combo": var_combo,
                "last_var": result.config[variables[-1]],
                "mean_payoff1": np.mean(result.payoffs1),
                "mean_payoff2": np.mean(result.payoffs2),
                "payoff_diff": np.mean(result.payoffs1) - np.mean(result.payoffs2),
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Create pivot table for heatmap
        pivot = pd.pivot_table(
            df,
            values=["payoff_diff", "mean_payoff1", "mean_payoff2"],
            index="variable_combo",
            columns="last_var",
            aggfunc="mean",
        )

        # Create figure
        plt.figure(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(
            pivot["payoff_diff"],
            annot=False,
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Payoff Difference (P1 - P2)"},
        )

        # Add text annotations with both players' payoffs
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns.levels[1])):
                p1 = pivot["mean_payoff1"].iloc[i, j]
                p2 = pivot["mean_payoff2"].iloc[i, j]
                diff = pivot["payoff_diff"].iloc[i, j]
                text = f"Δ: {diff:.2f}\nP1: {p1:.2f}\nP2: {p2:.2f}"
                plt.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

        # Customize labels
        plt.xlabel(f"{variables[-1]} values")
        plt.ylabel("Variable Combinations")
        plt.title("Payoff Analysis for All Variable Combinations")

        plt.tight_layout()
        plt.show()
