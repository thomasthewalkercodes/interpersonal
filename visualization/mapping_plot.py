import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """Store results from a single simulation run"""

    config: Dict[str, float]
    payoffs1: List[float]
    payoffs2: List[float]
    joint_probs: Dict[str, List[float]]
    final_strategies: Tuple[float, float]


class ResultMapper:
    """Analyze and visualize results from multiple simulation runs"""

    def __init__(self, results: List[SimulationResult]):
        self.results = results
        self.summary_stats = self._calculate_summary_stats()

    def _calculate_summary_stats(self) -> Dict:
        """Calculate summary statistics for all runs"""
        stats = {}
        for result in self.results:
            key = tuple(result.config.items())  # Make dict hashable
            if key not in stats:
                stats[key] = {
                    "mean_payoff1": [],
                    "mean_payoff2": [],
                    "std_payoff1": [],
                    "std_payoff2": [],
                    "convergence_speed": [],
                    "final_strategies": [],
                }

            stats[key]["mean_payoff1"].append(np.mean(result.payoffs1))
            stats[key]["mean_payoff2"].append(np.mean(result.payoffs2))
            stats[key]["std_payoff1"].append(np.std(result.payoffs1))
            stats[key]["std_payoff2"].append(np.std(result.payoffs2))
            stats[key]["final_strategies"].append(result.final_strategies)

        return stats

    def create_heatmap(self, var1: str, var2: str, metric: str = "mean_payoff1"):
        """Create heatmap comparing two variables"""
        # Get unique values for each variable
        vals1 = sorted(set(res.config[var1] for res in self.results))
        vals2 = sorted(set(res.config[var2] for res in self.results))

        # Create matrix for heatmap
        matrix = np.zeros((len(vals1), len(vals2)))

        for i, v1 in enumerate(vals1):
            for j, v2 in enumerate(vals2):
                relevant_results = [
                    r
                    for r in self.results
                    if r.config[var1] == v1 and r.config[var2] == v2
                ]
                if metric.startswith("mean_payoff"):
                    matrix[i, j] = np.mean(
                        [
                            np.mean(r.payoffs1 if metric.endswith("1") else r.payoffs2)
                            for r in relevant_results
                        ]
                    )

        # Create heatmap
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
        plt.title(f"Heatmap of {metric} for different {var1} and {var2} values")
        plt.show()

    def plot_payoff_distributions(self, var1: str, var2: str):
        """Create boxplots showing payoff distributions"""
        data = []
        for result in self.results:
            v1, v2 = result.config[var1], result.config[var2]
            data.extend([(v1, v2, p, 1) for p in result.payoffs1])
            data.extend([(v1, v2, p, 2) for p in result.payoffs2])

        plt.figure(figsize=(12, 6))
        df = pd.DataFrame(data, columns=[var1, var2, "Payoff", "Player"])
        sns.boxplot(x=var1, y="Payoff", hue="Player", data=df)
        plt.title(f"Payoff Distributions for Different {var1} Values")
        plt.show()

    def plot_convergence_metrics(self, var1: str, var2: str):
        """Plot metrics showing convergence to Nash equilibrium"""
        plt.figure(figsize=(10, 6))
        for result in self.results:
            label = f"{var1}={result.config[var1]}, {var2}={result.config[var2]}"
            plt.plot(
                np.cumsum(result.payoffs1) / np.arange(1, len(result.payoffs1) + 1),
                label=label,
                alpha=0.5,
            )

        plt.xlabel("Round")
        plt.ylabel("Cumulative Average Payoff")
        plt.title("Convergence Analysis")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()
