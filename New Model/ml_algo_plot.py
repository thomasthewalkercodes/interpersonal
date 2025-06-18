import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict


def plot_evolution_history(history: List[Dict]):
    """Visualize the evolution progress with multiple metrics"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot fitness over generations
    generations = [h["generation"] for h in history]
    best_fitness = [h["best_fitness"] for h in history]
    avg_fitness = [h["avg_fitness"] for h in history]

    ax1.plot(generations, best_fitness, "b-", label="Best Fitness", linewidth=2)
    ax1.plot(generations, avg_fitness, "r--", label="Average Fitness", linewidth=2)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness Score")
    ax1.set_title("Evolution of Fitness Scores")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot parameter evolution
    params = list(history[0]["best_params"].keys())
    for param in params:
        values = [h["best_params"][param] for h in history]
        ax2.plot(generations, values, label=param, linewidth=2)

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Parameter Value")
    ax2.set_title("Evolution of Best Parameters")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    return fig
