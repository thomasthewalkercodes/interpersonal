"""
PARAMETER SWEEP ANALYSIS SCRIPT
===============================
This script loads previously saved simulation results and performs complete analysis
without re-running simulations. Perfect for when you want to re-analyze or
generate new visualizations from existing data.

Usage: python analysis_only_script.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from datetime import datetime
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, friedmanchisquare
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import os
import glob
import warnings

warnings.filterwarnings("ignore")


def load_simulation_results():
    """Load all saved simulation results and reconstruct the dataset"""
    print("🔍 LOADING SAVED SIMULATION RESULTS...")

    # Define the parameter space that was tested
    alpha_values = [2.0, 4.0, 6.0]
    beta_values = [5.0, 10.0, 15.0]
    personalities = ["cooperative", "competitive", "adaptive", "cautious"]

    print(f"   🔍 Looking for parameter combinations:")
    print(f"      Alpha values: {alpha_values}")
    print(f"      Beta values: {beta_values}")
    print(f"      Personalities: {personalities}")
    print(
        f"      Expected total: {len(personalities)**2 * len(alpha_values) * len(beta_values)} combinations"
    )

    results_data = []
    loaded_count = 0
    missing_count = 0

    # Try to load results from multiple possible locations/formats
    for agent1_type in personalities:
        for agent2_type in personalities:
            for alpha in alpha_values:
                for beta in beta_values:
                    # Your exact filename pattern: results/sweep_adaptive_cautious_alpha_6.0_beta_15.0/results.json
                    filename_pattern = (
                        f"sweep_{agent1_type}_{agent2_type}_alpha_{alpha}_beta_{beta}"
                    )

                    # Try different possible file paths in your directory structure
                    possible_files = [
                        f"results/{filename_pattern}/results.json",  # Your exact structure
                        f"results/{filename_pattern}/config.json",  # Alternative config file
                        f"results/{filename_pattern}/results.pkl",  # In case some are pickle
                        f"results/{filename_pattern}/final_results.json",  # Alternative name
                        f"results/{filename_pattern}/simulation_results.json",  # Alternative name
                        f"{filename_pattern}/results.json",  # Without results/ prefix
                        f"{filename_pattern}/config.json",  # Without results/ prefix
                    ]

                    result_loaded = False
                    for filepath in possible_files:
                        if os.path.exists(filepath):
                            try:
                                # Load the JSON result file
                                if filepath.endswith(".json"):
                                    import json

                                    with open(filepath, "r") as f:
                                        result = json.load(f)
                                elif filepath.endswith(".pkl"):
                                    import pickle

                                    with open(filepath, "rb") as f:
                                        result = pickle.load(f)
                                else:
                                    # Default to JSON since your files are results.json
                                    import json

                                    with open(filepath, "r") as f:
                                        result = json.load(f)

                                    # Extract the key metrics (adjust based on your result structure)
                                if (
                                    isinstance(result, dict)
                                    and "final_evaluation" in result
                                ):
                                    final_eval = result["final_evaluation"]

                                    # Convert string values to float if needed
                                    try:
                                        agent1_reward = float(
                                            final_eval["agent1_avg_reward"]
                                        )
                                        agent2_reward = float(
                                            final_eval["agent2_avg_reward"]
                                        )
                                    except (ValueError, TypeError) as e:
                                        print(
                                            f"   ⚠️ Could not convert rewards to numbers: {e}"
                                        )
                                        print(
                                            f"       agent1_avg_reward: {final_eval.get('agent1_avg_reward')} (type: {type(final_eval.get('agent1_avg_reward'))})"
                                        )
                                        print(
                                            f"       agent2_avg_reward: {final_eval.get('agent2_avg_reward')} (type: {type(final_eval.get('agent2_avg_reward'))})"
                                        )
                                        continue

                                    # DEBUG: Print what parameters we're actually loading
                                    if (
                                        loaded_count < 10
                                    ):  # Only print first 10 for brevity
                                        print(
                                            f"   🔍 DEBUG: Loading α={alpha}, β={beta} -> rewards: {agent1_reward:.2f}, {agent2_reward:.2f}"
                                        )

                                    results_data.append(
                                        {
                                            "total_reward": agent1_reward
                                            + agent2_reward,
                                            "agent1_reward": agent1_reward,
                                            "agent2_reward": agent2_reward,
                                            "reward_difference": abs(
                                                agent1_reward - agent2_reward
                                            ),
                                            "agent1_type": agent1_type,
                                            "agent2_type": agent2_type,
                                            "personality_pairing": f"{agent1_type}-{agent2_type}",
                                            "alpha": alpha,
                                            "beta": beta,
                                        }
                                    )
                                    loaded_count += 1
                                    result_loaded = True
                                    print(
                                        f"   ✅ Loaded: {agent1_type}-{agent2_type}, α={alpha}, β={beta} from {filepath}"
                                    )
                                    break
                                else:
                                    print(
                                        f"   ⚠️ File found but structure unexpected: {filepath}"
                                    )
                                    print(f"       Content type: {type(result)}")
                                    if isinstance(result, dict):
                                        print(
                                            f"       Keys available: {list(result.keys())}"
                                        )
                                        # Let's also check if the structure is slightly different
                                        if "results" in result:
                                            print(
                                                f"       'results' key found, checking its contents..."
                                            )
                                            results_section = result["results"]
                                            if isinstance(results_section, dict):
                                                print(
                                                    f"       Results section keys: {list(results_section.keys())}"
                                                )
                                    continue

                            except Exception as e:
                                print(f"   ❌ Error loading {filepath}: {e}")
                                continue

                    if not result_loaded:
                        missing_count += 1
                        print(
                            f"   ⚠️ Missing: {agent1_type}-{agent2_type}, α={alpha}, β={beta}"
                        )

    print(f"\n📊 LOADING SUMMARY:")
    print(f"   ✅ Successfully loaded: {loaded_count} simulations")
    print(f"   ⚠️ Missing results: {missing_count} simulations")
    print(
        f"   📈 Total expected: {len(personalities)**2 * len(alpha_values) * len(beta_values)} simulations"
    )

    if loaded_count == 0:
        print("\n❌ NO RESULTS FOUND!")
        print(
            "Please check that your simulation results are saved in one of these locations:"
        )
        print("   • Current directory (*.pkl, *.json)")
        print("   • ./results/ directory")
        print("   • ./data/ directory")
        print("   • ./data/results/ directory")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(results_data)

    # DEBUG: Show what parameter values were actually loaded
    print(f"\n🔍 PARAMETER VALUES ACTUALLY LOADED:")
    print(f"   Alpha values: {sorted(df['alpha'].unique())}")
    print(f"   Beta values: {sorted(df['beta'].unique())}")
    print(f"   Personality pairings: {len(df['personality_pairing'].unique())} unique")

    # Show count per parameter combination
    param_counts = df.groupby(["alpha", "beta"]).size().reset_index(name="count")
    print(f"\n📊 SIMULATIONS PER PARAMETER COMBINATION:")
    for _, row in param_counts.iterrows():
        print(f"   α={row['alpha']}, β={row['beta']}: {row['count']} simulations")

    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    return df


def load_from_csv_if_available():
    """Try to load from previously saved CSV analysis files"""
    print("🔍 Checking for previously saved CSV analysis files...")

    # Look for CSV files from previous analysis runs
    csv_patterns = [
        "parameter_sweep_results_*.csv",
        "detailed_results_*.csv",
        "sweep_results_*.csv",
    ]

    csv_files = []
    for pattern in csv_patterns:
        csv_files.extend(glob.glob(pattern))

    if csv_files:
        # Use the most recent CSV file
        latest_csv = max(csv_files, key=os.path.getctime)
        print(f"   📄 Found CSV file: {latest_csv}")

        try:
            df = pd.read_csv(latest_csv)
            print(f"   ✅ Loaded from CSV: {df.shape}")
            return df
        except Exception as e:
            print(f"   ❌ Error loading CSV: {e}")
            return None

    print("   📄 No CSV files found, will try to load from simulation files...")
    return None


def print_basic_results(df):
    """Print basic tabular results"""
    print(f"\n📋 PARAMETER SWEEP RESULTS:")
    print("=" * 85)
    print("Personality Pair\t\tParameters\t\tTotal\tAgent1\tAgent2\tDiff")
    print("-" * 85)

    for _, row in df.iterrows():
        personality_pair = f"{row['agent1_type']}-{row['agent2_type']}"
        params = f"α={row['alpha']}, β={row['beta']}"
        print(
            f"{personality_pair:<20}\t{params:<15}\t{row['total_reward']:.3f}\t{row['agent1_reward']:.3f}\t{row['agent2_reward']:.3f}\t{row['reward_difference']:.3f}"
        )


def analyze_parameter_sweep(df):
    """Comprehensive analysis of parameter sweep results"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 60)

    analysis_results = {}

    # 1. Find optimal parameter combinations - SEPARATE ANALYSIS
    analysis_results["optimal_total_reward"] = find_optimal_total_reward(df)
    analysis_results["optimal_balance"] = find_optimal_balance(df)

    # 2. Analyze personality adaptiveness
    analysis_results["personality_adaptiveness"] = analyze_personality_adaptiveness(df)

    return analysis_results


def find_optimal_total_reward(df):
    """Find optimal parameter combinations for TOTAL REWARD only"""
    print("\n🏆 OPTIMAL PARAMETERS FOR TOTAL REWARD:")
    print("-" * 50)

    optimal_params = {}

    # Best single combination
    best_total = df.loc[df["total_reward"].idxmax()]
    optimal_params["best_single"] = {
        "alpha": best_total["alpha"],
        "beta": best_total["beta"],
        "pairing": best_total["personality_pairing"],
        "total_reward": best_total["total_reward"],
    }
    print(f"🥇 Best Single Result: α={best_total['alpha']}, β={best_total['beta']}")
    print(
        f"   Pairing: {best_total['personality_pairing']}, Total: {best_total['total_reward']:.3f}"
    )

    # Best average performance across all personality pairings
    param_performance = (
        df.groupby(["alpha", "beta"])
        .agg({"total_reward": ["mean", "std", "min", "max", "count"]})
        .round(3)
    )
    param_performance.columns = ["mean", "std", "min", "max", "count"]

    best_avg_idx = param_performance["mean"].idxmax()
    best_avg = param_performance.loc[best_avg_idx]
    optimal_params["best_average"] = {
        "alpha": best_avg_idx[0],
        "beta": best_avg_idx[1],
        "avg_total_reward": best_avg["mean"],
        "std": best_avg["std"],
        "min": best_avg["min"],
        "max": best_avg["max"],
    }
    print(
        f"📈 Best Average Across All Pairings: α={best_avg_idx[0]}, β={best_avg_idx[1]}"
    )
    print(f"   Mean: {best_avg['mean']:.3f} ± {best_avg['std']:.3f}")
    print(f"   Range: {best_avg['min']:.3f} - {best_avg['max']:.3f}")

    # Show top 5 parameter combinations
    print(f"\n🔝 Top 5 Parameter Combinations (by average total reward):")
    top_5 = param_performance.sort_values("mean", ascending=False).head(5)
    for i, ((alpha, beta), row) in enumerate(top_5.iterrows(), 1):
        print(f"   {i}. α={alpha}, β={beta}: {row['mean']:.3f} ± {row['std']:.3f}")

    return optimal_params


def find_optimal_balance(df):
    """Find optimal parameter combinations for BALANCE (fairness) only"""
    print("\n⚖️ OPTIMAL PARAMETERS FOR BALANCE (FAIRNESS):")
    print("-" * 50)

    balance_params = {}

    # Most balanced single result
    most_balanced = df.loc[df["reward_difference"].idxmin()]
    balance_params["most_balanced_single"] = {
        "alpha": most_balanced["alpha"],
        "beta": most_balanced["beta"],
        "pairing": most_balanced["personality_pairing"],
        "difference": most_balanced["reward_difference"],
        "total_reward": most_balanced["total_reward"],
    }
    print(
        f"🥇 Most Balanced Single Result: α={most_balanced['alpha']}, β={most_balanced['beta']}"
    )
    print(
        f"   Pairing: {most_balanced['personality_pairing']}, Diff: {most_balanced['reward_difference']:.3f}"
    )
    print(f"   (Total reward: {most_balanced['total_reward']:.3f})")

    # Best average balance across all personality pairings
    balance_performance = (
        df.groupby(["alpha", "beta"])
        .agg({"reward_difference": ["mean", "std", "min", "max", "count"]})
        .round(3)
    )
    balance_performance.columns = ["mean", "std", "min", "max", "count"]

    best_balance_idx = balance_performance[
        "mean"
    ].idxmin()  # Lower difference = more balanced
    best_balance = balance_performance.loc[best_balance_idx]
    balance_params["best_average_balance"] = {
        "alpha": best_balance_idx[0],
        "beta": best_balance_idx[1],
        "avg_difference": best_balance["mean"],
        "std": best_balance["std"],
        "min": best_balance["min"],
        "max": best_balance["max"],
    }
    print(f"📈 Best Average Balance: α={best_balance_idx[0]}, β={best_balance_idx[1]}")
    print(f"   Mean Difference: {best_balance['mean']:.3f} ± {best_balance['std']:.3f}")
    print(f"   Range: {best_balance['min']:.3f} - {best_balance['max']:.3f}")

    # Show top 5 most balanced parameter combinations
    print(f"\n🔝 Top 5 Most Balanced Parameter Combinations:")
    top_5_balanced = balance_performance.sort_values("mean", ascending=True).head(5)
    for i, ((alpha, beta), row) in enumerate(top_5_balanced.iterrows(), 1):
        print(f"   {i}. α={alpha}, β={beta}: {row['mean']:.3f} ± {row['std']:.3f}")

    return balance_params


def analyze_personality_adaptiveness(df):
    """Analyze which personality types are most adaptive and provide reasoning"""
    print("\n🎭 PERSONALITY ADAPTIVENESS ANALYSIS:")
    print("-" * 50)

    adaptiveness_results = {}

    # Calculate adaptiveness metrics for each personality type
    personality_metrics = {}

    for personality in df["agent1_type"].unique():
        # Get all instances where this personality appears (as agent1 or agent2)
        as_agent1 = df[df["agent1_type"] == personality]["agent1_reward"]
        as_agent2 = df[df["agent2_type"] == personality]["agent2_reward"]
        all_rewards = pd.concat([as_agent1, as_agent2])

        # Calculate adaptiveness metrics
        personality_metrics[personality] = {
            "mean_reward": all_rewards.mean(),
            "std_reward": all_rewards.std(),
            "min_reward": all_rewards.min(),
            "max_reward": all_rewards.max(),
            "coefficient_of_variation": all_rewards.std()
            / all_rewards.mean(),  # Lower = more consistent
            "range": all_rewards.max() - all_rewards.min(),
            "sample_size": len(all_rewards),
        }

    # Rank personalities by different adaptiveness criteria
    adaptiveness_rankings = {}

    # 1. Highest average performance
    avg_ranking = sorted(
        personality_metrics.items(), key=lambda x: x[1]["mean_reward"], reverse=True
    )
    adaptiveness_rankings["highest_average"] = avg_ranking

    # 2. Most consistent (lowest coefficient of variation)
    consistency_ranking = sorted(
        personality_metrics.items(), key=lambda x: x[1]["coefficient_of_variation"]
    )
    adaptiveness_rankings["most_consistent"] = consistency_ranking

    # 3. Highest minimum performance (robust floor)
    floor_ranking = sorted(
        personality_metrics.items(), key=lambda x: x[1]["min_reward"], reverse=True
    )
    adaptiveness_rankings["highest_floor"] = floor_ranking

    # 4. Composite adaptiveness score
    # Normalize metrics and create composite score
    for personality in personality_metrics:
        metrics = personality_metrics[personality]
        # Higher mean reward, lower CV, higher min reward = more adaptive
        normalized_mean = (
            metrics["mean_reward"]
            - min([p["mean_reward"] for p in personality_metrics.values()])
        ) / (
            max([p["mean_reward"] for p in personality_metrics.values()])
            - min([p["mean_reward"] for p in personality_metrics.values()])
        )
        normalized_consistency = 1 - (
            (
                metrics["coefficient_of_variation"]
                - min(
                    [
                        p["coefficient_of_variation"]
                        for p in personality_metrics.values()
                    ]
                )
            )
            / (
                max(
                    [
                        p["coefficient_of_variation"]
                        for p in personality_metrics.values()
                    ]
                )
                - min(
                    [
                        p["coefficient_of_variation"]
                        for p in personality_metrics.values()
                    ]
                )
            )
        )
        normalized_floor = (
            metrics["min_reward"]
            - min([p["min_reward"] for p in personality_metrics.values()])
        ) / (
            max([p["min_reward"] for p in personality_metrics.values()])
            - min([p["min_reward"] for p in personality_metrics.values()])
        )

        # Composite score (equal weights)
        metrics["composite_adaptiveness"] = (
            normalized_mean + normalized_consistency + normalized_floor
        ) / 3

    composite_ranking = sorted(
        personality_metrics.items(),
        key=lambda x: x[1]["composite_adaptiveness"],
        reverse=True,
    )
    adaptiveness_rankings["composite"] = composite_ranking

    # Display results
    print("📊 Personality Performance Metrics:")
    print("Personality\t\tMean\tStd\tCV\tMin\tMax\tComposite")
    print("-" * 70)
    for personality, metrics in personality_metrics.items():
        print(
            f"{personality:<15}\t{metrics['mean_reward']:.3f}\t{metrics['std_reward']:.3f}\t{metrics['coefficient_of_variation']:.3f}\t{metrics['min_reward']:.3f}\t{metrics['max_reward']:.3f}\t{metrics['composite_adaptiveness']:.3f}"
        )

    print("\n🏆 ADAPTIVENESS RANKINGS & REASONING:")
    print("-" * 50)

    print("1️⃣ HIGHEST AVERAGE PERFORMANCE:")
    for i, (personality, metrics) in enumerate(avg_ranking, 1):
        print(f"   {i}. {personality}: {metrics['mean_reward']:.3f}")

    print("\n2️⃣ MOST CONSISTENT (Lowest Variability):")
    for i, (personality, metrics) in enumerate(consistency_ranking, 1):
        print(f"   {i}. {personality}: CV = {metrics['coefficient_of_variation']:.3f}")

    print("\n3️⃣ HIGHEST PERFORMANCE FLOOR (Robustness):")
    for i, (personality, metrics) in enumerate(floor_ranking, 1):
        print(f"   {i}. {personality}: Min = {metrics['min_reward']:.3f}")

    print("\n🎯 COMPOSITE ADAPTIVENESS RANKING:")
    for i, (personality, metrics) in enumerate(composite_ranking, 1):
        print(f"   {i}. {personality}: Score = {metrics['composite_adaptiveness']:.3f}")

    # Provide reasoning for most adaptive personality
    most_adaptive = composite_ranking[0][0]
    most_adaptive_metrics = composite_ranking[0][1]

    print(f"\n💡 SUGGESTED MOST ADAPTIVE PERSONALITY: {most_adaptive.upper()}")
    print("REASONING:")
    print(
        f"   • Average Performance: {most_adaptive_metrics['mean_reward']:.3f} (rank: {[p[0] for p in avg_ranking].index(most_adaptive) + 1})"
    )
    print(
        f"   • Consistency (CV): {most_adaptive_metrics['coefficient_of_variation']:.3f} (rank: {[p[0] for p in consistency_ranking].index(most_adaptive) + 1})"
    )
    print(
        f"   • Performance Floor: {most_adaptive_metrics['min_reward']:.3f} (rank: {[p[0] for p in floor_ranking].index(most_adaptive) + 1})"
    )
    print(
        f"   • Composite Score: {most_adaptive_metrics['composite_adaptiveness']:.3f}"
    )

    adaptiveness_results["metrics"] = personality_metrics
    adaptiveness_results["rankings"] = adaptiveness_rankings
    adaptiveness_results["most_adaptive"] = most_adaptive

    return adaptiveness_results


def perform_statistical_analysis(df):
    """Perform rigorous statistical testing with multiple comparison corrections"""
    print("\n" + "=" * 60)
    print("RIGOROUS STATISTICAL ANALYSIS")
    print("=" * 60)

    statistical_results = {}
    all_p_values = []
    test_descriptions = []

    print(
        "🔬 Performing comprehensive statistical testing with multiple comparison corrections..."
    )

    # 1. Test for main effects of alpha parameter
    print("\n1️⃣ ALPHA PARAMETER MAIN EFFECT:")
    alpha_unique = df["alpha"].unique()
    print(f"   Alpha values found: {sorted(alpha_unique)}")

    if len(alpha_unique) >= 2:
        alpha_groups = [
            df[df["alpha"] == alpha]["total_reward"].values for alpha in alpha_unique
        ]
        alpha_stat, alpha_p = kruskal(*alpha_groups)
        all_p_values.append(alpha_p)
        test_descriptions.append("Alpha parameter main effect (Kruskal-Wallis)")
        print(f"   Kruskal-Wallis H = {alpha_stat:.4f}, p = {alpha_p:.6f}")
        alpha_eta_squared = (alpha_stat - len(alpha_unique) + 1) / (
            len(df) - len(alpha_unique)
        )
    else:
        print(f"   ⚠️ Only one alpha value found - cannot perform statistical test")
        alpha_stat, alpha_p, alpha_eta_squared = None, None, None

    # 2. Test for main effects of beta parameter
    print("\n2️⃣ BETA PARAMETER MAIN EFFECT:")
    beta_unique = df["beta"].unique()
    print(f"   Beta values found: {sorted(beta_unique)}")

    if len(beta_unique) >= 2:
        beta_groups = [
            df[df["beta"] == beta]["total_reward"].values for beta in beta_unique
        ]
        beta_stat, beta_p = kruskal(*beta_groups)
        all_p_values.append(beta_p)
        test_descriptions.append("Beta parameter main effect (Kruskal-Wallis)")
        print(f"   Kruskal-Wallis H = {beta_stat:.4f}, p = {beta_p:.6f}")
        beta_eta_squared = (beta_stat - len(beta_unique) + 1) / (
            len(df) - len(beta_unique)
        )
    else:
        print(f"   ⚠️ Only one beta value found - cannot perform statistical test")
        beta_stat, beta_p, beta_eta_squared = None, None, None

    # 3. Test for personality pairing effects
    print("\n3️⃣ PERSONALITY PAIRING EFFECTS:")
    pairing_unique = df["personality_pairing"].unique()
    print(f"   Personality pairings found: {len(pairing_unique)} unique pairings")

    if len(pairing_unique) >= 2:
        pairing_groups = [
            df[df["personality_pairing"] == pairing]["total_reward"].values
            for pairing in pairing_unique
        ]
        pairing_stat, pairing_p = kruskal(*pairing_groups)
        all_p_values.append(pairing_p)
        test_descriptions.append("Personality pairing effect (Kruskal-Wallis)")
        print(f"   Kruskal-Wallis H = {pairing_stat:.4f}, p = {pairing_p:.6f}")
        pairing_eta_squared = (pairing_stat - len(pairing_unique) + 1) / (
            len(df) - len(pairing_unique)
        )
    else:
        print(
            f"   ⚠️ Only one personality pairing found - cannot perform statistical test"
        )
        pairing_stat, pairing_p, pairing_eta_squared = None, None, None

    # Calculate effect sizes (only for valid tests)
    print(f"\n📊 EFFECT SIZES:")
    if alpha_eta_squared is not None:
        print(f"   Alpha parameter effect size (η²): {alpha_eta_squared:.4f}")
    if beta_eta_squared is not None:
        print(f"   Beta parameter effect size (η²): {beta_eta_squared:.4f}")
    if pairing_eta_squared is not None:
        print(f"   Personality pairing effect size (η²): {pairing_eta_squared:.4f}")

    # 4. Multiple comparison correction (only if we have p-values)
    if all_p_values:
        print(f"\n🔬 MULTIPLE COMPARISON CORRECTION:")
        print(f"   Total number of statistical tests performed: {len(all_p_values)}")

        # Apply Benjamini-Hochberg FDR correction
        rejected_bh, corrected_p_bh, alpha_sidak, alpha_bonf = multipletests(
            all_p_values,
            alpha=0.05,
            method="fdr_bh",
            is_sorted=False,
            returnsorted=False,
        )

        print(f"   Benjamini-Hochberg FDR correction applied")
        print(
            f"   {sum(rejected_bh)}/{len(all_p_values)} tests remain significant after correction"
        )

        significant_tests_bh = sum(rejected_bh)
    else:
        print(
            f"\n⚠️ No statistical tests could be performed (insufficient group variation)"
        )
        rejected_bh, corrected_p_bh = [], []
        significant_tests_bh = 0

    # Store results (handling None values)
    statistical_results = {
        "alpha_main_effect": (
            {
                "statistic": alpha_stat,
                "p_value": alpha_p,
                "effect_size": alpha_eta_squared,
            }
            if alpha_stat is not None
            else None
        ),
        "beta_main_effect": (
            {"statistic": beta_stat, "p_value": beta_p, "effect_size": beta_eta_squared}
            if beta_stat is not None
            else None
        ),
        "personality_main_effect": (
            {
                "statistic": pairing_stat,
                "p_value": pairing_p,
                "effect_size": pairing_eta_squared,
            }
            if pairing_stat is not None
            else None
        ),
        "all_p_values": all_p_values,
        "test_descriptions": test_descriptions,
        "corrected_p_bh": corrected_p_bh,
        "rejected_bh": rejected_bh,
        "significant_tests_bh": significant_tests_bh,
    }

    return statistical_results


def create_all_visualizations(df):
    """Create all visualizations with the fixed code from earlier"""
    print("\n📈 GENERATING ALL VISUALIZATIONS...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_files = []

    # Use non-interactive backend
    import matplotlib

    matplotlib.use("Agg")

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # FIGURE 1: Total Reward Heatmap
    print("   Creating Figure 1: Total Reward Heatmap...")
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    pivot_total = df.groupby(["alpha", "beta"])["total_reward"].mean().unstack()
    im1 = ax1.imshow(pivot_total.values, cmap="viridis", aspect="auto")
    ax1.set_xticks(range(len(pivot_total.columns)))
    ax1.set_yticks(range(len(pivot_total.index)))
    ax1.set_xticklabels([f"β={x}" for x in pivot_total.columns])
    ax1.set_yticklabels([f"α={x}" for x in pivot_total.index])
    ax1.set_title(
        "Total Reward by Parameter Combinations\n(Higher = Better)",
        fontsize=16,
        fontweight="bold",
    )
    ax1.set_xlabel("Beta (Risk Penalty)", fontsize=12)
    ax1.set_ylabel("Alpha (Mismatch Penalty)", fontsize=12)

    # Add text annotations
    for i in range(len(pivot_total.index)):
        for j in range(len(pivot_total.columns)):
            ax1.text(
                j,
                i,
                f"{pivot_total.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    plt.colorbar(im1, ax=ax1, label="Average Total Reward")
    plt.tight_layout()
    filename1 = f"01_total_reward_heatmap_{timestamp}.png"
    plt.savefig(filename1, dpi=300, bbox_inches="tight")
    plot_files.append(filename1)
    # FIGURE 4: Parameter Sensitivity (modified for single beta value)
    print("   Creating Figure 4: Parameter Sensitivity...")
    if len(df["alpha"].unique()) > 1:
        fig4, ax4 = plt.subplots(figsize=(10, 6))

        # Alpha sensitivity (since we have multiple alpha values)
        alpha_effects = df.groupby("alpha").agg({"total_reward": ["mean", "std"]})
        alpha_effects.columns = ["mean", "std"]
        ax4.errorbar(
            alpha_effects.index,
            alpha_effects["mean"],
            yerr=alpha_effects["std"],
            marker="o",
            markersize=10,
            linewidth=3,
            capsize=8,
            capthick=2,
        )
        ax4.set_xlabel("Alpha (Mismatch Penalty)", fontsize=12)
        ax4.set_ylabel("Average Total Reward", fontsize=12)
        ax4.set_title(
            "Alpha Parameter Sensitivity\n(Beta fixed at 5.0)",
            fontsize=14,
            fontweight="bold",
        )
        ax4.grid(True, alpha=0.3)

        # Add text annotations for means
        for alpha, mean_val in zip(alpha_effects.index, alpha_effects["mean"]):
            ax4.text(
                alpha,
                mean_val + alpha_effects.loc[alpha, "std"] + 2,
                f"{mean_val:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        filename4 = f"04_parameter_sensitivity_{timestamp}.png"
        plt.savefig(filename4, dpi=300, bbox_inches="tight")
        plot_files.append(filename4)
        plt.close()
    else:
        print("   ⚠️ Skipping parameter sensitivity - insufficient parameter variation")

    # FIGURE 5: Balance by Personality Pairing
    print("   Creating Figure 5: Balance by Personality Pairing...")
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    balance_means = (
        df.groupby("personality_pairing")["reward_difference"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=True)
    )
    y_pos = range(len(balance_means))
    bars = ax5.barh(
        y_pos,
        balance_means["mean"],
        xerr=balance_means["std"],
        capsize=5,
        alpha=0.8,
        color="lightcoral",
    )
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(balance_means.index)
    ax5.set_xlabel("Average Reward Difference (Lower = More Balanced)", fontsize=12)
    ax5.set_title(
        "Balance by Personality Pairing\n(with Standard Deviation)",
        fontsize=16,
        fontweight="bold",
    )
    ax5.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, mean_val, std_val) in enumerate(
        zip(bars, balance_means["mean"], balance_means["std"])
    ):
        ax5.text(
            mean_val + std_val + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{mean_val:.3f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    filename5 = f"05_balance_by_personality_{timestamp}.png"
    plt.savefig(filename5, dpi=300, bbox_inches="tight")
    plot_files.append(filename5)
    plt.close()

    # FIGURE 6: Individual Personality Distributions
    print("   Creating Figure 6: Individual Personality Distributions...")
    fig6, axes = plt.subplots(2, 2, figsize=(16, 12))
    personalities = df["agent1_type"].unique()

    for i, personality in enumerate(personalities):
        ax = axes[i // 2, i % 2]

        # Get rewards for this personality
        as_agent1 = df[df["agent1_type"] == personality]["agent1_reward"]
        as_agent2 = df[df["agent2_type"] == personality]["agent2_reward"]
        all_rewards = pd.concat([as_agent1, as_agent2])

        # Create distribution plot
        ax.hist(
            all_rewards, bins=10, alpha=0.7, color=plt.cm.Set3(i), edgecolor="black"
        )
        ax.axvline(
            all_rewards.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {all_rewards.mean():.3f}",
        )
        ax.axvline(
            all_rewards.median(),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Median: {all_rewards.median():.3f}",
        )

        ax.set_title(
            f"{personality.capitalize()} Personality\nReward Distribution",
            fontweight="bold",
        )
        ax.set_xlabel("Individual Reward")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename6 = f"06_personality_distributions_{timestamp}.png"
    plt.savefig(filename6, dpi=300, bbox_inches="tight")
    plot_files.append(filename6)
    plt.close()

    # FIGURE 7: Personality Metrics Comparison
    print("   Creating Figure 7: Personality Metrics Comparison...")
    fig7, ((ax7a, ax7b), (ax7c, ax7d)) = plt.subplots(2, 2, figsize=(16, 12))

    # Calculate personality metrics
    personality_metrics = {}
    for personality in personalities:
        as_agent1 = df[df["agent1_type"] == personality]["agent1_reward"]
        as_agent2 = df[df["agent2_type"] == personality]["agent2_reward"]
        all_rewards = pd.concat([as_agent1, as_agent2])

        personality_metrics[personality] = {
            "mean": all_rewards.mean(),
            "std": all_rewards.std(),
            "min": all_rewards.min(),
            "max": all_rewards.max(),
            "cv": all_rewards.std() / all_rewards.mean(),
        }

    # Mean performance
    means = [personality_metrics[p]["mean"] for p in personalities]
    bars7a = ax7a.bar(
        personalities, means, color=plt.cm.Set3(range(len(personalities))), alpha=0.8
    )
    ax7a.set_title("Average Performance by Personality", fontweight="bold")
    ax7a.set_ylabel("Average Reward")
    ax7a.tick_params(axis="x", rotation=45)
    for bar, mean in zip(bars7a, means):
        ax7a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Consistency
    cvs = [personality_metrics[p]["cv"] for p in personalities]
    bars7b = ax7b.bar(
        personalities, cvs, color=plt.cm.Set3(range(len(personalities))), alpha=0.8
    )
    ax7b.set_title(
        "Consistency by Personality\n(Lower = More Consistent)", fontweight="bold"
    )
    ax7b.set_ylabel("Coefficient of Variation")
    ax7b.tick_params(axis="x", rotation=45)
    for bar, cv in zip(bars7b, cvs):
        ax7b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{cv:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Performance floor
    mins = [personality_metrics[p]["min"] for p in personalities]
    bars7c = ax7c.bar(
        personalities, mins, color=plt.cm.Set3(range(len(personalities))), alpha=0.8
    )
    ax7c.set_title(
        "Performance Floor by Personality\n(Higher = More Robust)", fontweight="bold"
    )
    ax7c.set_ylabel("Minimum Reward")
    ax7c.tick_params(axis="x", rotation=45)
    for bar, min_val in zip(bars7c, mins):
        ax7c.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{min_val:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Performance ceiling
    maxs = [personality_metrics[p]["max"] for p in personalities]
    bars7d = ax7d.bar(
        personalities, maxs, color=plt.cm.Set3(range(len(personalities))), alpha=0.8
    )
    ax7d.set_title(
        "Performance Ceiling by Personality\n(Higher = More Potential)",
        fontweight="bold",
    )
    ax7d.set_ylabel("Maximum Reward")
    ax7d.tick_params(axis="x", rotation=45)
    for bar, max_val in zip(bars7d, maxs):
        ax7d.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{max_val:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    filename7 = f"07_personality_metrics_{timestamp}.png"
    plt.savefig(filename7, dpi=300, bbox_inches="tight")
    plot_files.append(filename7)
    plt.close()

    # FIGURE 8: Data Quality Report
    print("   Creating Figure 8: Data Quality Report...")
    fig8, ((ax8a, ax8b), (ax8c, ax8d)) = plt.subplots(2, 2, figsize=(16, 12))

    # Parameter coverage
    param_counts = df.groupby(["alpha", "beta"]).size().reset_index(name="count")
    ax8a.bar(range(len(param_counts)), param_counts["count"])
    ax8a.set_title("Simulations per Parameter Combination", fontweight="bold")
    ax8a.set_xlabel("Parameter Set")
    ax8a.set_ylabel("Number of Simulations")
    ax8a.set_xticks(range(len(param_counts)))
    ax8a.set_xticklabels(
        [f"α={row['alpha']}, β={row['beta']}" for _, row in param_counts.iterrows()],
        rotation=45,
    )

    # Missing combinations report
    expected_combinations = 144  # 4×4×3×3
    actual_combinations = len(df)
    ax8b.pie(
        [actual_combinations, expected_combinations - actual_combinations],
        labels=[
            f"Found\n({actual_combinations})",
            f"Missing\n({expected_combinations - actual_combinations})",
        ],
        colors=["lightgreen", "lightcoral"],
        autopct="%1.1f%%",
    )
    ax8b.set_title("Simulation Completeness", fontweight="bold")

    # Reward distribution
    ax8c.hist(
        df["total_reward"], bins=15, alpha=0.7, color="skyblue", edgecolor="black"
    )
    ax8c.axvline(
        df["total_reward"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["total_reward"].mean():.1f}',
    )
    ax8c.set_title("Total Reward Distribution", fontweight="bold")
    ax8c.set_xlabel("Total Reward")
    ax8c.set_ylabel("Frequency")
    ax8c.legend()
    ax8c.grid(True, alpha=0.3)

    # Balance distribution
    ax8d.hist(
        df["reward_difference"],
        bins=15,
        alpha=0.7,
        color="lightcoral",
        edgecolor="black",
    )
    ax8d.axvline(
        df["reward_difference"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["reward_difference"].mean():.3f}',
    )
    ax8d.set_title("Reward Balance Distribution", fontweight="bold")
    ax8d.set_xlabel("Reward Difference")
    ax8d.set_ylabel("Frequency")
    ax8d.legend()
    ax8d.grid(True, alpha=0.3)

    plt.tight_layout()
    filename8 = f"08_data_quality_report_{timestamp}.png"
    plt.savefig(filename8, dpi=300, bbox_inches="tight")
    plot_files.append(filename8)
    plt.close()

    # FIGURE 2: Balance Heatmap
    print("   Creating Figure 2: Balance Heatmap...")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    pivot_balance = df.groupby(["alpha", "beta"])["reward_difference"].mean().unstack()
    im2 = ax2.imshow(pivot_balance.values, cmap="RdYlBu_r", aspect="auto")
    ax2.set_xticks(range(len(pivot_balance.columns)))
    ax2.set_yticks(range(len(pivot_balance.index)))
    ax2.set_xticklabels([f"β={x}" for x in pivot_balance.columns])
    ax2.set_yticklabels([f"α={x}" for x in pivot_balance.index])
    ax2.set_title(
        "Balance (Fairness) by Parameter Combinations\n(Lower = More Balanced)",
        fontsize=16,
        fontweight="bold",
    )
    ax2.set_xlabel("Beta (Risk Penalty)", fontsize=12)
    ax2.set_ylabel("Alpha (Mismatch Penalty)", fontsize=12)

    # Add text annotations
    for i in range(len(pivot_balance.index)):
        for j in range(len(pivot_balance.columns)):
            ax2.text(
                j,
                i,
                f"{pivot_balance.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    plt.colorbar(im2, ax=ax2, label="Average Reward Difference")
    plt.tight_layout()
    filename2 = f"02_balance_heatmap_{timestamp}.png"
    plt.savefig(filename2, dpi=300, bbox_inches="tight")
    plot_files.append(filename2)
    plt.close()

    # FIGURE 3: Personality Pairing Performance
    print("   Creating Figure 3: Personality Pairing Performance...")
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    pairing_means = (
        df.groupby("personality_pairing")["total_reward"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=True)
    )
    y_pos = range(len(pairing_means))
    bars = ax3.barh(
        y_pos,
        pairing_means["mean"],
        xerr=pairing_means["std"],
        capsize=5,
        alpha=0.8,
        color=plt.cm.Set3(range(len(pairing_means))),
    )
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(pairing_means.index)
    ax3.set_xlabel("Average Total Reward", fontsize=12)
    ax3.set_title(
        "Total Reward by Personality Pairing\n(with Standard Deviation)",
        fontsize=16,
        fontweight="bold",
    )
    ax3.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, mean_val, std_val) in enumerate(
        zip(bars, pairing_means["mean"], pairing_means["std"])
    ):
        ax3.text(
            mean_val + std_val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{mean_val:.3f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    filename3 = f"03_personality_pairing_performance_{timestamp}.png"
    plt.savefig(filename3, dpi=300, bbox_inches="tight")
    plot_files.append(filename3)
    plt.close()

    print(f"\n✅ Created {len(plot_files)} visualization files!")
    for filename in plot_files:
        print(f"   📊 {filename}")

    return plot_files


def save_comprehensive_results(df, analysis_results, statistical_results):
    """Save all analysis results to CSV files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []

    print("\n💾 SAVING COMPREHENSIVE RESULTS...")

    # 1. Save main dataset
    main_filename = f"analysis_results_main_{timestamp}.csv"
    df.to_csv(main_filename, index=False)
    saved_files.append(main_filename)
    print(f"   📄 Main results: {main_filename}")

    # 2. Save parameter optimization results
    param_filename = f"parameter_optimization_{timestamp}.csv"
    param_data = []

    # Add total reward optimization results
    if "optimal_total_reward" in analysis_results:
        opt_total = analysis_results["optimal_total_reward"]
        if "best_single" in opt_total:
            param_data.append(
                {
                    "optimization_type": "best_single_total_reward",
                    "alpha": opt_total["best_single"]["alpha"],
                    "beta": opt_total["best_single"]["beta"],
                    "pairing": opt_total["best_single"]["pairing"],
                    "value": opt_total["best_single"]["total_reward"],
                    "metric": "total_reward",
                }
            )

        if "best_average" in opt_total:
            param_data.append(
                {
                    "optimization_type": "best_average_total_reward",
                    "alpha": opt_total["best_average"]["alpha"],
                    "beta": opt_total["best_average"]["beta"],
                    "pairing": "all_pairings",
                    "value": opt_total["best_average"]["avg_total_reward"],
                    "metric": "average_total_reward",
                }
            )

    # Add balance optimization results
    if "optimal_balance" in analysis_results:
        opt_balance = analysis_results["optimal_balance"]
        if "most_balanced_single" in opt_balance:
            param_data.append(
                {
                    "optimization_type": "most_balanced_single",
                    "alpha": opt_balance["most_balanced_single"]["alpha"],
                    "beta": opt_balance["most_balanced_single"]["beta"],
                    "pairing": opt_balance["most_balanced_single"]["pairing"],
                    "value": opt_balance["most_balanced_single"]["difference"],
                    "metric": "reward_difference",
                }
            )

    if param_data:
        pd.DataFrame(param_data).to_csv(param_filename, index=False)
        saved_files.append(param_filename)
        print(f"   🎯 Parameter optimization: {param_filename}")

    # 3. Save personality analysis results
    personality_filename = f"personality_analysis_{timestamp}.csv"
    if "personality_adaptiveness" in analysis_results:
        personality_data = []
        metrics = analysis_results["personality_adaptiveness"]["metrics"]

        for personality, data in metrics.items():
            personality_data.append(
                {
                    "personality": personality,
                    "mean_reward": data["mean_reward"],
                    "std_reward": data["std_reward"],
                    "min_reward": data["min_reward"],
                    "max_reward": data["max_reward"],
                    "coefficient_of_variation": data["coefficient_of_variation"],
                    "composite_adaptiveness": data["composite_adaptiveness"],
                    "sample_size": data["sample_size"],
                }
            )

        pd.DataFrame(personality_data).to_csv(personality_filename, index=False)
        saved_files.append(personality_filename)
        print(f"   🎭 Personality analysis: {personality_filename}")

    # 4. Save statistical results (handle None values)
    stats_filename = f"statistical_analysis_{timestamp}.csv"
    stats_data = []

    if statistical_results:
        # Main effects (only add if not None)
        for effect_name in [
            "alpha_main_effect",
            "beta_main_effect",
            "personality_main_effect",
        ]:
            if (
                effect_name in statistical_results
                and statistical_results[effect_name] is not None
            ):
                effect = statistical_results[effect_name]
                stats_data.append(
                    {
                        "test_name": effect_name.replace("_", " ").title(),
                        "statistic": effect["statistic"],
                        "p_value": effect["p_value"],
                        "effect_size": effect["effect_size"],
                        "significant_after_correction": (
                            effect["p_value"] < 0.05
                            if effect["p_value"] is not None
                            else False
                        ),
                    }
                )
            else:
                # Add a row indicating the test could not be performed
                stats_data.append(
                    {
                        "test_name": effect_name.replace("_", " ").title(),
                        "statistic": "N/A - insufficient groups",
                        "p_value": "N/A",
                        "effect_size": "N/A",
                        "significant_after_correction": False,
                    }
                )

        if stats_data:
            pd.DataFrame(stats_data).to_csv(stats_filename, index=False)
            saved_files.append(stats_filename)
            print(f"   🔬 Statistical analysis: {stats_filename}")

    # 5. Save summary report
    summary_filename = f"analysis_summary_{timestamp}.txt"
    with open(summary_filename, "w") as f:
        f.write("PARAMETER SWEEP ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Dataset Overview:\n")
        f.write(f"- Total simulations: {len(df)}\n")
        f.write(f"- Personality pairings: {df['personality_pairing'].nunique()}\n")
        f.write(f"- Parameter combinations: {len(df.groupby(['alpha', 'beta']))}\n\n")

        f.write(f"Key Findings:\n")
        f.write(f"- Best total reward: {df['total_reward'].max():.3f}\n")
        f.write(f"- Most balanced outcome: {df['reward_difference'].min():.3f}\n")
        f.write(
            f"- Average total reward: {df['total_reward'].mean():.3f} ± {df['total_reward'].std():.3f}\n\n"
        )

        if "personality_adaptiveness" in analysis_results:
            most_adaptive = analysis_results["personality_adaptiveness"][
                "most_adaptive"
            ]
            f.write(f"Most adaptive personality: {most_adaptive}\n\n")

        if statistical_results and "significant_tests_bh" in statistical_results:
            f.write(f"Statistical significance:\n")
            f.write(
                f"- Significant tests after correction: {statistical_results['significant_tests_bh']}\n"
            )

        f.write(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    saved_files.append(summary_filename)
    print(f"   📋 Summary report: {summary_filename}")

    print(f"\n✅ Saved {len(saved_files)} analysis files!")
    return saved_files


def main():
    """Main analysis pipeline"""
    print("🚀 PARAMETER SWEEP ANALYSIS PIPELINE")
    print("=" * 60)

    # Step 1: Try to load data
    df = load_from_csv_if_available()

    if df is None:
        df = load_simulation_results()

    if df is None:
        print("❌ Could not load any results. Please check your file locations.")
        print("\nExpected file locations:")
        print("1. Simulation result files (*.pkl, *.json) in:")
        print("   - Current directory")
        print("   - ./results/")
        print("   - ./data/")
        print("   - ./data/results/")
        print("2. OR previous CSV analysis files:")
        print("   - parameter_sweep_results_*.csv")
        print("   - detailed_results_*.csv")
        return

    # Step 2: Basic results overview
    print_basic_results(df)

    # Step 3: Comprehensive analysis
    analysis_results = analyze_parameter_sweep(df)

    # Step 4: Statistical analysis
    statistical_results = perform_statistical_analysis(df)

    # Step 5: Create visualizations
    plot_files = create_all_visualizations(df)

    # Step 6: Save all results
    saved_files = save_comprehensive_results(df, analysis_results, statistical_results)

    # Step 7: Final summary
    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 60)

    print(f"\n📊 Generated Files:")
    print(f"   📈 Visualizations: {len(plot_files)} files")
    for plot_file in plot_files:
        print(f"      • {plot_file}")

    print(f"   💾 Analysis Results: {len(saved_files)} files")
    for saved_file in saved_files:
        print(f"      • {saved_file}")

    print(f"\n🎯 Key Insights:")
    if (
        "optimal_total_reward" in analysis_results
        and "best_single" in analysis_results["optimal_total_reward"]
    ):
        best = analysis_results["optimal_total_reward"]["best_single"]
        print(f"   • Best parameters: α={best['alpha']}, β={best['beta']}")
        print(f"   • Best pairing: {best['pairing']}")
        print(f"   • Best score: {best['total_reward']:.3f}")

    if "personality_adaptiveness" in analysis_results:
        most_adaptive = analysis_results["personality_adaptiveness"]["most_adaptive"]
        print(f"   • Most adaptive personality: {most_adaptive}")

    if statistical_results and "significant_tests_bh" in statistical_results:
        print(
            f"   • Significant statistical effects: {statistical_results['significant_tests_bh']}"
        )

    print(f"\n📝 Next Steps:")
    print(f"   1. Review the visualization files for key patterns")
    print(f"   2. Use the CSV files for further analysis or Paper 1")
    print(f"   3. Check the summary report for a complete overview")

    return df, analysis_results, statistical_results, plot_files, saved_files


if __name__ == "__main__":
    # Run the complete analysis pipeline
    results = main()

    if results[0] is not None:  # If data was successfully loaded
        df, analysis_results, statistical_results, plot_files, saved_files = results
        print(f"\n✅ Analysis objects available in memory:")
        print(f"   • df: Main dataset ({df.shape[0]} rows)")
        print(f"   • analysis_results: Comprehensive analysis dictionary")
        print(f"   • statistical_results: Statistical test results")
        print(f"   • plot_files: List of generated visualization files")
        print(f"   • saved_files: List of saved analysis files")
    else:
        print(f"\n❌ Analysis failed - no data could be loaded.")
        print(f"Please check your simulation result files and try again.")
