"""
Markov Chain Data Analysis for Behavioral Sequences

Analyzes behavioral sequence data (L_C, L_A, R_C, R_A) using the Interpersonal Circumplex:
1. Convert Control/Agency coordinates to circumplex octants (PA, BC, DE, FG, HI, JK, LM, NO)
2. Count octant occurrences and transitions across all sessions
3. Calculate transition probabilities and octant ratios
4. Create visualizations showing distributions and Markov chains
5. Compare Left (L) vs Right (R) patterns on the circumplex
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict


def cartesian_to_circumplex(control, agency):
    """
    Convert Control (warmth) and Agency (dominance) to circumplex angle and radius.

    Args:
        control: Control dimension (warmth, x-axis)
        agency: Agency dimension (dominance, y-axis)

    Returns:
        Tuple of (angle, radius)
    """
    if pd.isna(control) or pd.isna(agency):
        return None, None

    # Convert to angle (0-360 degrees)
    angle = np.degrees(np.arctan2(agency, control)) % 360

    # Calculate radius (intensity)
    radius = np.hypot(control, agency)

    return angle, radius


def angle_to_octant(angle):
    """
    Convert circumplex angle to octant label.

    Octants:
    - PA: 337.5° - 22.5° (Warm-Dominant, Gregarious-Extraverted)
    - BC: 22.5° - 67.5° (Dominant, Assured-Dominant)
    - DE: 67.5° - 112.5° (Cold-Dominant, Arrogant-Calculating)
    - FG: 112.5° - 157.5° (Cold, Aloof-Introverted)
    - HI: 157.5° - 202.5° (Cold-Submissive, Unassured-Submissive)
    - JK: 202.5° - 247.5° (Submissive, Unassuming-Ingenuous)
    - LM: 247.5° - 292.5° (Warm-Submissive, Warm-Agreeable)
    - NO: 292.5° - 337.5° (Warm, Assured-Dominant)

    Args:
        angle: Angle in degrees (0-360)

    Returns:
        Octant label (PA, BC, DE, FG, HI, JK, LM, NO)
    """
    if pd.isna(angle):
        return None

    # Normalize angle to 0-360
    angle = angle % 360

    # Determine octant (45-degree segments, starting at -22.5 degrees from 0)
    if angle < 22.5 or angle >= 337.5:
        return 'PA'
    elif angle < 67.5:
        return 'BC'
    elif angle < 112.5:
        return 'DE'
    elif angle < 157.5:
        return 'FG'
    elif angle < 202.5:
        return 'HI'
    elif angle < 247.5:
        return 'JK'
    elif angle < 292.5:
        return 'LM'
    else:  # 292.5 to 337.5
        return 'NO'


def load_and_categorize_data(file_path):
    """
    Load behavioral data and convert to circumplex octants.

    Returns:
        - df: DataFrame with original data plus octant columns
        - sequences: Dictionary of octant sequences for Left and Right
        - radius_data: Dictionary of radius (intensity) data
    """
    df = pd.read_csv(file_path)

    print(f"\nLoading: {file_path.name}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Convert Control/Agency pairs to circumplex coordinates
    # Left side: L_C (control/warmth), L_A (agency/dominance)
    df['L_angle'], df['L_radius'] = zip(*df.apply(
        lambda row: cartesian_to_circumplex(row['L_C'], row['L_A']),
        axis=1
    ))
    df['L_octant'] = df['L_angle'].apply(angle_to_octant)

    # Right side: R_C (control/warmth), R_A (agency/dominance)
    df['R_angle'], df['R_radius'] = zip(*df.apply(
        lambda row: cartesian_to_circumplex(row['R_C'], row['R_A']),
        axis=1
    ))
    df['R_octant'] = df['R_angle'].apply(angle_to_octant)

    # Create sequences (remove None values)
    sequences = {
        'L': [o for o in df['L_octant'].tolist() if o is not None],
        'R': [o for o in df['R_octant'].tolist() if o is not None]
    }

    # Store radius data for intensity analysis
    radius_data = {
        'L': [r for r in df['L_radius'].tolist() if not pd.isna(r)],
        'R': [r for r in df['R_radius'].tolist() if not pd.isna(r)]
    }

    return df, sequences, radius_data


def build_markov_chain(sequence):
    """
    Build first-order Markov chain from sequence.

    Returns:
        - transitions: Dictionary of transition counts
        - transition_probs: Dictionary of transition probabilities
    """
    transitions = defaultdict(Counter)

    for i in range(len(sequence) - 1):
        current_state = sequence[i]
        next_state = sequence[i + 1]
        transitions[current_state][next_state] += 1

    # Convert counts to probabilities
    transition_probs = {}
    for state, next_states in transitions.items():
        total = sum(next_states.values())
        transition_probs[state] = {
            next_state: count / total
            for next_state, count in next_states.items()
        }

    return transitions, transition_probs


def build_second_order_markov_chain(sequence):
    """
    Build second-order Markov chain from sequence.

    Returns:
        - transitions: Dictionary of transition counts
        - transition_probs: Dictionary of transition probabilities
    """
    transitions = defaultdict(Counter)

    for i in range(len(sequence) - 2):
        current_state = (sequence[i], sequence[i + 1])
        next_state = sequence[i + 2]
        transitions[current_state][next_state] += 1

    # Convert counts to probabilities
    transition_probs = {}
    for state, next_states in transitions.items():
        total = sum(next_states.values())
        transition_probs[state] = {
            next_state: count / total
            for next_state, count in next_states.items()
        }

    return transitions, transition_probs


def calculate_behavior_statistics(sequences):
    """
    Calculate statistics for each side (Left/Right).

    Returns:
        Dictionary with counts and ratios for each side
    """
    stats = {}

    for side, sequence in sequences.items():
        counts = Counter(sequence)
        total = len(sequence)
        ratios = {state: count/total for state, count in counts.items()}

        stats[side] = {
            'counts': counts,
            'ratios': ratios,
            'total': total
        }

    return stats


def create_behavior_distribution_plot(stats, output_path):
    """
    Create bar plots showing octant distributions for Left and Right.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sides = ['L', 'R']
    titles = ['Left Side Octant Distribution', 'Right Side Octant Distribution']
    colors = ['#3498db', '#e74c3c']

    # All octants in circumplex order
    all_octants = ['PA', 'BC', 'DE', 'FG', 'HI', 'JK', 'LM', 'NO']

    for idx, (side, title, color) in enumerate(zip(sides, titles, colors)):
        ax = axes[idx]

        if side not in stats:
            continue

        ratios = stats[side]['ratios']
        counts = stats[side]['counts']

        # Ensure all octants are represented
        ratio_vals = [ratios.get(octant, 0) for octant in all_octants]
        count_vals = [counts.get(octant, 0) for octant in all_octants]

        bars = ax.bar(all_octants, ratio_vals, color=color, alpha=0.7,
                     edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Circumplex Octant', fontweight='bold', fontsize=13)
        ax.set_ylabel('Ratio', fontweight='bold', fontsize=13)
        ax.set_title(title, fontweight='bold', fontsize=15)
        ax.set_ylim(0, max(ratio_vals) * 1.2 if max(ratio_vals) > 0 else 1)
        ax.grid(axis='y', alpha=0.3)

        # Add count and ratio labels
        for bar, ratio, count in zip(bars, ratio_vals, count_vals):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{ratio:.3f}\n(n={count:,})',
                       ha='center', va='bottom', fontweight='bold',
                       fontsize=9)

    plt.suptitle('Interpersonal Circumplex Octant Distributions',
                fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved behavior distribution plot: {output_path}")


def create_comparison_plot(stats, output_path):
    """
    Create overlaid comparison of Left vs Right octant distributions.
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    all_octants = ['PA', 'BC', 'DE', 'FG', 'HI', 'JK', 'LM', 'NO']
    x = np.arange(len(all_octants))
    width = 0.35

    # Get ratios for both sides
    l_vals = [stats['L']['ratios'].get(octant, 0) for octant in all_octants]
    r_vals = [stats['R']['ratios'].get(octant, 0) for octant in all_octants]

    bars1 = ax.bar(x - width/2, l_vals, width, label='Left',
                  color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, r_vals, width, label='Right',
                  color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_xlabel('Circumplex Octant', fontweight='bold', fontsize=14)
    ax.set_ylabel('Ratio', fontweight='bold', fontsize=14)
    ax.set_title('Left vs Right: Circumplex Octant Comparison',
                fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(all_octants, fontsize=12)
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9,
                       fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot: {output_path}")


def visualize_markov_chain(transition_probs, title, output_path, min_prob=0.1):
    """
    Visualize the Markov chain as a directed graph.
    """
    G = nx.DiGraph()

    # Get all unique states
    states = set()
    for from_state, to_states in transition_probs.items():
        states.add(from_state)
        states.update(to_states.keys())

    # Add nodes
    for state in states:
        G.add_node(state)

    # Add edges with weights
    for from_state, to_states in transition_probs.items():
        for to_state, prob in to_states.items():
            if prob >= min_prob:
                G.add_edge(from_state, to_state, weight=prob)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Use circular layout
    pos = nx.circular_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=2500, ax=ax, edgecolors='black', linewidths=2)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', ax=ax)

    # Draw edges
    edges = G.edges(data=True)
    for u, v, d in edges:
        prob = d['weight']
        width = prob * 15  # Scale for visibility
        alpha = min(0.3 + prob, 1.0)

        if u == v:  # Self-loop
            nx.draw_networkx_edges(
                G, pos, [(u, v)],
                connectionstyle='arc3,rad=0.3',
                width=width, alpha=alpha,
                edge_color='red', arrows=True,
                arrowsize=25, arrowstyle='->',
                ax=ax
            )
        else:
            nx.draw_networkx_edges(
                G, pos, [(u, v)],
                width=width, alpha=alpha,
                edge_color='gray', arrows=True,
                arrowsize=25, arrowstyle='->',
                connectionstyle='arc3,rad=0.15',
                ax=ax
            )

    # Add edge labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, ax=ax)

    ax.set_title(f'{title}\n(showing transitions with probability ≥ {min_prob})',
                fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved Markov chain visualization: {output_path}")


def create_transition_matrix_heatmap(transition_probs, title, output_path):
    """
    Create heatmap visualization of transition probabilities.
    """
    # Get all unique states
    states = sorted(set(list(transition_probs.keys()) +
                       [s for ts in transition_probs.values() for s in ts.keys()]))

    # Create matrix
    matrix = pd.DataFrame(0.0, index=states, columns=states)

    for from_state, to_states in transition_probs.items():
        for to_state, prob in to_states.items():
            matrix.at[from_state, to_state] = prob

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
               cbar_kws={'label': 'Transition Probability'},
               linewidths=0.5, ax=ax)

    ax.set_title(f'{title} - Transition Probability Matrix',
                fontweight='bold', fontsize=14)
    ax.set_xlabel('To State', fontweight='bold', fontsize=12)
    ax.set_ylabel('From State', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved transition matrix heatmap: {output_path}")


def create_summary_table(stats, output_path):
    """
    Create comprehensive summary statistics table for octant distributions.
    """
    all_octants = ['PA', 'BC', 'DE', 'FG', 'HI', 'JK', 'LM', 'NO']
    sides = ['L', 'R']
    side_names = ['Left', 'Right']

    summary_data = []

    for side, side_name in zip(sides, side_names):
        for octant in all_octants:
            count = stats[side]['counts'].get(octant, 0)
            ratio = stats[side]['ratios'].get(octant, 0)

            summary_data.append({
                'Side': side_name,
                'Octant': octant,
                'Count': count,
                'Ratio': ratio,
                'Percentage': ratio * 100
            })

    df = pd.DataFrame(summary_data)

    # Save as CSV
    csv_path = output_path.parent / 'behavior_summary_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved summary table: {csv_path}")

    # Create visual table
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('tight')
    ax.axis('off')

    # Group by side for better readability
    table_data = [['Side', 'Octant', 'Count', 'Ratio', 'Percentage']]

    for side_name in side_names:
        side_df = df[df['Side'] == side_name]
        for _, row in side_df.iterrows():
            table_data.append([
                row['Side'],
                row['Octant'],
                f"{row['Count']:,}",
                f"{row['Ratio']:.4f}",
                f"{row['Percentage']:.2f}%"
            ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style header
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')

    # Color rows by side
    colors_left = '#e3f2fd'
    colors_right = '#ffebee'

    for i in range(1, len(table_data)):
        current_side = table_data[i][0]
        color = colors_left if current_side == 'Left' else colors_right

        for j in range(5):
            table[(i, j)].set_facecolor(color)

    plt.title('Circumplex Octant Statistics Summary',
             fontweight='bold', fontsize=16, pad=20)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved visual summary table: {output_path}")

    return df


def main():
    """
    Main analysis function.
    """
    print("=" * 70)
    print("MARKOV CHAIN DATA ANALYSIS: Interpersonal Circumplex")
    print("=" * 70)

    # Set up paths
    data_dir = Path(__file__).parent
    output_dir = data_dir / 'markov_analysis_results'
    output_dir.mkdir(exist_ok=True)

    data_file = data_dir / 'AAKTSE_wide_detrended_stacked.csv'

    # Load and categorize data
    print("\n" + "=" * 70)
    print("LOADING AND CONVERTING TO CIRCUMPLEX OCTANTS")
    print("=" * 70)
    df, sequences, radius_data = load_and_categorize_data(data_file)

    # Calculate statistics
    print("\n" + "=" * 70)
    print("CALCULATING OCTANT STATISTICS")
    print("=" * 70)
    stats = calculate_behavior_statistics(sequences)

    all_octants = ['PA', 'BC', 'DE', 'FG', 'HI', 'JK', 'LM', 'NO']

    for side in ['L', 'R']:
        side_name = 'Left' if side == 'L' else 'Right'
        print(f"\n{side_name} Side:")
        print(f"  Total observations: {stats[side]['total']:,}")
        for octant in all_octants:
            count = stats[side]['counts'].get(octant, 0)
            ratio = stats[side]['ratios'].get(octant, 0)
            if count > 0:
                print(f"    {octant}: {count:,} ({ratio:.4f})")

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    create_behavior_distribution_plot(
        stats, output_dir / 'octant_distributions.png'
    )

    create_comparison_plot(
        stats, output_dir / 'left_vs_right_comparison.png'
    )

    summary_df = create_summary_table(
        stats, output_dir / 'octant_summary_table.png'
    )

    # Build and visualize Markov chains
    print("\n" + "=" * 70)
    print("BUILDING MARKOV CHAINS")
    print("=" * 70)

    for side in ['L', 'R']:
        side_name = 'Left' if side == 'L' else 'Right'
        print(f"\nProcessing {side_name} side...")

        transitions, transition_probs = build_markov_chain(sequences[side])

        # Visualize
        visualize_markov_chain(
            transition_probs,
            f'{side_name} Side Markov Chain',
            output_dir / f'{side}_markov_chain.png',
            min_prob=0.05
        )

        create_transition_matrix_heatmap(
            transition_probs,
            f'{side_name} Side',
            output_dir / f'{side}_transition_matrix.png'
        )

    # Save text summary
    print("\n" + "=" * 70)
    print("SAVING TEXT SUMMARY")
    print("=" * 70)

    summary_file = output_dir / 'analysis_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("MARKOV CHAIN DATA ANALYSIS SUMMARY\n")
        f.write("Interpersonal Circumplex Octant Analysis\n")
        f.write("=" * 70 + "\n\n")

        f.write("Octant Labels:\n")
        f.write("  PA: Warm-Dominant (Gregarious-Extraverted)\n")
        f.write("  BC: Dominant (Assured-Dominant)\n")
        f.write("  DE: Cold-Dominant (Arrogant-Calculating)\n")
        f.write("  FG: Cold (Aloof-Introverted)\n")
        f.write("  HI: Cold-Submissive (Unassured-Submissive)\n")
        f.write("  JK: Submissive (Unassuming-Ingenuous)\n")
        f.write("  LM: Warm-Submissive (Warm-Agreeable)\n")
        f.write("  NO: Warm (Gregarious)\n\n")

        for side in ['L', 'R']:
            side_name = 'Left' if side == 'L' else 'Right'
            f.write(f"\n{side_name} SIDE\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total observations: {stats[side]['total']:,}\n\n")

            for octant in all_octants:
                count = stats[side]['counts'].get(octant, 0)
                ratio = stats[side]['ratios'].get(octant, 0)
                f.write(f"  {octant}: {count:,} ({ratio:.6f})\n")

            f.write("\n")

        # Add mean radius (intensity) information
        f.write("\nMEAN INTENSITY (RADIUS)\n")
        f.write("-" * 70 + "\n")
        for side in ['L', 'R']:
            side_name = 'Left' if side == 'L' else 'Right'
            mean_radius = np.mean(radius_data[side])
            std_radius = np.std(radius_data[side])
            f.write(f"{side_name}: {mean_radius:.2f} (SD={std_radius:.2f})\n")

    print(f"Saved text summary: {summary_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
