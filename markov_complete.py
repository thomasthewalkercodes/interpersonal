import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
import random
import os


def get_octants_from_coordinates(communion, agency):
    angle_rad = np.arctan2(agency, communion)
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    octant_boundaries = [22.5 + i * 45 for i in range(8)]
    octants = ["LM", "NO", "PA", "BC", "DE", "FG", "HI", "JK"]
    for i, boundary in enumerate(octant_boundaries):
        if angle_deg < boundary:
            return octants[i]
    return octants[0]


df = pd.read_csv("C:\\Users\\thoma\\Downloads\\AAKTSE_wide_detrended_stacked.csv")

octants = ["LM", "NO", "PA", "BC", "DE", "FG", "HI", "JK"]

df["l_behavior"] = df.apply(
    lambda row: get_octants_from_coordinates(row["L_C"], row["L_A"]), axis=1
)
L_behavior_list = df["l_behavior"].tolist()


def circular_distance(o1, o2):
    """Smallest circular distance (steps) between two octants."""
    i1, i2 = octants.index(o1), octants.index(o2)
    d = abs(i1 - i2)
    return min(d, len(octants) - d)


def logical_path_correct(L_behavior_list, max_distance=3):

    newname = L_behavior_list.tolist()
    corrected = newname.copy()
    n = len(newname)

    for i in range(1, n - 1):
        prev_o = newname[i - 1]
        curr_o = newname[i]
        next_o = newname[i + 1]

        # Check for single-frame behavior (only one occurrence)
        if curr_o != prev_o and curr_o != next_o:
            # Check if start and end are close enough in circumplex
            if circular_distance(prev_o, next_o) <= max_distance:
                # Check if curr_o is logically between prev_o and next_o on the circle
                idx_prev = octants.index(prev_o)
                idx_next = octants.index(next_o)
                idx_curr = octants.index(curr_o)

                # handle circular wrap-around
                path_clockwise = (idx_prev + 1) % len(octants)
                path_counter = (idx_prev - 1) % len(octants)

                if idx_curr == path_clockwise or idx_curr == path_counter:
                    corrected[i] = (
                        next_o  # change transitional state to intended next one
                    )

    return pd.Series(corrected)


L_behavior_list = logical_path_correct(df["l_behavior"]).tolist()


def build_markov_chain(L_behavior_list):
    transitions = defaultdict(Counter)
    for i in range(len(L_behavior_list) - 1):
        current_state = L_behavior_list[i]
        next_state = L_behavior_list[i + 1]
        transitions[current_state][next_state] += 1

    transition_probs = {}
    for state, next_states in transitions.items():
        total = sum(next_states.values())
        transition_probs[state] = {
            next_state: count / total for next_state, count in next_states.items()
        }
    return transition_probs


def create_transition_matrix_df(transition_probs, states):
    matrix = pd.DataFrame(0.0, index=states, columns=states)
    for from_state, to_states in transition_probs.items():
        for to_state, prob in to_states.items():
            matrix.loc[from_state, to_state] = prob
    return matrix


def visualize_markov_chain(
    transition_probs, states, min_prob=0.05, filename="markov_chain.png"
):
    """Visualize the Markov chain as a directed graph."""
    G = nx.DiGraph()

    for state in states:
        G.add_node(state)

    for from_state, to_states in transition_probs.items():
        for to_state, prob in to_states.items():
            if prob >= min_prob:
                G.add_edge(from_state, to_state, weight=prob)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)

    edges = G.edges(data=True)
    for u, v, d in edges:
        prob = d["weight"]
        width = prob * 10
        alpha = min(0.3 + prob, 1.0)

        if u == v:
            nx.draw_networkx_edges(
                G,
                pos,
                [(u, v)],
                connectionstyle="arc3,rad=0.1",
                width=width,
                alpha=alpha,
                edge_color="red",
                arrows=True,
                arrowsize=20,
                arrowstyle="->",
                ax=ax,
            )
        else:
            nx.draw_networkx_edges(
                G,
                pos,
                [(u, v)],
                width=width,
                alpha=alpha,
                edge_color="gray",
                arrows=True,
                arrowsize=25,
                arrowstyle="-|>",
                ax=ax,
                connectionstyle="arc3,rad=0.15",
                min_target_margin=20,
            )

    edge_labels = {}
    for u, v, d in edges:
        prob = d["weight"]
        edge_labels[(u, v)] = f"{prob:.2f}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

    plt.title(
        f"Markov Chain State Transition Diagram\n"
        + f"(showing transitions with probability ≥ {min_prob})",
        fontsize=14,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return G


def analyze_markov_chain(behavior_list, participant, section):
    """Analyze Markov chain and return statistics as dictionary"""
    if len(behavior_list) < 2:
        return None

    transition_probs = build_markov_chain(behavior_list)
    states = sorted(set(behavior_list))
    transition_matrix = create_transition_matrix_df(transition_probs, states)

    # Statistics
    state_counts = Counter(behavior_list)

    # Most likely transitions
    most_likely = {}
    for state in states:
        if state in transition_probs:
            max_prob_state = max(transition_probs[state].items(), key=lambda x: x[1])
            most_likely[state] = max_prob_state

    # Try to compute steady state
    steady_state = None
    try:
        matrix_np = transition_matrix.values
        eigenvalues, eigenvectors = np.linalg.eig(matrix_np.T)
        idx = np.argmax(np.abs(eigenvalues - 1) < 1e-8)
        if np.abs(eigenvalues[idx] - 1) < 1e-8:
            steady_state = np.real(eigenvectors[:, idx])
            steady_state = steady_state / steady_state.sum()
    except:
        pass

    return {
        "participant": participant,
        "section": section,
        "transition_probs": transition_probs,
        "transition_matrix": transition_matrix,
        "states": states,
        "state_counts": state_counts,
        "total_transitions": len(behavior_list) - 1,
        "behavior_list_length": len(behavior_list),
        "most_likely": most_likely,
        "steady_state": steady_state,
    }


# Create output directory
os.makedirs("markov_outputs", exist_ok=True)

# Store all results
all_results = []

# Process each section and participant
for section in ["AAKTSE1", "AAKTSE2", "AAKTSE3", "AAKTSE4"]:
    mask = df["Video"] == section

    # Process L participant
    df.loc[mask, f"L_behavior_{section}"] = df.loc[mask].apply(
        lambda row: get_octants_from_coordinates(row["L_C"], row["L_A"]), axis=1
    )
    L_behavior_section = logical_path_correct(
        df.loc[mask, f"L_behavior_{section}"]
    ).tolist()

    # Process R participant
    df.loc[mask, f"R_behavior_{section}"] = df.loc[mask].apply(
        lambda row: get_octants_from_coordinates(row["R_C"], row["R_A"]), axis=1
    )
    R_behavior_section = logical_path_correct(
        df.loc[mask, f"R_behavior_{section}"]
    ).tolist()

    # Analyze L
    result_L = analyze_markov_chain(L_behavior_section, "L", section)
    if result_L:
        all_results.append(result_L)
        filename_L = f"markov_outputs/markov_{section}_L.png"
        visualize_markov_chain(
            result_L["transition_probs"],
            result_L["states"],
            min_prob=0.05,
            filename=filename_L,
        )

    # Analyze R
    result_R = analyze_markov_chain(R_behavior_section, "R", section)
    if result_R:
        all_results.append(result_R)
        filename_R = f"markov_outputs/markov_{section}_R.png"
        visualize_markov_chain(
            result_R["transition_probs"],
            result_R["states"],
            min_prob=0.05,
            filename=filename_R,
        )

# Generate comprehensive markdown report
markdown_report = "# Markov Chain Analysis Report\n\n"
markdown_report += f"**Total Analyses:** {len(all_results)}\n\n"
markdown_report += "---\n\n"

for result in all_results:
    participant = result["participant"]
    section = result["section"]

    markdown_report += f"## {section} - Participant {participant}\n\n"
    markdown_report += f"**Image:** `markov_{section}_{participant}.png`\n\n"

    markdown_report += f"### Overview\n"
    markdown_report += f"- **Number of states:** {len(result['states'])}\n"
    markdown_report += f"- **Total observations:** {result['behavior_list_length']}\n"
    markdown_report += f"- **Total transitions:** {result['total_transitions']}\n\n"

    markdown_report += f"### State Frequencies\n"
    for state, count in sorted(result["state_counts"].items()):
        pct = count / result["behavior_list_length"] * 100
        markdown_report += f"- **{state}:** {count} times ({pct:.1f}%)\n"
    markdown_report += "\n"

    markdown_report += f"### Transition Probabilities\n"
    for from_state in result["states"]:
        if from_state in result["transition_probs"]:
            markdown_report += f"\n**From {from_state}:**\n"
            sorted_transitions = sorted(
                result["transition_probs"][from_state].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for to_state, prob in sorted_transitions:
                markdown_report += f"  - → {to_state}: {prob:.3f} ({prob*100:.1f}%)\n"
    markdown_report += "\n"

    markdown_report += f"### Most Likely Next State\n"
    for state, (next_state, prob) in result["most_likely"].items():
        markdown_report += f"- **{state}** → **{next_state}** (prob: {prob:.3f})\n"
    markdown_report += "\n"

    if result["steady_state"] is not None:
        markdown_report += f"### Steady-State Distribution\n"
        for i, state in enumerate(result["states"]):
            markdown_report += f"- **{state}:** {result['steady_state'][i]:.3f}\n"
        markdown_report += "\n"
    else:
        markdown_report += f"### Steady-State Distribution\n"
        markdown_report += "*Could not compute steady-state distribution*\n\n"

    markdown_report += "---\n\n"

# Save markdown report
with open("markov_outputs/markov_analysis_report.md", "w", encoding="utf-8") as f:
    f.write(markdown_report)

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nGenerated {len(all_results)} Markov chain visualizations")
print(f"\nAll outputs saved to: markov_outputs/")
print(f"  - {len(all_results)} PNG files (markov_AAKTSE*_L/R.png)")
print(f"  - 1 Markdown report (markov_analysis_report.md)")
print("\n" + "=" * 70)
