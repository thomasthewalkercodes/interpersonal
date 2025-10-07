import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
import random


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
    return octants[0]  # Wrap around to the first octant


df = pd.read_csv("C:\\Users\\thoma\\Downloads\\AAKTSE_wide_detrended_stacked.csv")

octants = ["LM", "NO", "PA", "BC", "DE", "FG", "HI", "JK"]

df["l_behavior"] = df.apply(
    lambda row: get_octants_from_coordinates(row["L_C"], row["L_A"]), axis=1
)
L_behavior_list = df["l_behavior"].tolist()
# print(L_behavior_list)


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

print(L_behavior_list)


# Build the transition matrix
def build_markov_chain(L_behavior_list):
    # Count transitions
    transitions = defaultdict(Counter)

    for i in range(len(L_behavior_list) - 1):
        current_state = L_behavior_list[i]
        next_state = L_behavior_list[i + 1]
        transitions[current_state][next_state] += 1

    # Convert counts to probabilities
    transition_probs = {}
    for state, next_states in transitions.items():
        total = sum(next_states.values())
        transition_probs[state] = {
            next_state: count / total for next_state, count in next_states.items()
        }

    return transition_probs


# Build the transition probability matrix
transition_probs = build_markov_chain(L_behavior_list)

# Get all unique states
states = sorted(set(L_behavior_list))


# Create a transition matrix as a DataFrame for better visualization
def create_transition_matrix_df(transition_probs, states):
    matrix = pd.DataFrame(0.0, index=states, columns=states)

    for from_state, to_states in transition_probs.items():
        for to_state, prob in to_states.items():
            matrix.loc[from_state, to_state] = prob

    return matrix


transition_matrix = create_transition_matrix_df(transition_probs, states)

# Print the transition matrix
print("Transition Probability Matrix:")
print("=" * 50)
print(transition_matrix.round(3))
print("\n")

# Print transition probabilities in a readable format
print("Transition Probabilities:")
print("=" * 50)
for from_state in states:
    if from_state in transition_probs:
        print(f"\nFrom {from_state}:")
        for to_state, prob in sorted(
            transition_probs[from_state].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  → {to_state}: {prob:.3f} ({prob*100:.1f}%)")


# Create a directed graph visualization
def visualize_markov_chain(transition_probs, min_prob=0.05):
    """
    Visualize the Markov chain as a directed graph.
    Only shows edges with probability >= min_prob for clarity.
    """
    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for state in states:
        G.add_node(state)

    # Add edges with weights (probabilities)
    for from_state, to_states in transition_probs.items():
        for to_state, prob in to_states.items():
            if prob >= min_prob:  # Only show significant transitions
                G.add_edge(from_state, to_state, weight=prob)

    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Use a circular layout for better visibility
    pos = nx.circular_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000, ax=ax)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)

    # Draw edges with varying thickness based on probability
    edges = G.edges(data=True)
    for u, v, d in edges:
        prob = d["weight"]
        # Make edge thickness proportional to probability
        width = prob * 10  # Scale for visibility
        # Color intensity based on probability
        alpha = min(0.3 + prob, 1.0)

        if u == v:  # Self-loop
            # Draw self-loops differently
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
                arrowstyle="->",
                ax=ax,
                connectionstyle="arc3,rad=0.15",
                min_target_margin=20,
            )

    # Add edge labels with probabilities
    edge_labels = {}
    for u, v, d in edges:
        prob = d["weight"]
        edge_labels[(u, v)] = f"{prob:.2f}"

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

    plt.title(
        "Markov Chain State Transition Diagram\n"
        + f"(showing transitions with probability ≥ {min_prob})",
        fontsize=14,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("markov_chain_l.png", dpi=300)
    plt.show()

    return G


# Visualize the Markov chain
print("\nGenerating visualization...")
graph = visualize_markov_chain(transition_probs, min_prob=0.05)

# Additional statistics
print("\n" + "=" * 50)
print("Markov Chain Statistics:")
print("=" * 50)
print(f"Number of states: {len(states)}")
print(f"Total transitions: {len(L_behavior_list) - 1}")
print(f"\nState frequencies:")
state_counts = Counter(L_behavior_list)
for state, count in sorted(state_counts.items()):
    print(f"  {state}: {count} times ({count/len(L_behavior_list)*100:.1f}%)")

# Find the most likely next state for each state
print(f"\nMost likely next state for each state:")
for state in states:
    if state in transition_probs:
        max_prob_state = max(transition_probs[state].items(), key=lambda x: x[1])
        print(f"  {state} → {max_prob_state[0]} (prob: {max_prob_state[1]:.3f})")

# Calculate steady-state distribution (if it exists)
try:
    # Convert to numpy array for eigenvalue calculation
    matrix_np = transition_matrix.values
    eigenvalues, eigenvectors = np.linalg.eig(matrix_np.T)

    # Find the eigenvector corresponding to eigenvalue 1
    idx = np.argmax(np.abs(eigenvalues - 1) < 1e-8)
    if np.abs(eigenvalues[idx] - 1) < 1e-8:
        steady_state = np.real(eigenvectors[:, idx])
        steady_state = steady_state / steady_state.sum()

        print(f"\nSteady-state distribution:")
        for i, state in enumerate(states):
            print(f"  {state}: {steady_state[i]:.3f}")
except Exception as e:
    print("\nCould not compute steady-state distribution:", e)
