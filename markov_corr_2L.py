import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
import random


import random
from collections import Counter


def build_markov_chain(sequence=None, length=1000):
    """
    Builds a transition probability matrix with circular neighbor correlations.
    - If `sequence` is given: states are derived automatically from it.
    - If not: uses the predefined parameters and generates a simulated sequence.
    """

    # --- Base frequencies for all states ---
    base_weights_dict = {
        "BC": 4.2,
        "DE": 1.9,
        "FG": 7.8,
        "HI": 41.4,
        "JK": 8.7,
        "LM": 3.9,
        "NO": 0.8,
        "PA": 31.2,
    }

    # --- Self-transition probabilities ---
    self_probs = {
        "PA": 0.93,
        "BC": 0.64,
        "DE": 0.55,
        "FG": 0.72,
        "HI": 0.91,
        "JK": 0.75,
        "LM": 0.70,
        "NO": 0.40,
    }

    # --- Circular arrangement ---
    circular_order = ["LM", "NO", "PA", "BC", "DE", "FG", "HI", "JK"]

    # --- Determine states dynamically ---
    if sequence is not None:
        states = sorted(set(sequence))
    else:
        states = list(base_weights_dict.keys())

    # Filter base weights for only the relevant states
    base_weights = [base_weights_dict[s] for s in states if s in base_weights_dict]
    total = sum(base_weights)
    base_probs = {
        s: base_weights_dict[s] / total for s in states if s in base_weights_dict
    }

    # --- Helper function to get circular distance ---
    def get_circular_distance(state1, state2):
        """Returns the minimum circular distance between two states (0-4)"""
        if state1 not in circular_order or state2 not in circular_order:
            return 2  # neutral if not in circular order

        idx1 = circular_order.index(state1)
        idx2 = circular_order.index(state2)
        n = len(circular_order)

        # Calculate both directions and take minimum
        forward = (idx2 - idx1) % n
        backward = (idx1 - idx2) % n
        return min(forward, backward)

    # --- Build transition probability matrix with circular correlations ---
    transition_probs = {}
    for state in states:
        row = {}
        self_p = self_probs.get(state, 0.5)
        row[state] = self_p
        remaining = 1 - self_p

        others = [s for s in states if s != state]

        # Calculate correlation weights based on circular distance
        correlation_weights = {}
        for o in others:
            dist = get_circular_distance(state, o)

            if dist == 1:  # Neighbors: highest correlation
                correlation_weights[o] = base_probs[o] * 3.0
            elif dist == 2:  # Orthogonal: neutral (base probability)
                correlation_weights[o] = base_probs[o] * 1.0
            elif dist == 3:  # Near-opposite: negative correlation
                correlation_weights[o] = base_probs[o] * 0.3
            elif dist == 4:  # Opposite: strong negative correlation
                correlation_weights[o] = base_probs[o] * 0.1
            else:
                correlation_weights[o] = base_probs[o]

        # Normalize correlation weights
        sum_weights = sum(correlation_weights.values())
        for o in others:
            row[o] = remaining * (correlation_weights[o] / sum_weights)

        transition_probs[state] = row

    # --- If no sequence given, generate one ---
    if sequence is None:
        sequence = generate_sequence_from_chain(
            transition_probs, states, base_weights, length
        )
        print(sequence)

    return transition_probs


def generate_sequence_from_chain(transition_probs, states, base_weights, length):
    """Generate a random sequence following a given Markov chain."""
    result = []
    current = random.choices(states, weights=base_weights, k=1)[0]
    result.append(current)

    for _ in range(length - 1):
        probs = [transition_probs[current][s] for s in states]
        current = random.choices(states, weights=probs, k=1)[0]
        result.append(current)

    return result


# Build the transition probability matrix
transition_probs = build_markov_chain()


sequence = ["LM", "NO", "PA", "BC", "DE", "FG", "HI", "JK"]
# Get all unique states
states = sorted(set())


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
                min_target_margin=10,
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
                arrowsize=20,
                arrowstyle="->",
                ax=ax,
                min_target_margin=20,
                connectionstyle="arc3,rad=0.1",
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
print(f"Total transitions: {len(sequence) - 1}")
print(f"\nState frequencies:")
state_counts = Counter(sequence)
for state, count in sorted(state_counts.items()):
    print(f"  {state}: {count} times ({count/len(sequence)*100:.1f}%)")

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
