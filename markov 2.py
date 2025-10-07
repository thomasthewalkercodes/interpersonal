import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
import random


def generate_random_list(length):

    params = ["LM", "NO", "PA", "BC", "DE", "FG", "HI", "JK"]
    return [random.choice(params) for _ in range(length)]


sequence = generate_random_list(10000)


# Build the transition matrix
def build_markov_chain(sequence):
    # Count transitions
    transitions = defaultdict(Counter)

    for i in range(len(sequence) - 2):
        current_state = sequence[i]
        next_state = sequence[i + 1]
        next_state2 = sequence[i + 2]
        transitions[(current_state, next_state)][next_state2] += 1

    # Convert counts to probabilities
    transition_probs = {}
    for state, next_states in transitions.items():
        total = sum(next_states.values())
        transition_probs[state] = {
            next_state: count / total for next_state, count in next_states.items()
        }

    return transition_probs


# Build the transition probability matrix
transition_probs = build_markov_chain(sequence)

# Get all unique states
states = sorted(set(sequence))


# Create a transition matrix as a DataFrame for better visualization
def create_transition_matrix_df(transition_probs, states):
    # Use tuples of states as index, single states as columns
    index = list(transition_probs.keys())
    matrix = pd.DataFrame(0.0, index=index, columns=states)

    for from_state, to_states in transition_probs.items():
        for to_state, prob in to_states.items():
            matrix.at[from_state, to_state] = prob  # <-- FIXED

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
for from_state in transition_probs:  # Use keys from transition_probs (tuples)
    print(f"\nFrom {from_state}:")
    for to_state, prob in sorted(
        transition_probs[from_state].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  → {to_state}: {prob:.3f} ({prob*100:.1f}%)")

# Find the most likely next state for each state pair
print(f"\nMost likely next state for each state pair:")
for from_state in transition_probs:
    max_prob_state = max(transition_probs[from_state].items(), key=lambda x: x[1])
    print(f"  {from_state} → {max_prob_state[0]} (prob: {max_prob_state[1]:.3f})")


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
                arrowsize=20,
                arrowstyle="->",
                ax=ax,
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
