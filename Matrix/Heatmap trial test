import numpy as np
import matplotlib.pyplot as plt

# Generate random position for Agent 1 (within unit circle)
while True:
    x1 = np.random.uniform(-1, 1)
    y1 = np.random.uniform(-1, 1)
    if np.sqrt(x1**2 + y1**2) <= 1:  # Check if point is inside unit circle
        break

# Create a grid of Agent 2's possible moves (between -1 and 1)
x2_vals = np.linspace(-1, 1, 100)  # Communion range: -1 to 1
y2_vals = np.linspace(-1, 1, 100)  # Agency range: -1 to 1
X2, Y2 = np.meshgrid(x2_vals, y2_vals)

# Calculate distances from center for both agents
dist1 = np.sqrt(x1**2 + y1**2)  # Distance of Agent 1 from center
dist2 = np.sqrt(X2**2 + Y2**2)  # Distance grid of Agent 2 from center

# Parameters
w_c = 1.0  # communion weight
w_a = 1.0  # agency weight
max_payoff = 10.0  # Maximum possible payoff when at optimal point

# Payoff calculation with circular normalization
# First normalize the positions to unit circle if they're outside
norm_factor = np.maximum(1, dist2)  # Will be 1 for points inside unit circle
X2_norm = X2 / norm_factor
Y2_norm = Y2 / norm_factor

# Calculate payoff using normalized coordinates
# Convert squared distances to similarity measure (higher = better)
communion_similarity = np.exp(-(w_c * (X2_norm - x1) ** 2))
agency_similarity = np.exp(-(w_a * (Y2_norm + y1) ** 2))
payoff = max_payoff * (communion_similarity * agency_similarity)

# Add circular mask to show unit circle boundary
circle_mask = dist2 > 1
payoff[circle_mask] = np.nan  # Make points outside unit circle transparent

# Generate a random point for Agent 2 (within unit circle)
while True:
    x2_random = np.random.uniform(-1, 1)
    y2_random = np.random.uniform(-1, 1)
    if (
        np.sqrt(x2_random**2 + y2_random**2) <= 1
    ):  # Check if point is inside unit circle
        break

# Calculate specific payoff for random point
dist2_random = np.sqrt(x2_random**2 + y2_random**2)
norm_factor_random = max(1, dist2_random)
x2_norm_random = x2_random / norm_factor_random
y2_norm_random = y2_random / norm_factor_random
communion_similarity_random = np.exp(-(w_c * (x2_norm_random - x1) ** 2))
agency_similarity_random = np.exp(-(w_a * (y2_norm_random + y1) ** 2))
random_payoff = max_payoff * (communion_similarity_random * agency_similarity_random)

# Plotting
plt.figure(figsize=(8, 8))
contour = plt.contourf(X2, Y2, payoff, levels=50, cmap="viridis")
plt.colorbar(contour, label="Agent 2 Payoff (higher is better)")
plt.plot(x1, y1, "ro", label="Agent 1 move")
plt.plot(
    x2_random,
    y2_random,
    "go",
    label=f"Agent 2 random move\nPayoff: {random_payoff:.3f}",
)
plt.xlabel("x2 (Communion)")
plt.ylabel("y2 (Agency)")
plt.title("Agent 2 Payoff Landscape\nBoth agents at random positions")
plt.legend(bbox_to_anchor=(1.15, 1))
plt.grid(True)
plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)

# Set equal aspect ratio and limits
plt.gca().set_aspect("equal")  # Forces square plotting
plt.xlim(-1.1, 1.1)  # Slightly larger than unit circle
plt.ylim(-1.1, 1.1)  # Slightly larger than unit circle

plt.tight_layout()
plt.show()
