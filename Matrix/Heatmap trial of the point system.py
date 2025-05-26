import numpy as np
import matplotlib.pyplot as plt

# Agent 1's fixed move (between -1 and 1)
x1, y1 = -0.5, 0.5  # Example values in new range

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

# Payoff calculation with circular normalization
# First normalize the positions to unit circle if they're outside
norm_factor = np.maximum(1, dist2)  # Will be 1 for points inside unit circle
X2_norm = X2 / norm_factor
Y2_norm = Y2 / norm_factor

# Calculate payoff using normalized coordinates
payoff = -(w_c * (X2_norm - x1) ** 2 + w_a * (Y2_norm + y1) ** 2)

# Add circular mask to show unit circle boundary
circle_mask = dist2 > 1
payoff[circle_mask] = np.nan  # Make points outside unit circle transparent

# Plotting
plt.figure(figsize=(8, 8))  # Changed to square figure size
contour = plt.contourf(X2, Y2, payoff, levels=50, cmap="viridis")
plt.colorbar(contour, label="Agent 2 Payoff")
plt.plot(x1, y1, "ro", label="Agent 1 move")
plt.xlabel("x2 (Communion)")
plt.ylabel("y2 (Agency)")
plt.title("Agent 2 Payoff Landscape")
plt.legend()
plt.grid(True)
plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)

# Set equal aspect ratio and limits
plt.gca().set_aspect("equal")  # Forces square plotting
plt.xlim(-1.1, 1.1)  # Slightly larger than unit circle
plt.ylim(-1.1, 1.1)  # Slightly larger than unit circle

plt.tight_layout()
plt.show()
