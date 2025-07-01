import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def calculate_warmth_payoff(
    w1: float, w2: float, alpha: float = 8, beta: float = 20
) -> float:
    """Calculate payoff for warmth interaction between two agents."""
    # Ensure inputs are in valid range
    w1 = np.clip(w1, 0, 1)
    w2 = np.clip(w2, 0, 1)

    # Calculate payoff components
    mismatch = (w1 - w2) ** 2
    warmth_bonus = (w1 + w2) / 2
    risk_factor = w1
    penalty = risk_factor * mismatch * beta
    base_payoff = np.exp(-alpha * mismatch)

    return base_payoff + warmth_bonus - penalty


def create_poster_plots(alpha=8, beta=20, resolution=100, save_path=None):
    """Create publication-quality plots for scientific poster."""

    # Set style for publication
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    # Generate data
    w1 = np.linspace(0, 1, resolution)
    w2 = np.linspace(0, 1, resolution)
    W1, W2 = np.meshgrid(w1, w2)

    # Calculate payoffs
    P = np.zeros_like(W1)
    for i in range(len(w1)):
        for j in range(len(w2)):
            P[i, j] = calculate_warmth_payoff(W1[i, j], W2[i, j], alpha, beta)

    # Create custom colormap

    n_bins = 256
    cmap = plt.get_cmap("viridis")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. 3D Surface Plot
    ax1 = fig.add_subplot(221, projection="3d")
    surf = ax1.plot_surface(
        W1, W2, P, cmap=cmap, alpha=0.9, linewidth=0, antialiased=True, edgecolor="none"
    )

    # Add contour lines on the surface
    contours = ax1.contour(
        W1, W2, P, levels=15, colors="white", alpha=0.6, linewidths=0.8
    )

    ax1.set_xlabel("Agent 1 Warmth", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Agent 2 Warmth", fontsize=12, fontweight="bold")
    ax1.set_zlabel("Payoff", fontsize=12, fontweight="bold")
    ax1.set_title("3D Payoff Surface", fontsize=14, fontweight="bold", pad=20)
    ax1.view_init(elev=30, azim=45)

    # 2. 2D Heatmap
    ax2 = fig.add_subplot(222)
    im = ax2.imshow(P, extent=[0, 1, 0, 1], origin="lower", cmap=cmap, aspect="equal")

    # Add contour lines
    contour_lines = ax2.contour(
        W1, W2, P, levels=15, colors="white", alpha=0.7, linewidths=1
    )
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")

    ax2.set_xlabel("Agent 1 Warmth", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Agent 2 Warmth", fontsize=12, fontweight="bold")
    ax2.set_title("Payoff Heatmap with Contours", fontsize=14, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label("Payoff", fontsize=12, fontweight="bold")

    # 3. Cross-sections along diagonal and anti-diagonal
    ax3 = fig.add_subplot(223)

    # Diagonal line (w1 = w2)
    diagonal_payoffs = [calculate_warmth_payoff(w, w, alpha, beta) for w in w1]
    ax3.plot(
        w1,
        diagonal_payoffs,
        "b-",
        linewidth=3,
        label="Equal Warmth (w₁ = w₂)",
        marker="o",
        markersize=4,
    )

    # Anti-diagonal samples
    anti_diag_w1 = np.linspace(0, 1, 50)
    anti_diag_w2 = 1 - anti_diag_w1
    anti_diag_payoffs = [
        calculate_warmth_payoff(w1_val, w2_val, alpha, beta)
        for w1_val, w2_val in zip(anti_diag_w1, anti_diag_w2)
    ]
    ax3.plot(
        anti_diag_w1,
        anti_diag_payoffs,
        "r--",
        linewidth=3,
        label="Opposite Warmth (w₁ + w₂ = 1)",
        marker="s",
        markersize=4,
    )

    # Fixed w2 = 0.5 line
    fixed_w2_payoffs = [calculate_warmth_payoff(w, 0.5, alpha, beta) for w in w1]
    ax3.plot(
        w1,
        fixed_w2_payoffs,
        "g:",
        linewidth=3,
        label="w₂ = 0.5 (fixed)",
        marker="^",
        markersize=4,
    )

    ax3.set_xlabel("Agent 1 Warmth", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Payoff", fontsize=12, fontweight="bold")
    ax3.set_title("Payoff Cross-Sections", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Parameter sensitivity analysis
    ax4 = fig.add_subplot(224)

    # Test different alpha values
    alphas = [1, 2, 4, 6, 8]
    w_test = 0.3  # Fixed mismatch scenario

    for alpha_test in alphas:
        payoffs = [calculate_warmth_payoff(w_test, w, alpha_test, beta) for w in w1]
        ax4.plot(
            w1,
            payoffs,
            linewidth=2.5,
            label=f"α = {alpha_test}",
            marker="o",
            markersize=3,
        )

    ax4.axvline(x=w_test, color="black", linestyle="--", alpha=0.7, linewidth=1)
    ax4.set_xlabel("Agent 2 Warmth", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Payoff", fontsize=12, fontweight="bold")
    ax4.set_title(
        f"Sensitivity to α (Agent 1 warmth = {w_test})", fontsize=14, fontweight="bold"
    )
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(
        f"Warmth Interaction Payoff Analysis (α={alpha}, β={beta})",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    if save_path:
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Plot saved to: {save_path}")

    plt.show()
    return fig


def create_simple_3d_plot(alpha=4, beta=10, resolution=100, save_path=None):
    """Create a single, clean 3D plot for poster."""

    # Generate data
    w1 = np.linspace(0, 1, resolution)
    w2 = np.linspace(0, 1, resolution)
    W1, W2 = np.meshgrid(w1, w2)

    P = np.zeros_like(W1)
    for i in range(len(w1)):
        for j in range(len(w2)):
            P[i, j] = calculate_warmth_payoff(W1[i, j], W2[i, j], alpha, beta)

    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Custom colormap
    colors = [
        "#2d1b69",
        "#1f4788",
        "#1f7892",
        "#5ba3a3",
        "#9ccabc",
        "#dcecc8",
        "#fff8e1",
    ]
    cmap = LinearSegmentedColormap.from_list("scientific", colors, N=256)

    # Create surface
    surf = ax.plot_surface(
        W1,
        W2,
        P,
        cmap=cmap,
        alpha=0.95,
        linewidth=0,
        antialiased=True,
        edgecolor="none",
    )

    # Add contour lines
    contours = ax.contour(
        W1, W2, P, levels=12, colors="white", alpha=0.8, linewidths=1.2
    )

    # Styling
    ax.set_xlabel("Agent 1 Warmth", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Agent 2 Warmth", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_zlabel("Payoff", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(
        f"Warmth Interaction Payoff Surface\n(α = {alpha}, β = {beta})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Set viewing angle
    ax.view_init(elev=25, azim=45)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label("Payoff", fontsize=12, fontweight="bold")

    # Clean up the plot
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Simple 3D plot saved to: {save_path}")

    plt.show()
    return fig


# Example usage
if __name__ == "__main__":
    # Create comprehensive analysis plots
    fig1 = create_poster_plots(
        alpha=8, beta=20, resolution=100, save_path="warmth_payoff_analysis.png"
    )

    # Create simple 3D plot
    fig2 = create_simple_3d_plot(
        alpha=4, beta=10, resolution=100, save_path="warmth_payoff_3d.png"
    )

    # You can also create plots with different parameters
    # fig3 = create_simple_3d_plot(alpha=2, beta=15, save_path="warmth_payoff_alt_params.png")
