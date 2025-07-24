def plot_simulation_results(
    payoffs1: list[float],
    payoffs2: list[float],
    agent1=None,
    agent2=None,
    window: int = 50,
):
    """Create visualization of payoffs and warmth values over time"""
    import matplotlib.pyplot as plt
    import numpy as np

    # Calculate moving averages
    def moving_average(data, window):
        return np.convolve(data, np.ones(window) / window, mode="valid")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top subplot: Payoffs
    ma1 = moving_average(payoffs1, window)
    ma2 = moving_average(payoffs2, window)
    rounds = range(len(payoffs1))
    ma_rounds = range(window - 1, len(payoffs1))

    ax1.plot(rounds, payoffs1, "b.", alpha=0.2, label="Agent 1 (per round)")
    ax1.plot(rounds, payoffs2, "r.", alpha=0.2, label="Agent 2 (per round)")
    ax1.plot(ma_rounds, ma1, "b-", linewidth=2, label=f"Agent 1 ({window}-round avg)")
    ax1.plot(ma_rounds, ma2, "r-", linewidth=2, label=f"Agent 2 ({window}-round avg)")

    ax1.set_ylabel("Payoff per Round")
    ax1.set_title("Per-Round Payoffs with Moving Average")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Bottom subplot: Warmth values (only if agents are provided)
    if (
        agent1
        and agent2
        and hasattr(agent1, "action_history")
        and hasattr(agent2, "action_history")
    ):
        warmth1 = agent1.action_history
        warmth2 = agent2.action_history

        ma_warmth1 = moving_average(warmth1, window)
        ma_warmth2 = moving_average(warmth2, window)

        ax2.plot(rounds, warmth1, "b.", alpha=0.2, label="Agent 1 warmth")
        ax2.plot(rounds, warmth2, "r.", alpha=0.2, label="Agent 2 warmth")
        ax2.plot(
            ma_rounds,
            ma_warmth1,
            "b-",
            linewidth=2,
            label=f"Agent 1 ({window}-round avg)",
        )
        ax2.plot(
            ma_rounds,
            ma_warmth2,
            "r-",
            linewidth=2,
            label=f"Agent 2 ({window}-round avg)",
        )

        ax2.set_xlabel("Round")
        ax2.set_ylabel("Warmth Level")
        ax2.set_title("Warmth Levels Over Time")
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add stats with warmth information
        stats_text = (
            f"Average Payoff/Round:\n"
            f"Agent 1: {np.mean(payoffs1):.2f}\n"
            f"Agent 2: {np.mean(payoffs2):.2f}\n"
            f"Average Warmth:\n"
            f"Agent 1: {np.mean(warmth1):.2f}\n"
            f"Agent 2: {np.mean(warmth2):.2f}"
        )
    else:
        # Stats without warmth information
        stats_text = (
            f"Average Payoff/Round:\n"
            f"Agent 1: {np.mean(payoffs1):.2f}\n"
            f"Agent 2: {np.mean(payoffs2):.2f}"
        )

    ax1.text(
        0.95,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    return fig
