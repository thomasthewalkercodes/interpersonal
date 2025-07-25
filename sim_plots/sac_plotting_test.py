"""
Test script to debug plotting import issues.
Run this to see what's wrong with the plotting module.
"""

import sys
import os


def test_plotting_import():
    """Test if we can import the plotting module."""

    print("Testing plotting module import...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

    # Check if plotting directory exists
    if os.path.exists("plotting"):
        print("[OK] plotting/ directory exists")

        # Check if __init__.py exists
        if os.path.exists("plotting/__init__.py"):
            print("[OK] plotting/__init__.py exists")
        else:
            print("[MISSING] plotting/__init__.py missing - creating it...")
            with open("plotting/__init__.py", "w") as f:
                f.write('"""Plotting package"""\n')

        # Check if simulation_plots.py exists
        if os.path.exists("plotting/simulation_plots.py"):
            print("[OK] plotting/simulation_plots.py exists")

            # Check file size
            size = os.path.getsize("plotting/simulation_plots.py")
            print(f"  File size: {size} bytes")

            if size == 0:
                print("[ERROR] File is empty! Need to copy content from artifacts.")
                return False

        else:
            print("[MISSING] plotting/simulation_plots.py missing")
            return False
    else:
        print("[MISSING] plotting/ directory missing")
        return False

    # Try to import
    try:
        print("Attempting import...")
        from plotting.simulation_plots import SimulationPlotter

        print("[OK] Successfully imported SimulationPlotter")

        # Test creating instance
        plotter = SimulationPlotter()
        print("[OK] Successfully created SimulationPlotter instance")

        return True

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Other error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dependencies():
    """Test if required plotting dependencies are available."""

    print("\nTesting plotting dependencies...")

    dependencies = ["matplotlib", "numpy", "pandas", "seaborn"]

    missing = []

    for dep in dependencies:
        try:
            __import__(dep)
            print(f"[OK] {dep} available")
        except ImportError:
            print(f"[MISSING] {dep} missing")
            missing.append(dep)

    if missing:
        print(f"\nMissing dependencies: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False

    return True


def test_matplotlib_backend():
    """Test matplotlib backend configuration."""

    print("\nTesting matplotlib backend...")

    try:
        import matplotlib
        import matplotlib.pyplot as plt

        print(f"[OK] Matplotlib version: {matplotlib.__version__}")
        print(f"[OK] Backend: {matplotlib.get_backend()}")

        # Try to create a simple plot
        plt.figure(figsize=(5, 3))
        plt.plot([1, 2, 3], [1, 4, 2])
        plt.title("Test Plot")
        plt.close()

        print("[OK] Basic plotting works")
        return True

    except Exception as e:
        print(f"[ERROR] Matplotlib issue: {e}")
        return False


def create_minimal_plotter():
    """Create a minimal plotter if the full one doesn't work."""

    print("\nCreating minimal plotter...")

    # Create plotting directory if it doesn't exist
    os.makedirs("plotting", exist_ok=True)

    # Create __init__.py
    with open("plotting/__init__.py", "w") as f:
        f.write('"""Plotting package"""\n')

    minimal_code = '''"""
Minimal plotting module for testing.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

class SimulationPlotter:
    """Minimal plotter for testing."""
    
    def __init__(self):
        self.colors = {
            'agent1': '#2E86AB',
            'agent2': '#A23B72'
        }
    
    def create_all_plots(self, results, output_dir):
        """Create basic plots."""
        print(f"Creating plots in: {output_dir}")
        
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Simple reward plot
        try:
            self.plot_simple_rewards(results, plots_dir)
            print("[OK] Created simple reward plot")
        except Exception as e:
            print(f"[ERROR] Error creating plots: {e}")
    
    def plot_simple_rewards(self, results, output_dir):
        """Create a simple reward plot."""
        training_results = results.get("training_results", {})
        episode_rewards = training_results.get("episode_rewards", {})
        
        agent1_rewards = episode_rewards.get("agent1", [])
        agent2_rewards = episode_rewards.get("agent2", [])
        
        if not agent1_rewards or not agent2_rewards:
            print("No reward data to plot")
            return
        
        plt.figure(figsize=(10, 6))
        episodes = range(len(agent1_rewards))
        
        plt.plot(episodes, agent1_rewards, color=self.colors['agent1'], 
                label='Agent 1', alpha=0.7)
        plt.plot(episodes, agent2_rewards, color=self.colors['agent2'], 
                label='Agent 2', alpha=0.7)
        
        plt.title('Episode Rewards Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, 'simple_rewards.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
'''

    try:
        with open("plotting/simulation_plots.py", "w") as f:
            f.write(minimal_code)
        print("[OK] Created minimal plotter")
        return True
    except Exception as e:
        print(f"[ERROR] Error creating minimal plotter: {e}")
        return False


def main():
    """Main test function."""

    print("=" * 60)
    print("PLOTTING MODULE DIAGNOSTIC")
    print("=" * 60)

    # Test dependencies first
    deps_ok = test_dependencies()

    if not deps_ok:
        print("\n[ERROR] Install missing dependencies first!")
        return

    # Test matplotlib
    mpl_ok = test_matplotlib_backend()

    # Test plotting import
    import_ok = test_plotting_import()

    if not import_ok:
        print("\n[FIX] Creating minimal plotter...")
        minimal_ok = create_minimal_plotter()

        if minimal_ok:
            print("[OK] Minimal plotter created. Try your simulation again!")
        else:
            print("[ERROR] Could not create minimal plotter")
    else:
        print("\n[SUCCESS] Plotting module is working!")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dependencies: {'[OK]' if deps_ok else '[ERROR]'}")
    print(f"Matplotlib: {'[OK]' if mpl_ok else '[ERROR]'}")
    print(f"Plotting Import: {'[OK]' if import_ok else '[ERROR]'}")


if __name__ == "__main__":
    main()
