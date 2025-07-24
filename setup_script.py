"""
Setup script to create the proper directory structure and files
for the modular SAC interpersonal simulation system.

Run this script from your project root directory.
"""

import os


def create_directory_structure():
    """Create the required directory structure."""

    directories = ["control_center", "plotting", "results", "models"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

        # Create __init__.py files to make them Python packages
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write(f'"""Package: {directory}"""\n')
            print(f"Created: {init_file}")


def check_required_files():
    """Check which files need to be created."""

    required_files = {
        "interfaces.py": "Abstract base classes",
        "agent_state.py": "Agent psychological state management",
        "control_center/sac_control.py": "Main control interface",
        "plotting/simulation_plots.py": "Visualization system",
        "example_usage.py": "Usage examples and demos",
    }

    missing_files = []
    existing_files = []

    for filepath, description in required_files.items():
        if os.path.exists(filepath):
            # Check if file is empty
            if os.path.getsize(filepath) == 0:
                missing_files.append((filepath, description, "empty"))
            else:
                existing_files.append((filepath, description))
        else:
            missing_files.append((filepath, description, "missing"))

    print("\n" + "=" * 60)
    print("FILE STATUS CHECK")
    print("=" * 60)

    if existing_files:
        print("\n[OK] EXISTING FILES:")
        for filepath, description in existing_files:
            size = os.path.getsize(filepath)
            print(f"  {filepath} ({size} bytes) - {description}")

    if missing_files:
        print("\n[MISSING] MISSING/EMPTY FILES:")
        for filepath, description, status in missing_files:
            print(f"  {filepath} ({status}) - {description}")
    else:
        print("\n[OK] All required files exist!")

    return missing_files


def create_minimal_files():
    """Create minimal versions of required files if they don't exist."""

    # Create interfaces.py if missing
    if not os.path.exists("interfaces.py"):
        print("Creating interfaces.py...")
        with open("interfaces.py", "w") as f:
            f.write('"""Abstract interfaces - COPY CONTENT FROM ARTIFACTS"""\n')
            f.write("# TODO: Copy the interfaces.py content from Claude artifacts\n")

    # Create agent_state.py if missing
    if not os.path.exists("agent_state.py"):
        print("Creating agent_state.py...")
        with open("agent_state.py", "w") as f:
            f.write('"""Agent state management - COPY CONTENT FROM ARTIFACTS"""\n')
            f.write("# TODO: Copy the agent_state.py content from Claude artifacts\n")

    # Create control_center/sac_control.py if missing or empty
    control_file = "control_center/sac_control.py"
    if not os.path.exists(control_file) or os.path.getsize(control_file) == 0:
        print("Creating control_center/sac_control.py...")
        with open(control_file, "w") as f:
            f.write('"""SAC Control Center - COPY CONTENT FROM ARTIFACTS"""\n')
            f.write("# TODO: Copy the sac_control.py content from Claude artifacts\n")

    # Create plotting/simulation_plots.py if missing
    plots_file = "plotting/simulation_plots.py"
    if not os.path.exists(plots_file):
        print("Creating plotting/simulation_plots.py...")
        with open(plots_file, "w") as f:
            f.write('"""Simulation plots - COPY CONTENT FROM ARTIFACTS"""\n')
            f.write(
                "# TODO: Copy the simulation_plots.py content from Claude artifacts\n"
            )

    # Create example_usage.py if missing
    if not os.path.exists("example_usage.py"):
        print("Creating example_usage.py...")
        with open("example_usage.py", "w") as f:
            f.write('"""Example usage - COPY CONTENT FROM ARTIFACTS"""\n')
            f.write("# TODO: Copy the example_usage.py content from Claude artifacts\n")


def print_next_steps():
    """Print instructions for next steps."""

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)

    steps = [
        "1. Copy the content from Claude's artifacts into the respective files:",
        "   - interfaces.py",
        "   - agent_state.py",
        "   - control_center/sac_control.py",
        "   - plotting/simulation_plots.py",
        "   - example_usage.py",
        "",
        "2. Make sure you have the required dependencies:",
        "   pip install torch numpy matplotlib seaborn pandas",
        "",
        "3. Test the setup:",
        "   python example_usage.py 1",
        "",
        "4. If you get import errors, check that all files have content",
        "   and that __init__.py files exist in each directory.",
    ]

    for step in steps:
        print(step)


def main():
    """Main setup function."""
    print("Setting up modular SAC interpersonal simulation system...")
    print("Current directory:", os.getcwd())

    # Create directory structure
    create_directory_structure()

    # Check file status
    missing_files = check_required_files()

    # Create minimal files
    create_minimal_files()

    # Print next steps
    print_next_steps()

    if missing_files:
        print(f"\n[WARNING] Setup incomplete: {len(missing_files)} files need content")
        print("Copy the artifacts from Claude to complete the setup.")
    else:
        print("\n[SUCCESS] Setup complete! You can now run:")
        print("python example_usage.py 1")


if __name__ == "__main__":
    main()
