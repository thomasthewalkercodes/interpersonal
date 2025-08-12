from dynamical_analysis_extension import quick_analysis
from interpersonal_dynamics_simulation import InterPersonalSimulation
from dynamical_analysis_extension import enhance_existing_simulation

# Run your existing simulation
sim = InterPersonalSimulation()
sim.add_agent("Agent_Balanced", "balanced")
sim.add_agent("Agent_Anxious", "anxious")
sim.add_agent("Agent_Narcissistic", "narcissistic")
sim.run_simulation(n_rounds=200)

# Get data and analyze
data = sim.get_results_dataframe()
results = quick_analysis(data)

# Simple output
print(f"Mood synchrony: {results['summary_stats']}")
