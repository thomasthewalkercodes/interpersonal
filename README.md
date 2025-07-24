# Modular SAC Interpersonal Simulation System

A modular, easy-to-verify system for running interpersonal agent simulations using Soft Actor-Critic (SAC) reinforcement learning. This system is designed with clear separation of concerns and easily testable components.

## 🏗️ System Architecture

```
├── agent_configs/
│   └── sac_agents.py          # Agent personality configurations
├── agent_state.py             # Agent psychological state management
├── control_center/
│   └── sac_control.py         # Main control interface (EMPTY - POPULATE THIS)
├── interfaces.py              # Abstract base classes
├── ml_algos/
│   └── sac_algo.py           # SAC algorithm implementation  
├── payoff_functions/
│   └── gaussian_payoff.py    # Reward calculation functions
├── plotting/
│   └── simulation_plots.py   # Visualization system
└── example_usage.py          # Usage examples and demos
```

## 🚀 Quick Start

### Basic Usage

```python
from control_center.sac_control import SACSControlCenter, SimulationConfig

# Initialize control center
control = SACSControlCenter()

# Run a quick simulation
results = control.quick_run('cooperative', 'competitive', episodes=500)

# Or use detailed configuration
config = SimulationConfig(
    agent1_type='adaptive',
    agent2_type='cautious',
    episodes=1000,
    save_plots=True,
    run_name="my_experiment"
)

results = control.run_simulation(config)
```

### Available Agent Types

- **cooperative**: Higher initial trust, less exploration, forgiving memory
- **competitive**: Lower initial trust, more exploration, longer memory  
- **adaptive**: Fast learning, quick adjustments, balanced approach
- **cautious**: Slow learning, conservative exploration, very long memory
- **base**: Default configuration with standard parameters

### Custom Agent Configuration

```python
config = SimulationConfig(
    agent1_type='base',
    agent2_type='base',
    agent1_custom_params={
        'lr_actor': 1e-3,
        'initial_trust': 0.8,
        'memory_length': 20
    },
    agent2_custom_params={
        'lr_actor': 5e-4,
        'initial_trust': -0.5,
        'memory_length': 80
    },
    episodes=500,
    payoff_alpha=3.0,  # Mismatch penalty
    payoff_beta=8.0,   # Risk penalty
    run_name="trust_experiment"
)
```

## 📊 Plotting and Visualization

The plotting system automatically generates comprehensive visualizations:

### Automatic Plots Generated
- **Training Progress**: Actor/critic losses, temperature evolution
- **Reward Dynamics**: Episode rewards, smoothed curves, distributions
- **Action Patterns**: Warmth levels over time, correlation analysis
- **Learning Curves**: Convergence analysis, stability metrics
- **Agent Comparison**: Performance comparison, learning speed
- **Summary Dashboard**: Complete overview with key metrics

### Manual Plotting
```python
from plotting.simulation_plots import SimulationPlotter

plotter = SimulationPlotter()

# Generate specific plots from saved results
plot_specific_results("./results/my_run/results.json", ["training", "rewards"])

# Create payoff landscape visualization
plotter.plot_payoff_landscape(alpha=4.0, beta=10.0, output_dir="./payoff_plots")
```

## 🔧 Modular Components

### 1. Agent Configurations (`agent_configs/sac_agents.py`)
Defines different personality types with specific SAC hyperparameters:
- Learning rates, exploration noise, memory length
- Initial trust and satisfaction levels  
- Reward discounting and update frequencies

### 2. Agent State Management (`agent_state.py`)
Implements psychological state tracking:
- **InterpersonalAgentState**: Basic trust, satisfaction, arousal tracking
- **AdaptiveAgentState**: Enhanced with meta-learning capabilities
- **GroupAgentState**: Multi-agent relationship management

### 3. SAC Algorithm (`ml_algos/sac_algo.py`)
Complete SAC implementation with:
- Actor-critic networks with target networks
- Automatic entropy tuning
- Experience replay buffer
- Soft updates and training metrics

### 4. Payoff Functions (`payoff_functions/gaussian_payoff.py`)
Configurable reward calculation:
- Gaussian-based warmth interaction payoff
- Mismatch penalty (α parameter)
- Risk penalty (β parameter)
- Visualization tools

### 5. Control Center (`control_center/sac_control.py`)
Main interface for running simulations:
- Simple and complex configuration options
- Batch comparison studies
- Automatic result saving and plot generation

## 🧪 Experimental Examples

Run the examples to see the system in action:

```bash
# Basic interaction
python example_usage.py 1

# Custom agent parameters  
python example_usage.py 2

# Systematic comparison
python example_usage.py 3

# Parameter sweep
python example_usage.py 4

# Longitudinal analysis
python example_usage.py 5

# Interactive demo
python example_usage.py 6

# All examples
python example_usage.py all

# Payoff visualizations
python example_usage.py payoff
```

## 📈 Understanding Results

### Key Metrics
- **Episode Rewards**: Total reward per episode for each agent
- **Final Evaluation**: Average performance over multiple test episodes
- **Trust Evolution**: How trust levels change over time
- **Action Patterns**: Warmth levels and correlation between agents
- **Learning Stability**: Variance in performance over time

### Interpreting Agent Behavior
- **High Trust + High Warmth**: Cooperative, mutually beneficial interaction
- **Low Trust + Low Warmth**: Defensive, minimal interaction
- **High Trust + Low Warmth**: Exploitative relationship  
- **Low Trust + High Warmth**: Risk-taking despite suspicion

## 🔬 Research Applications

### Hypothesis Testing
```python
# Test: Do cooperative agents perform better against each other?
comparison_results = control.run_comparison_study(
    agent_types=['cooperative', 'competitive'],
    base_config=base_config,
    num_runs=5
)

# Test: How does payoff structure affect cooperation?
for alpha in [2.0, 4.0, 6.0]:
    config.payoff_alpha = alpha
    results = control.run_simulation(config)
    # Analyze cooperation levels...
```

### Longitudinal Studies
```python
# Study long-term adaptation patterns
config = SimulationConfig(
    agent1_type='adaptive',
    agent2_type='adaptive', 
    episodes=5000,  # Long-term study
    evaluation_frequency=100
)
```

### Parameter Sensitivity Analysis
```python
# Systematic parameter exploration
for lr in [1e-4, 3e-4, 1e-3]:
    for memory in [20, 50, 100]:
        custom_params = {'lr_actor': lr, 'memory_length': memory}
        # Run simulation with these parameters...
```

## 🧩 Extending the System

### Adding New Agent Types
```python
class MyCustomAgentConfig(BaseAgentConfig):
    def __init__(self, **kwargs):
        defaults = {
            'lr_actor': 2e-4,
            'initial_trust': 0.2,
            'memory_length': 40,
            # ... other parameters
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

# Register in SACSControlCenter.AGENT_TYPES
```

### Adding New Payoff Functions
```python
def my_payoff_function(w1: float, w2: float, **params) -> float:
    # Implement your payoff logic
    return payoff_value

# Use in environment
environment = SimpleEnvironment(payoff_function=my_payoff_function)
```

### Adding New Metrics
```python
class CustomMetricsCollector(MetricsCollector):
    def record_episode(self, episode, agent1_reward, agent2_reward, **kwargs):
        # Implement custom metric collection
        pass
```

## 📂 File Structure Details

### Core Implementation Files
- **`interfaces.py`**: Abstract base classes defining the system contracts
- **`agent_state.py`**: Psychological state management with trust/satisfaction tracking
- **`sac_control.py`**: Main control interface (YOU NEED TO POPULATE THIS)

### Configuration Files  
- **`sac_agents.py`**: Pre-defined agent personality configurations
- **`gaussian_payoff.py`**: Warmth-based interaction payoff functions

### Algorithm Files
- **`sac_algo.py`**: Complete SAC implementation with PyTorch networks

### Visualization Files
- **`simulation_plots.py`**: Comprehensive plotting and analysis tools

### Example Files
- **`example_usage.py`**: Demonstrates all major use cases

## ⚙️ System Requirements

### Dependencies
```bash
pip install torch numpy matplotlib seaborn pandas
```

### Optional Dependencies
```bash
pip install jupyter  # For notebook-based analysis
pip install plotly   # For interactive plots
```

## 🔍 Verification and Testing

Each component is designed to be easily verifiable:

### Component Testing
```python
# Test agent configuration
config = CooperativeAgentConfig()
assert config.initial_trust > 0
assert config.lr_actor < BaseAgentConfig().lr_actor

# Test agent state
state = InterpersonalAgentState()
initial_state = state.get_state_vector()
state.update(0.5, 0.3, 0.8)
updated_state = state.get_state_vector()
assert not np.array_equal(initial_state, updated_state)

# Test payoff function
payoff = calculate_warmth_payoff(0.8, 0.8, alpha=4.0, beta=10.0)
assert payoff > 0  # Matching high warmth should give positive payoff
```

### Integration Testing
```python
# Test full pipeline
control = SACSControlCenter()
config = SimulationConfig(
    agent1_type='cooperative',
    agent2_type='competitive',
    episodes=10,  # Quick test
    save_models=False,
    save_plots=False
)
results = control.run_simulation(config)
assert 'final_evaluation' in results
assert 'training_results' in results
```

## 📝 Key Design Principles

### 1. Modularity
- Each component has a single, clear responsibility
- Components communicate through well-defined interfaces
- Easy to swap implementations without affecting other parts

### 2. Verifiability  
- Each function/class has clear inputs and outputs
- Minimal side effects and hidden state
- Comprehensive logging and metric collection

### 3. Extensibility
- Abstract base classes allow easy addition of new components
- Configuration-driven behavior reduces code duplication
- Plugin-style architecture for payoff functions and agent types

### 4. Reproducibility
- All configurations saved automatically
- Random seeds managed consistently
- Complete experiment provenance tracking

## 🎯 Next Steps

### Immediate Actions
1. **Populate `control_center/sac_control.py`** with the provided implementation
2. **Create the `plotting/` directory** and add `simulation_plots.py`
3. **Run examples** to verify everything works
4. **Customize** agent types and payoff functions for your research

### Advanced Usage
1. **Multi-agent scenarios**: Extend to 3+ agents using `GroupAgentState`
2. **Custom environments**: Implement new interaction environments
3. **Advanced analysis**: Add statistical testing and hypothesis validation
4. **Real-time visualization**: Create live plotting during training

### Research Extensions
1. **Personality trait modeling**: Map SAC parameters to psychological constructs
2. **Cultural variation**: Implement different payoff structures
3. **Developmental studies**: Track long-term personality change
4. **Intervention studies**: Test different "therapy" interventions

## 🤝 Contributing

The modular design makes it easy to contribute:
- Add new agent types in `agent_configs/`
- Implement new payoff functions in `payoff_functions/`
- Create new visualization in `plotting/`
- Add new state tracking in `agent_state.py`

## 📞 Support

For questions about:
- **Agent configuration**: See `agent_configs/sac_agents.py` and examples
- **Simulation setup**: Check `example_usage.py` for patterns
- **Plotting options**: Review `plotting/simulation_plots.py` documentation
- **Research applications**: See the CIIT framework in project knowledge

---

**Ready to run interpersonal simulations with clear, verifiable components!** 🎉
