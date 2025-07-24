"""
Example usage of the SAC interpersonal behavior simulation system.
Demonstrates how to create agents, train them, and analyze results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import os

# Import our modules
from interfaces import PayoffCalculator
from agent_config import (
    BaseAgentConfig, CooperativeAgentConfig, 
    CompetitiveAgentConfig, AdaptiveAgentConfig
)
from agent_state import InterpersonalAgentState
from sac_algorithm import SACAgent, SACTrainer
from interaction_environment import InterpersonalEnvironment, SimplePayoffCalculator


class ExampleRunner:
    """Class to demonstrate the SAC system usage."""
    
    def __init__(self):
        self.results = {}
    
    def create_agent_pair(self, config1, config2, agent1_id="agent1", agent2_id="agent2"):
        """
        Create a pair of agents with different configurations.
        
        Args:
            config1: Configuration for first agent
            config2: Configuration for second agent
            agent1_id: ID for first agent
            agent2_id: ID for second agent
            
        Returns:
            Tuple of (agent1, agent2, environment)
        """
        # Create agent states
        state1 = config1.create_initial_state()
        state2 = config2.create_initial_state()
        
        # Get state dimension
        state_dim = state1.get_state_dimension()
        
        # Create SAC agents
        agent1 = SACAgent(state_dim, config1.get_sac_params())
        agent2 = SACAgent(state_dim, config2.get_sac_params())
        
        # Create payoff calculator (you can replace this with your own)
        payoff_calculator = SimplePayoffCalculator(
            cooperation_bonus=5.0,
            betrayal_penalty=-3.0,
            neutral_payoff=1.0
        )
        
        # Create environment
        environment = InterpersonalEnvironment(
            payoff_calculator=payoff_calculator,
            agent1_state=state1,
            agent2_state=state2,
            agent1_id=agent1_id,
            agent2_id=agent2_id,
            max_steps_per_episode=50
        )
        
        return agent1, agent2, environment
    
    def train_agents(self, agent1, agent2, environment, 
                    experiment_name: str, episodes: int = 2000):
        """
        Train a pair of agents and save results.
        
        Args:
            agent1: First SAC agent
            agent2: Second SAC agent
            environment: Interaction environment
            experiment_name: Name for this experiment
            episodes: Number of training episodes
        """
        print(f"Training {experiment_name}...")
        
        # Create trainer
        trainer = SACTrainer(
            agent1=agent1,
            agent2=agent2,
            environment=environment,
            payoff_calculator=environment.payoff_calculator,
            episodes_per_training=episodes,
            steps_per_episode=50,
            evaluation_frequency=200,
            save_frequency=500
        )
        
        # Train agents
        save_dir = f"./models/{experiment_name}"
        results = trainer.train(save_dir)
        
        # Store results
        self.results[experiment_name] = results
        
        print(f"Training {experiment_name} completed!")
        return results
    
    def analyze_results(self, experiment_name: str):
        """Analyze and plot training results."""
        if experiment_name not in self.results:
            print(f"No results found for {experiment_name}")
            return
        
        results = self.results[experiment_name]
        
        # Plot episode rewards
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Episode rewards
        plt.subplot(1, 3, 1)
        episodes = range(len(results['episode_rewards']['agent1']))
        plt.plot(episodes, results['episode_rewards']['agent1'], label='Agent 1', alpha=0.7)
        plt.plot(episodes, results['episode_rewards']['agent2'], label='Agent 2', alpha=0.7)
        
        # Add moving averages
        window = 50
        if len(episodes) > window:
            avg1 = np.convolve(results['episode_rewards']['agent1'], 
                              np.ones(window)/window, mode='valid')
            avg2 = np.convolve(results['episode_rewards']['agent2'], 
                              np.ones(window)/window, mode='valid')
            avg_episodes = episodes[window-1:]
            plt.plot(actor_losses2, label='Agent 2 Actor Loss', alpha=0.7)
            plt.plot(critic_losses2, label='Agent 2 Critic Loss', alpha=0.7)
        
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title(f'{experiment_name}: Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot 3: Summary statistics
        plt.subplot(1, 3, 3)
        final_rewards1 = results['episode_rewards']['agent1'][-100:]  # Last 100 episodes
        final_rewards2 = results['episode_rewards']['agent2'][-100:]
        
        labels = ['Agent 1', 'Agent 2']
        means = [np.mean(final_rewards1), np.mean(final_rewards2)]
        stds = [np.std(final_rewards1), np.std(final_rewards2)]
        
        plt.bar(labels, means, yerr=stds, capsize=5, alpha=0.7)
        plt.ylabel('Average Reward (Last 100 Episodes)')
        plt.title(f'{experiment_name}: Final Performance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'./results_{experiment_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\n=== {experiment_name} Results ===")
        print(f"Total Episodes: {len(results['episode_rewards']['agent1'])}")
        print(f"Agent 1 - Final Avg Reward: {np.mean(final_rewards1):.2f} ± {np.std(final_rewards1):.2f}")
        print(f"Agent 2 - Final Avg Reward: {np.mean(final_rewards2):.2f} ± {np.std(final_rewards2):.2f}")
        
        if results['training_metrics']['agent1']:
            final_alpha1 = results['training_metrics']['agent1'][-1]['alpha']
            print(f"Agent 1 - Final Temperature: {final_alpha1:.3f}")
        if results['training_metrics']['agent2']:
            final_alpha2 = results['training_metrics']['agent2'][-1]['alpha']
            print(f"Agent 2 - Final Temperature: {final_alpha2:.3f}")
    
    def run_experiments(self):
        """Run a series of experiments with different agent configurations."""
        experiments = [
            {
                'name': 'cooperative_vs_competitive',
                'config1': CooperativeAgentConfig(memory_length=30),
                'config2': CompetitiveAgentConfig(memory_length=70),
                'episodes': 2000
            },
            {
                'name': 'adaptive_vs_cautious',
                'config1': AdaptiveAgentConfig(),
                'config2': AdaptiveAgentConfig(
                    lr_actor=1e-4, lr_critic=1e-4, alpha=0.05,
                    memory_length=100, noise_scale=0.02
                ),
                'episodes': 2000
            },
            {
                'name': 'symmetric_cooperative',
                'config1': CooperativeAgentConfig(),
                'config2': CooperativeAgentConfig(),
                'episodes': 1500
            },
            {
                'name': 'symmetric_competitive',
                'config1': CompetitiveAgentConfig(),
                'config2': CompetitiveAgentConfig(),
                'episodes': 1500
            }
        ]
        
        for exp in experiments:
            try:
                print(f"\n=== Starting Experiment: {exp['name']} ===")
                
                # Create agents
                agent1, agent2, environment = self.create_agent_pair(
                    exp['config1'], exp['config2'], 
                    f"{exp['name']}_agent1", f"{exp['name']}_agent2"
                )
                
                # Train agents
                self.train_agents(agent1, agent2, environment, 
                                exp['name'], exp['episodes'])
                
                # Analyze results
                self.analyze_results(exp['name'])
                
            except Exception as e:
                print(f"Error running {exp['name']}: {e}")
                print("Make sure all required modules are available and properly implemented.")
    
    def test_single_interaction(self):
        """Test a single interaction to verify the system works."""
        print("Testing single interaction...")
        
        # Create simple configs
        config1 = BaseAgentConfig(memory_length=10)
        config2 = BaseAgentConfig(memory_length=10)
        
        # Create agents
        agent1, agent2, environment = self.create_agent_pair(config1, config2)
        
        # Reset environment
        state1, state2 = environment.reset()
        
        print(f"Initial state dimensions: {len(state1)}, {len(state2)}")
        print(f"State 1 sample: {state1[:5]}...")
        print(f"State 2 sample: {state2[:5]}...")
        
        # Test one step
        action1 = agent1.select_action(state1, training=False)
        action2 = agent2.select_action(state2, training=False)
        
        print(f"Actions: Agent1={action1:.3f}, Agent2={action2:.3f}")
        
        next_state1, next_state2, reward1, reward2, done = environment.step(action1, action2)
        
        print(f"Rewards: Agent1={reward1:.3f}, Agent2={reward2:.3f}")
        print(f"Done: {done}")
        print("Single interaction test completed successfully!")


def main():
    """Main function to run the SAC interpersonal behavior simulation."""
    print("SAC Interpersonal Behavior Simulation")
    print("=====================================")
    
    # Create runner
    runner = ExampleRunner()
    
    # Test single interaction first
    runner.test_single_interaction()
    
    # Run full experiments
    runner.run_experiments()
    
    print("\nAll experiments completed!")
    print("Check the generated plots and model files for results.")


if __name__ == "__main__":
    main()(avg_episodes, avg1, '--', linewidth=2, label='Agent 1 (avg)')
            plt.plot(avg_episodes, avg2, '--', linewidth=2, label='Agent 2 (avg)')
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'{experiment_name}: Episode Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Training losses (if available)
        plt.subplot(1, 3, 2)
        if results['training_metrics']['agent1']:
            actor_losses1 = [m['actor_loss'] for m in results['training_metrics']['agent1']]
            critic_losses1 = [m['critic_loss'] for m in results['training_metrics']['agent1']]
            
            plt.plot(actor_losses1, label='Agent 1 Actor Loss', alpha=0.7)
            plt.plot(critic_losses1, label='Agent 1 Critic Loss', alpha=0.7)
        
        if results['training_metrics']['agent2']:
            actor_losses2 = [m['actor_loss'] for m in results['training_metrics']['agent2']]
            critic_losses2 = [m['critic_loss'] for m in results['training_metrics']['agent2']]
            
            plt.plot