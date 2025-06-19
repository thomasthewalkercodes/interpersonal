import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque  # Fixed incomplete import
from typing import Dict, Any, List, Tuple, Optional  # Added Optional
from interfaces import ReinforcementLearner


class ReplayBuffer:
    """Experience replay buffer for SAC"""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim

    def push(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences from the buffer."""
        batch = random

        states = torch.Floattensor([e[0] for e in batch])
        actions = torch.FloatTensor([e[1] for e in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([e[2] for e in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch]).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return


class Actor(nn.Module):
    """Actor network for SAC"""

    def __init__(self, state_dim: int, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, 1)
        self.log_std = nn.Linear(hidden_size, 1)

        # action bounds for warmth
        self.action_scale = 1.0
        self.action_bias = 0.0

        # limits for log_std
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the actor network."""
        x = F.relu(self.fc1(state))
        y = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action from the actor network."""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick

        # apply tanh squashing
        action = torch.tanh(x_t)

        # calculate log probability
        log_prob = normal.log_prob(x_t)
        # enforce action bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        # scale action to [-1, 1]
        action = action * self.action_scale + self.action_bias

        return action, log_prob


class Critic(nn.Module):
    """Critic network for SAC."""

    def __init__(self, state_dim: int, hidden_size: int = 256):
        super().__init__()
        # Q1 network
        self.q1_fc1 = nn.Linear(state_dim + 1, hidden_size)  # +1 for action
        self.q1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1_fc3 = nn.Linear(hidden_size, 1)

        # Q2 network
        self.q2_fc1 = nn.Linear(state_dim + 1, hidden_size)
        self.q2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q2_fc3 = nn.Linear(hidden_size, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both Q networks."""
        sa = torch.cat([state, action], 1)

        # Q1
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        # Q2
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)

        return q1, q2


class SACAgent(ReinforcementLearner):
    """Soft Actor-Critic agent implementation."""

    def __init__(self, state_dim: int, config: Dict[str, Any]):
        """
        Initialize SAC agent.

        Args:
            state_dim: Dimension of state space
            config: Dictionary containing SAC hyperparameters
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim

        # Extract hyperparameters
        self.lr_actor = config.get("lr_actor", 3e-4)
        self.lr_critic = config.get("lr_critic", 3e-4)
        self.lr_temperature = config.get("lr_temperature", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.batch_size = config.get("batch_size", 256)
        self.buffer_size = config.get("buffer_size", 100000)
        self.hidden_size = config.get("hidden_size", 256)
        self.noise_scale = config.get("noise_scale", 0.1)

        # Temperature parameter (alpha)
        self.alpha = config.get("alpha", 0.2)
        self.target_entropy = config.get("target_entropy", -1.0)
        self.automatic_entropy_tuning = True

        # Initialize networks
        self.actor = Actor(state_dim, self.hidden_size).to(self.device)
        self.critic = Critic(state_dim, self.hidden_size).to(self.device)
        self.critic_target = Critic(state_dim, self.hidden_size).to(self.device)

        # Copy parameters to target network
        self.hard_update(self.critic_target, self.critic)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Temperature parameter
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_temperature)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, state_dim)

        # Training metrics
        self.train_metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "alpha_loss": 0.0,
            "alpha": self.alpha,
        }

    def select_action(self, state: np.ndarray, training: bool = True) -> float:
        """Select action based on current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if training:
            with torch.no_grad():
                action, _ = self.actor.sample(state_tensor)
                # Add exploration noise
                action += torch.normal(0, self.noise_scale, action.shape).to(
                    self.device
                )
                action = torch.clamp(action, -1.0, 1.0)
        else:
            with torch.no_grad():
                mean, _ = self.actor.forward(state_tensor)
                action = torch.tanh(mean)

        return action.cpu().numpy()[0, 0]

    def store_transition(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update critic
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)

        # Update actor and temperature
        actor_loss, alpha_loss = self._update_actor_and_temperature(states)

        # Soft update target networks
        self.soft_update(self.critic_target, self.critic, self.tau)

        # Update metrics
        self.train_metrics.update(
            {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "alpha_loss": alpha_loss,
                "alpha": self.alpha,
            }
        )

        return self.train_metrics.copy()

    def _update_critic(self, states, actions, rewards, next_states, dones) -> float:
        """Update critic networks."""
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones.float()) * self.gamma * q_next

        # Current Q values
        q1_current, q2_current = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(q1_current, target_q) + F.mse_loss(
            q2_current, target_q
        )

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor_and_temperature(self, states) -> Tuple[float, float]:
        """Update actor network and temperature parameter."""
        # Sample actions from current policy
        actions, log_probs = self.actor.sample(states)

        # Q values for sampled actions
        q1, q2 = self.critic(states, actions)
        q_min = torch.min(q1, q2)

        # Actor loss
        actor_loss = (self.alpha * log_probs - q_min).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature parameter
        alpha_loss = 0.0
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            alpha_loss = alpha_loss.item()

        return actor_loss.item(), alpha_loss

    def soft_update(self, target, source, tau):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        """Hard update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha if self.automatic_entropy_tuning else None,
                "alpha_optimizer_state_dict": (
                    self.alpha_optimizer.state_dict()
                    if self.automatic_entropy_tuning
                    else None
                ),
                "alpha": self.alpha,
                "config": {
                    "state_dim": self.state_dim,
                    "lr_actor": self.lr_actor,
                    "lr_critic": self.lr_critic,
                    "lr_temperature": self.lr_temperature,
                    "gamma": self.gamma,
                    "tau": self.tau,
                    "batch_size": self.batch_size,
                    "buffer_size": self.buffer_size,
                    "hidden_size": self.hidden_size,
                    "noise_scale": self.noise_scale,
                    "target_entropy": self.target_entropy,
                },
            },
            filepath,
        )

    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer.state_dict"])

        if self.automatic_entropy_tuning and checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(
                checkpoint["alpha_optimizer_state_dict"]
            )

        self.alpha = checkpoint["alpha"]


class SACTrainer:
    """Trainer class for managing SAC training process."""

    def __init__(
        self,
        agent1: SACAgent,
        agent2: SACAgent,
        environment,
        payoff_calculator,
        episodes_per_training: int = 1000,
        steps_per_episode: int = 50,
        training_frequency: int = 1,
        evaluation_frequency: int = 100,
        save_frequency: int = 1000,
    ):
        """
        Initialize SAC trainer.

        Args:
            agent1: First SAC agent
            agent2: Second SAC agent
            environment: Interaction environment
            payoff_calculator: Payoff calculation function
            episodes_per_training: Number of episodes per training session
            steps_per_episode: Number of steps per episode
            training_frequency: Train every N steps
            evaluation_frequency: Evaluate every N episodes
            save_frequency: Save models every N episodes
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.environment = environment
        self.payoff_calculator = payoff_calculator

        self.episodes_per_training = episodes_per_training
        self.steps_per_episode = steps_per_episode
        self.training_frequency = training_frequency
        self.evaluation_frequency = evaluation_frequency
        self.save_frequency = save_frequency

        # Training statistics
        self.episode_rewards = {"agent1": [], "agent2": []}
        self.training_metrics = {"agent1": [], "agent2": []}
        self.step_count = 0

    def train(self, save_dir: str = "./models") -> Dict[str, Any]:
        """
        Train both agents simultaneously.

        Args:
            save_dir: Directory to save models

        Returns:
            Dictionary containing training statistics
        """
        import os

        os.makedirs(save_dir, exist_ok=True)

        for episode in range(self.episodes_per_training):
            # Reset environment
            state1, state2 = self.environment.reset()

            episode_reward1 = 0
            episode_reward2 = 0

            for step in range(self.steps_per_episode):
                # Select actions
                action1 = self.agent1.select_action(state1, training=True)
                action2 = self.agent2.select_action(state2, training=True)

                # Execute actions and get rewards
                next_state1, next_state2, reward1, reward2, done = (
                    self.environment.step(action1, action2)
                )

                # Store transitions
                self.agent1.store_transition(
                    state1, action1, reward1, next_state1, done
                )
                self.agent2.store_transition(
                    state2, action2, reward2, next_state2, done
                )

                # Update episode rewards
                episode_reward1 += reward1
                episode_reward2 += reward2

                # Train agents
                if self.step_count % self.training_frequency == 0:
                    metrics1 = self.agent1.train_step()
                    metrics2 = self.agent2.train_step()

                    if metrics1:
                        self.training_metrics["agent1"].append(metrics1)
                    if metrics2:
                        self.training_metrics["agent2"].append(metrics2)

                # Update states
                state1 = next_state1
                state2 = next_state2
                self.step_count += 1

                if done:
                    break

            # Store episode rewards
            self.episode_rewards["agent1"].append(episode_reward1)
            self.episode_rewards["agent2"].append(episode_reward2)

            # Evaluation
            if episode % self.evaluation_frequency == 0:
                eval_results = self.evaluate(num_episodes=10)
                print(
                    f"Episode {episode}: Agent1 Avg Reward: {eval_results['agent1_avg_reward']:.2f}, "
                    f"Agent2 Avg Reward: {eval_results['agent2_avg_reward']:.2f}"
                )

            # Save models
            if episode % self.save_frequency == 0 and episode > 0:
                self.agent1.save_model(f"{save_dir}/agent1_episode_{episode}.pth")
                self.agent2.save_model(f"{save_dir}/agent2_episode_{episode}.pth")

        # Final save
        self.agent1.save_model(f"{save_dir}/agent1_final.pth")
        self.agent2.save_model(f"{save_dir}/agent2_final.pth")

        return {
            "episode_rewards": self.episode_rewards,
            "training_metrics": self.training_metrics,
            "total_episodes": self.episodes_per_training,
            "total_steps": self.step_count,
        }

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agents without training."""
        total_reward1 = 0
        total_reward2 = 0

        for _ in range(num_episodes):
            state1, state2 = self.environment.reset()
            episode_reward1 = 0
            episode_reward2 = 0

            for _ in range(self.steps_per_episode):
                # Select actions without exploration
                action1 = self.agent1.select_action(state1, training=False)
                action2 = self.agent2.select_action(state2, training=False)

                # Execute actions
                next_state1, next_state2, reward1, reward2, done = (
                    self.environment.step(action1, action2)
                )

                episode_reward1 += reward1
                episode_reward2 += reward2

                state1 = next_state1
                state2 = next_state2

                if done:
                    break

            total_reward1 += episode_reward1
            total_reward2 += episode_reward2

        return {
            "agent1_avg_reward": total_reward1 / num_episodes,
            "agent2_avg_reward": total_reward2 / num_episodes,
        }
