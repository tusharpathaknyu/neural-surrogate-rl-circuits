"""
Simple PPO Agent for Circuit Design.
No external RL library dependencies - pure PyTorch.

Phase C: The RL agent learns to design circuits.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Actor: Outputs mean action (what to do)
    Critic: Outputs state value (how good is this state)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head - outputs action mean
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )
        
        # Learnable log standard deviation
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head - outputs state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, state: torch.Tensor):
        """Forward pass."""
        features = self.shared(state)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, self.actor_log_std, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample an action from the policy."""
        action_mean, log_std, value = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = action_mean
            log_prob = None
        else:
            # Sample from Gaussian
            noise = torch.randn_like(action_mean)
            action = action_mean + std * noise
            
            # Compute log probability
            log_prob = -0.5 * (((action - action_mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
            log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update."""
        action_mean, log_std, values = self.forward(states)
        std = log_std.exp()
        
        # Log probability
        log_prob = -0.5 * (((actions - action_mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1)
        
        # Entropy
        entropy = 0.5 * (1 + np.log(2 * np.pi) + 2 * log_std).sum(dim=-1)
        
        return log_prob, entropy, values.squeeze()


class PPOAgent:
    """
    PPO Agent for circuit design.
    
    Learns a policy that maps target waveforms to circuit parameter adjustments.
    """
    
    def __init__(
        self,
        env,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = 'cpu',
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Create network
        self.policy = ActorCritic(
            env.state_dim, 
            env.action_dim, 
            hidden_dim
        ).to(device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Tracking
        self.episode_rewards = []
        self.episode_mses = []
    
    def collect_rollouts(self, n_steps: int) -> Dict:
        """Collect experience by running policy in environment."""
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        state = self.env.reset()
        episode_reward = 0
        episode_best_mse = float('inf')
        
        self.policy.eval()
        
        for _ in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(state_tensor)
            
            action_np = action.cpu().numpy().squeeze()
            next_state, reward, done, info = self.env.step(action_np)
            
            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            dones.append(done)
            
            episode_reward += reward
            episode_best_mse = min(episode_best_mse, info['mse'])
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_mses.append(episode_best_mse)
                
                state = self.env.reset()
                episode_reward = 0
                episode_best_mse = float('inf')
            else:
                state = next_state
        
        # BUG FIX: Bootstrap last value from critic when episode hasn't ended.
        # Old code always used next_value=0 at rollout boundary, which is wrong
        # when the episode is mid-way through (not terminal).
        last_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, last_value = self.policy.get_action(last_state_tensor)
        last_value = last_value.item()
        
        # Compute advantages using GAE
        advantages, returns = self._compute_gae(rewards, values, dones, last_value)
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'returns': returns,
            'advantages': advantages,
        }
    
    def _compute_gae(self, rewards, values, dones, last_value=0):
        """Compute Generalized Advantage Estimation.
        
        BUG FIX: last_value now comes from critic's bootstrap estimate
        instead of always being 0. When rollout ends mid-episode, we need
        the critic's V(s_next) to properly estimate future returns.
        """
        advantages = np.zeros(len(rewards))
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value  # Bootstrap from critic (was: always 0)
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + np.array(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, rollout: Dict, n_epochs: int = 10, batch_size: int = 64) -> Dict:
        """Update policy using PPO."""
        self.policy.train()
        
        states = torch.FloatTensor(rollout['states']).to(self.device)
        actions = torch.FloatTensor(rollout['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(self.device)
        returns = torch.FloatTensor(rollout['returns']).to(self.device)
        advantages = torch.FloatTensor(rollout['advantages']).to(self.device)
        
        total_loss = 0
        n_updates = 0
        
        n_samples = len(states)
        
        for _ in range(n_epochs):
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                # Get batch
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                
                # Evaluate current policy
                log_probs, entropy, values = self.policy.evaluate(batch_states, batch_actions)
                
                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                n_updates += 1
        
        return {'loss': total_loss / n_updates}
    
    def train(
        self,
        total_timesteps: int,
        n_steps_per_update: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        log_interval: int = 10,
        save_path: Optional[str] = None,
    ):
        """Main training loop."""
        n_updates = total_timesteps // n_steps_per_update
        
        history = {'rewards': [], 'mses': [], 'loss': []}
        
        print(f"Training for {total_timesteps:,} timesteps ({n_updates} updates)")
        print(f"Using surrogate - each step is microseconds, not milliseconds!")
        print()
        
        for update in tqdm(range(n_updates), desc="Training"):
            # Collect experience
            rollout = self.collect_rollouts(n_steps_per_update)
            
            # Update policy
            update_info = self.update(rollout, n_epochs, batch_size)
            
            # Log
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_mse = np.mean(self.episode_mses[-10:])
            else:
                mean_reward = 0
                mean_mse = 1
            
            history['rewards'].append(mean_reward)
            history['mses'].append(mean_mse)
            history['loss'].append(update_info['loss'])
            
            if (update + 1) % log_interval == 0:
                print(f"Update {update+1}/{n_updates} | "
                      f"Reward: {mean_reward:.2f} | "
                      f"MSE: {mean_mse:.6f}")
        
        if save_path:
            self.save(save_path)
            self._plot_training(history, save_path)
        
        return history
    
    def save(self, path: str):
        """Save agent."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"Saved agent to {path}")
    
    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded agent from {path}")
    
    def _plot_training(self, history: Dict, save_path: str):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(history['rewards'])
        axes[0].set_xlabel('Update')
        axes[0].set_ylabel('Mean Reward')
        axes[0].set_title('Episode Rewards')
        axes[0].grid(True)
        
        axes[1].semilogy(history['mses'])
        axes[1].set_xlabel('Update')
        axes[1].set_ylabel('Mean MSE (log)')
        axes[1].set_title('Waveform MSE')
        axes[1].grid(True)
        
        axes[2].plot(history['loss'])
        axes[2].set_xlabel('Update')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('PPO Loss')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.pt', '_training.png'), dpi=150)
        plt.close()
        print(f"Saved training plot")
    
    def design_circuit(
        self,
        target_waveform: np.ndarray,
        max_steps: int = 50,
    ) -> Dict:
        """
        Use trained agent to design a circuit.
        
        This is the INFERENCE function - instant circuit design!
        """
        self.policy.eval()
        
        # Setup environment with target
        self.env.target_waveform = target_waveform
        self.env.current_params = self.env._random_params()
        self.env.current_step = 0
        self.env.prev_mse = None
        
        state = self.env._get_state()
        
        best_params = None
        best_mse = float('inf')
        
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, _, _ = self.policy.get_action(state_tensor, deterministic=True)
            
            action_np = action.cpu().numpy().squeeze()
            state, _, done, info = self.env.step(action_np)
            
            if info['mse'] < best_mse:
                best_mse = info['mse']
                best_params = self.env.current_params.copy()
            
            if done:
                break
        
        return {
            'params': {name: best_params[i] for i, name in enumerate(self.env.PARAM_NAMES)},
            'mse': best_mse,
            'steps': step + 1,
        }


if __name__ == '__main__':
    from environment import CircuitDesignEnv
    from models.forward_surrogate import ForwardSurrogate
    
    print("Testing PPO Agent...")
    
    surrogate = ForwardSurrogate()
    env = CircuitDesignEnv(surrogate)
    agent = PPOAgent(env, hidden_dim=128)
    
    # Quick test
    history = agent.train(
        total_timesteps=4096,
        n_steps_per_update=1024,
        n_epochs=4,
        log_interval=1,
    )
    
    print("✓ PPO Agent test passed!")
