"""
Generate demo visuals for LinkedIn / README:
1. Animated GIF: RL agent optimizing a waveform step-by-step (random -> converged)
2. Baseline comparison: Random Search vs RL Agent across topologies
3. Before/After static plot

Usage: python generate_demo_visuals.py
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.multi_topology_surrogate import load_trained_model
from rl.environment import CircuitDesignEnv
from rl.ppo_agent import PPOAgent, ActorCritic
from train_intensive_spice import create_target_waveform, TOPOLOGY_CONFIG

DEVICE = 'cpu'


def load_agent(topology, surrogate):
    """Load a trained RL agent for a given topology."""
    config = TOPOLOGY_CONFIG[topology]
    env = CircuitDesignEnv(
        surrogate,
        device=DEVICE,
        topology=topology,
        use_spice_reward=False,
    )
    agent = PPOAgent(
        env,
        hidden_dim=config['hidden_dim'],
        lr=config['lr'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_epsilon=config['clip_epsilon'],
        entropy_coef=config['entropy_coef'],
        device=DEVICE,
    )
    ckpt_path = f'checkpoints/rl_agent_{topology}.pt'
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)
        return agent
    return None


def run_agent_with_trajectory(agent, target_waveform, max_steps=50):
    """Run the agent and record the waveform at every step."""
    agent.policy.eval()
    env = agent.env

    env.target_waveform = target_waveform.copy()
    env.current_params = env._random_params()
    env.current_step = 0
    env.prev_mse = None

    state = env._get_state()

    trajectory = []
    params_history = []
    mse_history = []

    # Record initial state
    initial_waveform = env._simulate(env.current_params)
    initial_mse = np.mean((initial_waveform - target_waveform) ** 2)
    trajectory.append(initial_waveform.copy())
    params_history.append(env.current_params.copy())
    mse_history.append(initial_mse)

    for step in range(max_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, _, _ = agent.policy.get_action(state_tensor, deterministic=True)
        action_np = action.cpu().numpy().squeeze()
        state, _, done, info = env.step(action_np)

        current_waveform = env._simulate(env.current_params)
        trajectory.append(current_waveform.copy())
        params_history.append(env.current_params.copy())
        mse_history.append(info['mse'])

        if done:
            break

    return trajectory, params_history, mse_history


def random_search(surrogate, topology, target_waveform, n_trials=500):
    """Baseline: random search over parameter space."""
    env = CircuitDesignEnv(surrogate, device=DEVICE, topology=topology, use_spice_reward=False)
    env.target_waveform = target_waveform

    best_mse = float('inf')
    best_waveform = None
    best_params = None
    mse_over_trials = []

    for _ in range(n_trials):
        params = env._random_params()
        waveform = env._simulate(params)
        mse = np.mean((waveform - target_waveform) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_waveform = waveform.copy()
            best_params = params.copy()
        mse_over_trials.append(best_mse)

    return best_mse, best_waveform, best_params, mse_over_trials


# ============================================================================
# VISUAL 1: Animated GIF -- agent optimizing step by step
# ============================================================================
def generate_optimization_gif(topology='buck'):
    print(f"\n--- Generating optimization GIF for {topology} ---")
    surrogate = load_trained_model(device=DEVICE)
    agent = load_agent(topology, surrogate)
    if agent is None:
        print(f"  No checkpoint found for {topology}, skipping.")
        return

    target = create_target_waveform(topology, v_in=12.0, duty=0.5)
    trajectory, params_hist, mse_hist = run_agent_with_trajectory(agent, target)

    n_frames = len(trajectory)
    t = np.arange(len(target))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#c9d1d9')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # Left: waveform
    ax1 = axes[0]
    ax1.plot(t, target, 'w-', linewidth=2.5, label='Target Waveform', zorder=5)
    line_pred, = ax1.plot([], [], '#58a6ff', linewidth=2, label='Agent Prediction')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Voltage (V)')
    title1 = ax1.set_title(f'{topology.upper()} -- Step 0 / {n_frames-1}')
    ax1.legend(loc='lower right', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    y_min = min(np.min(target), min(np.min(tr) for tr in trajectory)) - 1
    y_max = max(np.max(target), max(np.max(tr) for tr in trajectory)) + 1
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlim(0, len(target) - 1)

    # Right: MSE over steps
    ax2 = axes[1]
    ax2.set_xlabel('Agent Step')
    ax2.set_ylabel('MSE')
    ax2.set_title('Waveform Error')
    ax2.set_xlim(0, n_frames - 1)
    ax2.set_ylim(0, max(mse_hist) * 1.1 + 0.1)
    line_mse, = ax2.plot([], [], '#f0883e', linewidth=2)
    mse_text = ax2.text(
        0.95, 0.95, '', transform=ax2.transAxes,
        ha='right', va='top', fontsize=13, color='#58a6ff',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', edgecolor='#30363d')
    )

    fig.tight_layout(pad=2)

    def animate(frame):
        line_pred.set_data(t, trajectory[frame])
        line_mse.set_data(range(frame + 1), mse_hist[:frame + 1])
        title1.set_text(f'{topology.upper()} -- Step {frame} / {n_frames-1}')
        mse_text.set_text(f'MSE: {mse_hist[frame]:.2f}')

        # Color the prediction line based on quality
        if mse_hist[frame] < mse_hist[0] * 0.1:
            line_pred.set_color('#3fb950')  # green = good
        elif mse_hist[frame] < mse_hist[0] * 0.5:
            line_pred.set_color('#58a6ff')  # blue = improving
        else:
            line_pred.set_color('#f85149')  # red = far off

        return line_pred, line_mse, title1, mse_text

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=200, blit=False)
    gif_path = f'demo_optimization_{topology}.gif'
    anim.save(gif_path, writer=PillowWriter(fps=5))
    plt.close(fig)
    print(f"  Saved: {gif_path} ({n_frames} frames)")
    return gif_path


# ============================================================================
# VISUAL 2: Before/After static comparison
# ============================================================================
def generate_before_after(topologies=None):
    print("\n--- Generating before/after comparison ---")
    surrogate = load_trained_model(device=DEVICE)

    if topologies is None:
        topologies = ['buck', 'boost', 'flyback']

    available = []
    for topo in topologies:
        if os.path.exists(f'checkpoints/rl_agent_{topo}.pt'):
            available.append(topo)
    if not available:
        print("  No trained agents found, skipping.")
        return

    n = len(available)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    fig.patch.set_facecolor('#0d1117')
    if n == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('RL Agent: Random Start vs Optimized Result', fontsize=16,
                 color='#c9d1d9', fontweight='bold', y=0.98)

    for row, topology in enumerate(available):
        agent = load_agent(topology, surrogate)
        target = create_target_waveform(topology, v_in=12.0, duty=0.5)
        trajectory, params_hist, mse_hist = run_agent_with_trajectory(agent, target)

        t = np.arange(len(target))

        for col, ax in enumerate(axes[row]):
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#c9d1d9')
            ax.xaxis.label.set_color('#c9d1d9')
            ax.yaxis.label.set_color('#c9d1d9')
            ax.title.set_color('#c9d1d9')
            for spine in ax.spines.values():
                spine.set_color('#30363d')

        # Before (step 0)
        ax_before = axes[row, 0]
        ax_before.plot(t, target, 'w-', linewidth=2, label='Target')
        ax_before.plot(t, trajectory[0], '#f85149', linewidth=2, label='Random Init', linestyle='--')
        ax_before.set_title(f'{topology.upper()} -- Random Start (MSE: {mse_hist[0]:.1f})')
        ax_before.set_ylabel('Voltage (V)')
        ax_before.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)
        ax_before.fill_between(t, target, trajectory[0], alpha=0.15, color='#f85149')

        # After (final step)
        ax_after = axes[row, 1]
        ax_after.plot(t, target, 'w-', linewidth=2, label='Target')
        ax_after.plot(t, trajectory[-1], '#3fb950', linewidth=2, label='RL Optimized', linestyle='--')
        ax_after.set_title(f'{topology.upper()} -- After {len(trajectory)-1} Steps (MSE: {mse_hist[-1]:.1f})')
        ax_after.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)
        ax_after.fill_between(t, target, trajectory[-1], alpha=0.15, color='#3fb950')

        # Match y-axis limits
        y_min = min(np.min(target), np.min(trajectory[0]), np.min(trajectory[-1])) - 1
        y_max = max(np.max(target), np.max(trajectory[0]), np.max(trajectory[-1])) + 1
        ax_before.set_ylim(y_min, y_max)
        ax_after.set_ylim(y_min, y_max)

    for ax in axes[-1]:
        ax.set_xlabel('Time Step')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = 'demo_before_after.png'
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================================
# VISUAL 3: RL Agent vs Random Search baseline comparison
# ============================================================================
def generate_baseline_comparison(topologies=None):
    print("\n--- Generating baseline comparison (RL vs Random Search) ---")
    surrogate = load_trained_model(device=DEVICE)

    if topologies is None:
        topologies = ['buck', 'boost', 'buck_boost', 'sepic', 'flyback']

    available = []
    for topo in topologies:
        if os.path.exists(f'checkpoints/rl_agent_{topo}.pt'):
            available.append(topo)
    if not available:
        print("  No trained agents found, skipping.")
        return

    rl_mses = []
    random_mses = []
    labels = []

    for topology in available:
        print(f"  Testing {topology}...")
        agent = load_agent(topology, surrogate)
        target = create_target_waveform(topology, v_in=12.0, duty=0.5)

        # RL agent: run 10 episodes, take mean
        rl_episode_mses = []
        for _ in range(10):
            _, _, mse_hist = run_agent_with_trajectory(agent, target)
            rl_episode_mses.append(mse_hist[-1])
        rl_mse = np.mean(rl_episode_mses)

        # Random search: 500 trials (comparable to 50 steps x 10 episodes)
        rand_mse, _, _, _ = random_search(surrogate, topology, target, n_trials=500)

        rl_mses.append(rl_mse)
        random_mses.append(rand_mse)
        labels.append(topology.replace('_', '\n'))
        print(f"    RL: {rl_mse:.1f}  |  Random: {rand_mse:.1f}")

    # Bar chart
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#c9d1d9')
    ax.xaxis.label.set_color('#c9d1d9')
    ax.yaxis.label.set_color('#c9d1d9')
    ax.title.set_color('#c9d1d9')
    for spine in ax.spines.values():
        spine.set_color('#30363d')

    bars1 = ax.bar(x - width/2, random_mses, width, label='Random Search (500 random samples)',
                   color='#f85149', alpha=0.85, edgecolor='#30363d')
    bars2 = ax.bar(x + width/2, rl_mses, width, label='RL Agent (50-step optimization, mean of 10 runs)',
                   color='#3fb950', alpha=0.85, edgecolor='#30363d')

    # Value labels on bars
    for bar, val in zip(bars1, random_mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.0f}', ha='center', va='bottom', color='#f85149', fontweight='bold', fontsize=10)
    for bar, val in zip(bars2, rl_mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.0f}', ha='center', va='bottom', color='#3fb950', fontweight='bold', fontsize=10)

    ax.set_xlabel('Topology', fontsize=13)
    ax.set_ylabel('Best MSE (lower is better)', fontsize=13)
    ax.set_title('RL Agent vs Random Search Baseline', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=11)
    ax.set_yscale('log')

    fig.tight_layout()
    path = 'demo_baseline_comparison.png'
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Generating demo visuals for LinkedIn / README")
    print("=" * 60)

    # 1. Animated GIF for buck (simplest, best trained)
    gif = generate_optimization_gif('buck')

    # 2. Before/After for 3 topologies
    ba = generate_before_after(['buck', 'boost', 'flyback'])

    # 3. Baseline comparison across all available topologies
    bl = generate_baseline_comparison()

    print("\n" + "=" * 60)
    print("Done! Generated files:")
    for f in [gif, ba, bl]:
        if f:
            size_kb = os.path.getsize(f) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")
    print("=" * 60)
