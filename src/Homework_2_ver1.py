import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import os

# This script trains a DQN agent for Hw2Env using either:
# 1) pixel observations with a CNN Q-network, or
# 2) compact high-level state with an MLP Q-network.
#
# The training loop uses:
# - experience replay,
# - a periodically updated target network,
# - epsilon-greedy exploration,
# - Huber loss + gradient clipping for stability.

# ──────────────────────────────────────────────
# Set env vars for headless rendering if needed
# ──────────────────────────────────────────────
# os.environ["MUJOCO_GL"] = "egl"
# os.environ["PYOPENGL_PLATFORM"] = "egl"

from homework2 import Hw2Env

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
N_ACTIONS          = 8
GAMMA              = 0.99
EPSILON            = 1.0
EPSILON_DECAY      = 0.999   # was 0.999 — reaches MIN_EPSILON ~ep 250
EPSILON_DECAY_ITER = 10       # was 10 — decay more frequently
MIN_EPSILON        = 0.1
LEARNING_RATE      = 1e-4
BATCH_SIZE         = 32
UPDATE_FREQ        = 4         # env steps between network updates
TARGET_UPDATE_FREQ = 100       # network updates between target-net syncs
BUFFER_LENGTH      = 10_000
N_EPISODES         = 3000       # total training episodes
USE_PIXELS         = False      # set False to use high_level_state

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "version_1")
# Keep outputs local to this script's folder so checkpoints/plots are easy to find.
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────
class ReplayBuffer:
    """Fixed-size experience replay buffer.

    Stores transitions as (s, a, r, s', done). Sampling uniformly from this
    buffer reduces temporal correlation between updates and improves learning
    stability compared to training on consecutive steps directly.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Old experiences are discarded automatically when maxlen is reached.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        # Uniform random mini-batch sampling.
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32, device=DEVICE),
            torch.tensor(actions,               dtype=torch.long,    device=DEVICE),
            torch.tensor(rewards,               dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=DEVICE),
            torch.tensor(dones,                 dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────
# Networks
# ──────────────────────────────────────────────
class DQNPixels(nn.Module):
    """CNN-based DQN for raw 128×128 RGB images."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32,  4, 2, 1), nn.ReLU(),   # → (32, 64, 64)
            nn.Conv2d(32, 64,  4, 2, 1), nn.ReLU(),   # → (64, 32, 32)
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),   # → (128,16, 16)
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),  # → (256, 8,  8)
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(),  # → (512, 4,  4)
        )
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        # x: (B, 3, H, W) float32 in [0,1] — state() already normalises
        feat = self.conv(x)                   # (B, 512, 4, 4)
        feat = feat.mean(dim=[2, 3])          # global avg pool → (B, 512)
        return self.head(feat)


class DQNHighLevel(nn.Module):
    """MLP-based DQN for the high-level state vector."""

    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────
class DQNAgent:
    """Minimal DQN agent with target network and replay buffer."""

    def __init__(self, n_actions: int, state_dim=None):
        if USE_PIXELS:
            self.online_net = DQNPixels(n_actions).to(DEVICE)
            self.target_net = DQNPixels(n_actions).to(DEVICE)
        else:
            assert state_dim is not None
            self.online_net = DQNHighLevel(state_dim, n_actions).to(DEVICE)
            self.target_net = DQNHighLevel(state_dim, n_actions).to(DEVICE)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.buffer     = ReplayBuffer(BUFFER_LENGTH)
        self.epsilon    = EPSILON
        self.n_actions  = n_actions
        # Counts gradient updates (distinct from env steps/episodes).
        self.update_count = 0

    # ── action selection ──
    def select_action(self, state) -> int:
        # Explore with probability epsilon, otherwise exploit current Q-values.
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = self.online_net(s)
        return int(q.argmax(dim=1).item())

    # ── one gradient update ──
    def update(self):
        # Wait until enough data is available for one full mini-batch.
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Current Q-values
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman targets from target network (no gradient through target path).
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + GAMMA * next_q * (1.0 - dones)

        # Huber loss is more robust than plain MSE on noisy TD targets.
        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid rare exploding updates destabilizing training.
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1

        # Decay epsilon
        if self.update_count % EPSILON_DECAY_ITER == 0:
            self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

        # Sync target network
        if self.update_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str):
        # Save full training state for resuming later.
        torch.save({
            "online": self.online_net.state_dict(),
            "target": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "update_count": self.update_count,
        }, path)
        print(f"  ✓ checkpoint saved → {path}")

    def load(self, path: str):
        # Load full training state from a previous checkpoint.
        ckpt = torch.load(path, map_location=DEVICE)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.epsilon    = ckpt["epsilon"]
        self.update_count = ckpt["update_count"]
        print(f"  ✓ checkpoint loaded ← {path}")


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train():
    """Run DQN training and save periodic checkpoints/plots."""

    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")

    # Probe state dimension for MLP mode
    env.reset()
    if USE_PIXELS:
        state_dim = None
        # Pixel mode: use RGB observation tensor.
        def get_state(e):   return e.state().numpy()    # (3, H, W) float32 in [0,1]
    else:
        probe = env.high_level_state()
        state_dim = int(np.prod(probe.shape))
        # High-level mode:
        # Normalize each coordinate so different dimensions have similar scale.
        # x in [0.25,0.75] -> [-1,1], y in [-0.3,0.3] -> [-1,1].
        STATE_MEAN = np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0], dtype=np.float32)
        STATE_STD  = np.array([0.25, 0.3, 0.25, 0.3, 0.25, 0.3], dtype=np.float32)
        def get_state(e):
            s = e.high_level_state().flatten().astype(np.float32)
            return (s - STATE_MEAN) / STATE_STD

    agent = DQNAgent(N_ACTIONS, state_dim=state_dim)

    rewards_history  = []
    rps_history      = []
    env_step         = 0

    for episode in range(N_EPISODES):
        env.reset()
        state        = get_state(env)
        done         = False
        cum_reward   = 0.0
        ep_steps     = 0

        while not done:
            action = agent.select_action(state)
            # next_obs is unused because state is read via get_state(env)
            # for a single consistent state path in both modes.
            next_obs, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated

            if USE_PIXELS:
                next_state = get_state(env)  # (3,H,W) float32, consistent with state
            else:
                next_state = get_state(env)  # normalized high_level_state

            agent.buffer.push(state, action, reward, next_state, float(done))
            state      = next_state
            cum_reward += reward
            ep_steps   += 1
            env_step   += 1

            # Update every UPDATE_FREQ env steps
            if env_step % UPDATE_FREQ == 0:
                agent.update()

        rps = cum_reward / max(ep_steps, 1)
        rewards_history.append(cum_reward)
        rps_history.append(rps)

        print(f"Episode {episode+1:4d}/{N_EPISODES} | "
              f"reward={cum_reward:8.3f} | RPS={rps:.4f} | "
              f"ε={agent.epsilon:.4f} | updates={agent.update_count}")

        # Save checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            agent.save(os.path.join(OUTPUT_DIR, f"dqn_ep{episode+1}.pt"))
            plot_curves(rewards_history, rps_history, episode + 1)

    agent.save(os.path.join(OUTPUT_DIR, "dqn_final.pt"))
    plot_curves(rewards_history, rps_history, N_EPISODES, final=True)
    return agent, rewards_history, rps_history


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def moving_avg(data, window=20):
    """Simple moving average used for smoother trend curves."""

    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_curves(rewards, rps_list, episode, final=False):
    """Plot raw and smoothed reward metrics and save to disk."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"DQN Training — Episode {episode}", fontsize=14, fontweight="bold")

    for ax, data, label, color in zip(
        axes,
        [rewards, rps_list],
        ["Cumulative Reward", "Reward per Step (RPS)"],
        ["steelblue", "darkorange"],
    ):
        eps = range(1, len(data) + 1)
        ax.plot(eps, data, alpha=0.3, color=color, linewidth=0.8, label="raw")
        if len(data) >= 20:
            ma = moving_avg(data)
            ax.plot(range(20, len(data) + 1), ma,
                    color=color, linewidth=2, label="20-ep avg")
        ax.set_xlabel("Episode")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fname = "dqn_curves_final.png" if final else f"dqn_curves_ep{episode}.png"
    fname = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  ✓ plot saved → {fname}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    train()