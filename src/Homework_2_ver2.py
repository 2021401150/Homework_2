"""
Deep Q-Network (DQN) for Robot Object Pushing Task
Supports both pixel-based (CNN) and high-level state inputs.

Core algorithmic pieces in this implementation:
- epsilon-greedy exploration,
- replay buffer for decorrelated training batches,
- target network to stabilize Bellman targets,
- Double-DQN style target action selection,
- periodic plotting/checkpoint-style output for monitoring.
"""

import random
import collections
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from homework2 import Hw2Env

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
N_ACTIONS            = 8
GAMMA                = 0.99
EPSILON              = 1.0
EPSILON_DECAY        = 0.999
EPSILON_DECAY_ITER   = 10        # decay every N gradient updates
MIN_EPSILON          = 0.1
LEARNING_RATE        = 1e-4
BATCH_SIZE           = 32
UPDATE_FREQ          = 4         # env steps between gradient updates
TARGET_UPDATE_FREQ   = 100       # gradient updates between target-net syncs
BUFFER_LENGTH        = 10_000
N_EPISODES           = 3000
USE_HIGH_LEVEL_STATE = True     # set True for faster experimentation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
PLOT_OUTPUT_DIR = "version_2"
# Store all curves for this variant in a dedicated directory.
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# State Normalization (high-level state only)
# x ∈ [0.25, 0.75] → [-1, 1],  y ∈ [-0.3, 0.3] → [-1, 1]
# ─────────────────────────────────────────────
_X_MEAN, _X_RANGE = 0.5, 0.25
_Y_MEAN, _Y_RANGE = 0.0, 0.3

def normalize_state(state: np.ndarray) -> np.ndarray:
    """Normalize high-level state [ee_x, ee_y, obj_x, obj_y, goal_x, goal_y] to [-1, 1]."""
    # Copy avoids mutating caller-owned arrays in replay buffer or env returns.
    state = state.copy()
    state[0::2] = (state[0::2] - _X_MEAN) / _X_RANGE   # x dims
    state[1::2] = (state[1::2] - _Y_MEAN) / _Y_RANGE   # y dims
    return state


# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────
Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """Fixed-capacity replay memory of Transition tuples."""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        # Appends one (s, a, r, s', done) transition.
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        # Uniform random sample gives i.i.d.-like mini-batches for SGD.
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────
class CNNQNetwork(nn.Module):
    """Pixel-based Q-network (128x128 RGB input)."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32,  4, 2, 1), nn.ReLU(),   # -> (32, 64, 64)
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),   # -> (64, 32, 32)
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),  # -> (128,16, 16)
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(), # -> (256, 8,  8)
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(), # -> (512, 4,  4)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)           # -> (512, 1,  1)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        # x: (B, 3, 128, 128), float in [0, 1]
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class MLPQNetwork(nn.Module):
    """High-level-state Q-network (6-dim input: ee, obj, goal positions)."""

    def __init__(self, n_actions: int, state_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────
class DQNAgent:
    """DQN agent wrapper around policy/target networks and optimization logic."""

    def __init__(self, n_actions: int, use_high_level: bool = False):
        self.n_actions = n_actions
        self.use_high_level = use_high_level
        self.epsilon = EPSILON
        # Counts gradient updates (not environment steps).
        self._update_count = 0

        if use_high_level:
            self.policy_net = MLPQNetwork(n_actions).to(DEVICE)
            self.target_net = MLPQNetwork(n_actions).to(DEVICE)
        else:
            self.policy_net = CNNQNetwork(n_actions).to(DEVICE)
            self.target_net = CNNQNetwork(n_actions).to(DEVICE)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Keep target net in eval mode since it's only used for forward passes.
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.buffer = ReplayBuffer(BUFFER_LENGTH)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    # ── helpers ──────────────────────────────
    def _to_tensor(self, state):
        """Convert single state to model input tensor on the selected device."""
        if self.use_high_level:
            return torch.tensor(normalize_state(state), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        else:
            # state is already a (3, H, W) float tensor in [0,1]
            return state.unsqueeze(0).to(DEVICE)

    def _batch_to_tensor(self, states):
        """Convert list/tuple of states into a mini-batch tensor."""
        if self.use_high_level:
            return torch.tensor(np.array([normalize_state(s) for s in states]), dtype=torch.float32).to(DEVICE)
        else:
            return torch.stack(states).to(DEVICE)

    # ── action selection ─────────────────────
    @torch.no_grad()
    def select_action(self, state):
        # Epsilon-greedy: random action for exploration, argmax for exploitation.
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        t = self._to_tensor(state)
        q_vals = self.policy_net(t)
        return q_vals.argmax(dim=1).item()

    # ── learning step ────────────────────────
    def update(self):
        # Skip update until enough transitions are accumulated.
        if len(self.buffer) < BATCH_SIZE:
            return None

        batch = self.buffer.sample(BATCH_SIZE)

        states      = self._batch_to_tensor(batch.state)
        actions     = torch.tensor(batch.action,  dtype=torch.long).unsqueeze(1).to(DEVICE)
        rewards     = torch.tensor(batch.reward,  dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_states = self._batch_to_tensor(batch.next_state)
        dones       = torch.tensor(batch.done,    dtype=torch.float32).unsqueeze(1).to(DEVICE)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions)

        # Double-DQN target:
        # - action selection from policy_net
        # - action evaluation from target_net
        # This reduces overestimation vs max over a single network.
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions)
            target_q = rewards + GAMMA * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping helps avoid instability from rare large TD errors.
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self._update_count += 1

        # Decay epsilon
        if self._update_count % EPSILON_DECAY_ITER == 0:
            self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

        # Sync target network
        if self._update_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
def train():
    """Train DQN and save periodic diagnostic plots."""

    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
    agent = DQNAgent(N_ACTIONS, use_high_level=USE_HIGH_LEVEL_STATE)

    episode_rewards = []
    episode_rps     = []
    episode_losses  = []

    global_step = 0

    for episode in range(N_EPISODES):
        env.reset()

        # Choose observation type once per step based on configured mode.
        state = env.high_level_state() if USE_HIGH_LEVEL_STATE else env.state()
        done = False
        ep_reward = 0.0
        ep_steps  = 0
        ep_losses = []

        while not done:
            action = agent.select_action(state)
            _, reward, terminal, truncated = env.step(action)
            next_state = env.high_level_state() if USE_HIGH_LEVEL_STATE else env.state()
            done = terminal or truncated

            agent.buffer.push(state, action, reward, next_state, float(done))
            state = next_state

            ep_reward += reward
            ep_steps  += 1
            global_step += 1

            # Gradient update
            if global_step % UPDATE_FREQ == 0:
                loss = agent.update()
                if loss is not None:
                    ep_losses.append(loss)

        episode_rewards.append(ep_reward)
        episode_rps.append(ep_reward / max(ep_steps, 1))
        episode_losses.append(np.mean(ep_losses) if ep_losses else 0.0)

        print(
            f"Episode {episode+1:4d}/{N_EPISODES} | "
            f"Reward={ep_reward:.3f} | RPS={ep_reward/max(ep_steps,1):.4f} | "
            f"Steps={ep_steps} | Epsilon={agent.epsilon:.3f} | "
            f"BufferSize={len(agent.buffer)}"
        )

        if (episode + 1) % 50 == 0:
            # Save intermediate curve snapshots to monitor training progress.
            plot_results(episode_rewards, episode_rps, episode_losses,
                         save_path=os.path.join(PLOT_OUTPUT_DIR, f"dqn_training_curves_ep{episode+1}.png"))

    # Save trained policy parameters (not full optimizer/replay state).
    torch.save(agent.policy_net.state_dict(), "dqn_policy.pt")
    plot_results(
        episode_rewards,
        episode_rps,
        episode_losses,
        save_path=os.path.join(PLOT_OUTPUT_DIR, "dqn_training_curves.png"),
    )
    return agent, episode_rewards, episode_rps


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def smooth(arr, window=20):
    """Simple moving average for plotting."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_results(rewards, rps, losses=None, save_path="dqn_training_curves.png"):
    """Create reward/RPS/loss curves and persist as a PNG figure."""

    n_plots = 3 if losses is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    # — Reward —
    ax = axes[0]
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode reward")
    ax.plot(
        range(len(smooth(rewards))),
        smooth(rewards),
        color="steelblue", linewidth=2, label="Smoothed (20 ep)"
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward over Episodes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # — RPS —
    ax = axes[1]
    ax.plot(rps, alpha=0.3, color="darkorange", label="RPS")
    ax.plot(
        range(len(smooth(rps))),
        smooth(rps),
        color="darkorange", linewidth=2, label="Smoothed (20 ep)"
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward per Step")
    ax.set_title("RPS over Episodes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # — Loss (optional) —
    if losses is not None and n_plots == 3:
        ax = axes[2]
        ax.plot(losses, alpha=0.3, color="crimson", label="Loss")
        ax.plot(
            range(len(smooth(losses))),
            smooth(losses),
            color="crimson", linewidth=2, label="Smoothed (20 ep)"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Huber Loss")
        ax.set_title("Loss over Episodes")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training curves → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────
def evaluate(model_path: str = "dqn_policy.pt", n_episodes: int = 10):
    """Run greedy policy rollouts in GUI mode for quick qualitative evaluation."""

    env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")
    agent = DQNAgent(N_ACTIONS, use_high_level=USE_HIGH_LEVEL_STATE)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.epsilon = 0.0  # greedy

    for ep in range(n_episodes):
        env.reset()
        state = env.high_level_state() if USE_HIGH_LEVEL_STATE else env.state()
        done = False
        total_r = 0.0
        steps = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminal, truncated = env.step(action)
            state = next_state
            total_r += reward
            steps += 1
            done = terminal or truncated
        print(f"Eval ep {ep+1}: reward={total_r:.3f}, RPS={total_r/steps:.4f}, steps={steps}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    train()