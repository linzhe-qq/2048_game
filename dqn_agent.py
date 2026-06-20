import numpy as np
import torch
import torch.nn.functional as F

from dqn_network import DQNNetwork
from replay_buffer import ReplayBuffer
from config import Config


class DQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.device = self._resolve_device(config.device)

        self.q_net = DQNNetwork(
            config.state_size, config.action_size,
            config.hidden_size, config.num_hidden_layers,
        ).to(self.device)

        self.target_net = DQNNetwork(
            config.state_size, config.action_size,
            config.hidden_size, config.num_hidden_layers,
        ).to(self.device)
        self.target_net.sync_target(self.q_net)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        self.buffer = ReplayBuffer(config.buffer_size, config.state_size)
        self.train_step_count = 0

    # ── device ──────────────────────────────────────────────
    @staticmethod
    def _resolve_device(spec: str) -> torch.device:
        if spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(spec)

    # ── action selection ────────────────────────────────────
    def select_action(self, state: np.ndarray, epsilon: float,
                      valid_actions: list[int] | None = None) -> int:
        if np.random.random() < epsilon:
            pool = valid_actions if valid_actions else list(range(self.config.action_size))
            return int(np.random.choice(pool))

        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self.q_net(t).cpu().numpy()[0]
            if valid_actions:
                masked = np.full_like(q_vals, -np.inf)
                for a in valid_actions:
                    masked[a] = q_vals[a]
                return int(np.argmax(masked))
            return int(np.argmax(q_vals))

    # ── experience storage ──────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ── single training step ────────────────────────────────
    def train_step(self) -> float | None:
        cfg = self.config
        if len(self.buffer) < cfg.min_buffer_before_train:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(cfg.batch_size)

        states_t   = torch.FloatTensor(states).to(self.device)
        actions_t  = torch.LongTensor(actions).to(self.device)
        rewards_t  = torch.FloatTensor(rewards).to(self.device)
        next_st_t  = torch.FloatTensor(next_states).to(self.device)
        dones_t    = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_st_t).max(1)[0]
            targets = rewards_t + cfg.gamma * next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), cfg.grad_clip)
        self.optimizer.step()

        self.train_step_count += 1
        return loss.item()

    # ── target network sync ─────────────────────────────────
    def sync_target(self):
        self.target_net.sync_target(self.q_net)

    # ── epsilon schedule ────────────────────────────────────
    def get_epsilon(self, total_steps: int) -> float:
        cfg = self.config
        if total_steps >= cfg.epsilon_decay_steps:
            return cfg.epsilon_end
        ratio = total_steps / cfg.epsilon_decay_steps
        return cfg.epsilon_start * ((cfg.epsilon_end / cfg.epsilon_start) ** ratio)

    # ── learning-rate helpers ───────────────────────────────
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = max(lr, self.config.lr_decay_min)

    def halve_lr(self):
        self.set_lr(self.get_lr() * self.config.lr_decay_factor)

    # ── checkpoint I/O ──────────────────────────────────────
    def save_networks(self, q_path: str, target_path: str):
        torch.save(self.q_net.state_dict(), q_path)
        torch.save(self.target_net.state_dict(), target_path)

    def load_networks(self, q_path: str, target_path: str):
        self.q_net.load_state_dict(torch.load(q_path, map_location=self.device, weights_only=True))
        self.target_net.load_state_dict(torch.load(target_path, map_location=self.device, weights_only=True))
