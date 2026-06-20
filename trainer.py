import os
import csv
import json
import logging
import threading
import traceback
import signal
from collections import deque
from datetime import datetime

from config import Config
from dqn_agent import DQNAgent
from engine_wrapper import Game2048Env


class Trainer:
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"

    def __init__(self, config: Config):
        self.config = config
        self.agent = DQNAgent(config)
        self.env = Game2048Env()

        self.state = self.IDLE
        self.current_episode = 0
        self.total_steps = 0
        self.episode_steps = 0
        self.last_loss: float | None = None
        self.last_action: int | None = None
        self.best_score = 0
        self.current_grid = self.env.get_grid()
        self.current_score = 0

        self.episode_scores: deque[int] = deque(maxlen=500)

        # Threading: pause_event starts cleared so train() blocks until start()
        self.pause_event = threading.Event()  # cleared = paused
        self.start_event = threading.Event()   # set when user presses S first time
        self.stop_requested = False
        self.reset_requested = False
        self.lock = threading.Lock()

        self.logger = self._setup_logging()
        self._metrics_file = None
        self._metrics_writer = None
        self._lr_decay_history: list[float] = []

    # ── logging setup ───────────────────────────────────────
    def _setup_logging(self) -> logging.Logger:
        os.makedirs(self.config.log_dir, exist_ok=True)
        logger = logging.getLogger("dqn2048")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        fh = logging.FileHandler(
            os.path.join(self.config.log_dir, "training.log"), encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [EP:%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)
        return logger

    def _open_metrics_csv(self):
        path = os.path.join(self.config.log_dir, "metrics.csv")
        new_file = not os.path.exists(path)
        self._metrics_file = open(path, "a", newline="", encoding="utf-8")
        self._metrics_writer = csv.writer(self._metrics_file)
        if new_file:
            self._metrics_writer.writerow([
                "episode", "steps", "score",
                "avg10", "avg100", "epsilon", "loss",
            ])

    # ── thread-safe snapshot for UI ─────────────────────────
    def snapshot(self) -> dict:
        with self.lock:
            return {
                "state": self.state,
                "episode": self.current_episode,
                "episode_steps": self.episode_steps,
                "total_steps": self.total_steps,
                "score": self.current_score,
                "best_score": self.best_score,
                "last_loss": self.last_loss,
                "last_action": self.last_action,
                "grid": [row[:] for row in self.current_grid],
                "epsilon": self.agent.get_epsilon(self.total_steps),
                "lr": self.agent.get_lr(),
                "buffer_usage": self.agent.buffer.usage,
                "avg10": self.avg_score(10),
                "avg100": self.avg_score(100),
                "steps_to_target_update": self.steps_to_next_target_update(),
                "epsilon_progress": self.epsilon_progress(),
                "recent_scores": list(self.episode_scores)[-self.config.chart_window:],
            }

    # ── statistics helpers ───────────────────────────────────
    def avg_score(self, n: int) -> float:
        if not self.episode_scores:
            return 0.0
        return sum(list(self.episode_scores)[-n:]) / min(len(self.episode_scores), n)

    # ── checkpoint ──────────────────────────────────────────
    def save_checkpoint(self, tag: str = "latest"):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        q_path = os.path.join(self.config.checkpoint_dir, f"q_net_{tag}.pth")
        t_path = os.path.join(self.config.checkpoint_dir, f"target_net_{tag}.pth")
        self.agent.save_networks(q_path, t_path)

        meta = {
            "episode": self.current_episode,
            "total_steps": self.total_steps,
            "best_score": self.best_score,
            "epsilon": self.agent.get_epsilon(self.total_steps),
            "learning_rate": self.agent.get_lr(),
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(self.config.checkpoint_dir, f"metadata_{tag}.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load_checkpoint(self, tag: str = "latest") -> bool:
        q_path = os.path.join(self.config.checkpoint_dir, f"q_net_{tag}.pth")
        t_path = os.path.join(self.config.checkpoint_dir, f"target_net_{tag}.pth")
        m_path = os.path.join(self.config.checkpoint_dir, f"metadata_{tag}.json")

        if not all(os.path.exists(p) for p in (q_path, t_path, m_path)):
            return False

        self.agent.load_networks(q_path, t_path)
        with open(m_path) as f:
            meta = json.load(f)

        self.current_episode = meta["episode"]
        self.total_steps = meta["total_steps"]
        self.best_score = meta.get("best_score", 0)
        lr = meta.get("learning_rate", self.config.learning_rate)
        self.agent.set_lr(lr)
        return True

    # ── learning-rate plateau decay ─────────────────────────
    def _maybe_decay_lr(self):
        cfg = self.config
        if not cfg.lr_decay_enabled:
            return
        if len(self.episode_scores) < cfg.lr_decay_window:
            return
        recent = list(self.episode_scores)[-cfg.lr_decay_window:]
        avg = sum(recent) / len(recent)
        if self._lr_decay_history and avg <= self._lr_decay_history[-1]:
            self.agent.halve_lr()
            self.logger.info(
                f"LR plateau -> halved to {self.agent.get_lr():.2e}"
            )
        self._lr_decay_history.append(avg)

    # ── the training loop (blocking) ────────────────────────
    def train(self):
        self._open_metrics_csv()

        # Install SIGINT handler for emergency save
        def _sigint_handler(sig, frame):
            self.save_checkpoint("emergency")
            self.logger.info("SIGINT received - emergency checkpoint saved")
            self.stop_requested = True
            self.pause_event.set()
            self.start_event.set()

        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _sigint_handler)

        try:
            # Block until user presses S for the first time
            self.start_event.wait()
            if self.stop_requested:
                return

            self.state = self.RUNNING

            while not self.stop_requested:
                self._run_episode()
                # Wait if paused (pause_event is cleared when paused)
                self.pause_event.wait()
        except Exception as exc:
            tb = traceback.format_exc()
            err_path = os.path.join(self.config.log_dir, "error.log")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()}\n{tb}\n")
            self.save_checkpoint("emergency")
            self.logger.error(f"CRASH: {exc}")
            raise
        finally:
            self.state = self.STOPPED
            if self._metrics_file:
                self._metrics_file.close()
            signal.signal(signal.SIGINT, old_handler)

    def _run_episode(self):
        if self.reset_requested:
            self.reset_requested = False
            with self.lock:
                self.current_episode = 0
                self.total_steps = 0
                self.episode_steps = 0
                self.last_loss = None
                self.last_action = None
                self.best_score = 0
                self.current_score = 0
                self.episode_scores.clear()
            self._lr_decay_history.clear()
            self.agent.buffer.clear()
            self.agent.set_lr(self.config.learning_rate)
            self.logger.info("=== FULL RESET ===")

        state = self.env.reset()
        self.episode_steps = 0
        with self.lock:
            self.current_grid = self.env.get_grid()
            self.current_score = 0
        done = False

        while not done and not self.stop_requested:
            self.pause_event.wait()
            if self.stop_requested:
                break

            epsilon = self.agent.get_epsilon(self.total_steps)
            valid = self.env.get_valid_actions()
            action = self.agent.select_action(state, epsilon, valid)

            next_state, reward, done, _info = self.env.step(action)
            self.agent.store(state, action, reward, next_state, done)

            self.total_steps += 1
            self.episode_steps += 1

            with self.lock:
                self.last_action = action
                self.current_grid = self.env.get_grid()
                self.current_score = self.env.get_score()

            if self.total_steps % self.config.train_freq == 0:
                loss = self.agent.train_step()
                if loss is not None:
                    with self.lock:
                        self.last_loss = loss

            if self.total_steps % self.config.target_update_freq == 0:
                self.agent.sync_target()
                self.logger.info("Target network synced")

            state = next_state

        self.current_episode += 1
        self.episode_scores.append(self.env.get_score())
        with self.lock:
            if self.env.get_score() > self.best_score:
                self.best_score = self.env.get_score()
            self.current_score = self.env.get_score()

        self.logger.info(
            f"{self.current_episode}] steps={self.episode_steps} "
            f"score={self.env.get_score()} eps={self.agent.get_epsilon(self.total_steps):.4f}"
        )

        if self.current_episode % 10 == 0 and self._metrics_writer:
            self._metrics_writer.writerow([
                self.current_episode,
                self.episode_steps,
                self.env.get_score(),
                f"{self.avg_score(10):.1f}",
                f"{self.avg_score(100):.1f}",
                f"{self.agent.get_epsilon(self.total_steps):.6f}",
                f"{self.last_loss or 0:.6f}",
            ])
            self._metrics_file.flush()

        if self.current_episode % self.config.checkpoint_interval == 0:
            self.save_checkpoint()

        if self.current_episode % self.config.lr_decay_window == 0:
            self._maybe_decay_lr()

    # ── external controls ───────────────────────────────────
    def start(self):
        if self.state == self.IDLE:
            self.start_event.set()
            self.pause_event.set()
        elif self.state == self.PAUSED:
            self.state = self.RUNNING
            self.pause_event.set()

    def pause(self):
        if self.state == self.RUNNING:
            self.state = self.PAUSED
            self.pause_event.clear()

    def reset_all(self):
        self.reset_requested = True
        self.pause_event.set()
        self.start_event.set()

    def safe_quit(self):
        self.stop_requested = True
        self.pause_event.set()
        self.start_event.set()

    def steps_to_next_target_update(self) -> int:
        return self.config.target_update_freq - (
            self.total_steps % self.config.target_update_freq
        )

    def epsilon_progress(self) -> float:
        cfg = self.config
        eps = self.agent.get_epsilon(self.total_steps)
        if cfg.epsilon_start == cfg.epsilon_end:
            return 1.0
        return (cfg.epsilon_start - eps) / (cfg.epsilon_start - cfg.epsilon_end)
