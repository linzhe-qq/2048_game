from dataclasses import dataclass


@dataclass
class Config:
    # Network architecture
    state_size: int = 16
    action_size: int = 4
    hidden_size: int = 256
    num_hidden_layers: int = 3

    # Optimization
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 100_000

    # Epsilon schedule (decay over total steps, not episodes)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100_000

    # Target network
    target_update_freq: int = 1_000

    # Training cadence
    train_freq: int = 4
    min_buffer_before_train: int = 1_000

    # Loss & gradient
    loss_fn: str = "huber"
    grad_clip: float = 10.0

    # Checkpoint & logging
    checkpoint_interval: int = 50
    log_interval: int = 1
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Learning-rate decay on plateau
    lr_decay_enabled: bool = True
    lr_decay_window: int = 50
    lr_decay_factor: float = 0.5
    lr_decay_min: float = 1e-7

    # Device: "auto" | "cuda" | "cpu"
    device: str = "auto"

    # Number of recent episodes for the mini chart
    chart_window: int = 100
