# 2048 DQN Trainer

A terminal-based DQN reinforcement learning trainer for the 2048 game. Features a rich dashboard, automatic checkpoint recovery, and non-blocking keyboard controls.

## Features

- **DQN Agent**: 3-layer fully connected network (256 hidden units) with target network, epsilon-greedy exploration, and replay buffer
- **Terminal Dashboard**: Real-time training metrics, live game board visualization, and ASCII score trend chart
- **Checkpoint & Recovery**: Auto-save every 50 episodes; emergency save on crash/Ctrl+C; automatic resume on restart
- **Non-blocking Controls**: Keyboard shortcuts via `pynput` — no `input()` blocking
- **LR Plateau Decay**: Automatically halves learning rate when average score stagnates over 50 episodes
- **Structured Logging**: Per-episode logs in `training.log`, metrics CSV every 10 episodes, error traces in `error.log`

## File Structure

```
2048game/
├── core_engine.py       # Game2048 engine (pure game logic)
├── gui_tkinter.py       # Tkinter GUI for manual play
├── config.py            # All hyperparameters (@dataclass)
├── replay_buffer.py     # Experience replay buffer (numpy pre-allocated)
├── dqn_network.py       # Q-network + target network (PyTorch)
├── dqn_agent.py         # Agent: action selection, training step, checkpoint I/O
├── engine_wrapper.py    # Adapts Game2048 to RL interface (reset/step/get_valid_actions)
├── trainer.py           # Training loop, state machine, checkpoint management
├── console_ui.py        # Rich terminal dashboard + keyboard listener
├── main.py              # Entry point with auto-recovery
└── README.md
```

## Requirements

```
torch
numpy
rich
pynput
```

Install:

```bash
pip install torch numpy rich pynput
```

## Quick Start

```bash
python main.py
```

On first launch the trainer starts in **IDLE** state. Press **S** to begin training.

If a checkpoint exists in `./checkpoints/`, the trainer automatically loads it and prints a `[RECOVERY]` message.

## Keyboard Controls

| Key | Action |
|-----|--------|
| **S** | Start / Resume training |
| **P** | Pause (freezes epsilon, steps, CPU usage drops) |
| **R** | Reset all state (requires pressing **R** twice to confirm) |
| **M** | Manual checkpoint save |
| **Q** | Safe quit (finishes current episode before exiting) |

## Hyperparameters

All parameters are defined in `config.py` as a `@dataclass`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 256 | Hidden layer width |
| `num_hidden_layers` | 3 | Number of fully connected hidden layers |
| `learning_rate` | 1e-4 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor |
| `batch_size` | 128 | Mini-batch size for training |
| `buffer_size` | 100,000 | Replay buffer capacity |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Final exploration rate |
| `epsilon_decay_steps` | 100,000 | Steps for exponential epsilon decay |
| `target_update_freq` | 1,000 | Steps between target network syncs |
| `train_freq` | 4 | Steps between training updates |
| `min_buffer_before_train` | 1,000 | Minimum buffer size before training starts |
| `grad_clip` | 10.0 | Max gradient norm for clipping |
| `checkpoint_interval` | 50 | Episodes between auto-saves |
| `lr_decay_enabled` | True | Auto-halve LR on score plateau |
| `lr_decay_window` | 50 | Episodes to evaluate for plateau detection |
| `lr_decay_factor` | 0.5 | LR multiplier on plateau |
| `lr_decay_min` | 1e-7 | Minimum learning rate floor |
| `device` | auto | `auto`, `cuda`, or `cpu` |

## Dashboard Layout

```
╭──────────────────────────────────────────────────────────╮
│ 2048 DQN TRAINER                            14:30:05     │
│ ● RUNNING                                                │
│ [S] Start/Resume  [P] Pause  [R] Reset  [M] Save  [Q] Quit │
├─────────────────────────┬────────────────────────────────┤
│ Current                 │ Statistics                     │
│ Episode:        1,234   │ Avg (10 ep):          3,456   │
│ Ep. Steps:        312   │ Best Score:           8,192   │
│ Score:            2,048 │ Total Steps:      234,567     │
│ Epsilon:    0.0521 ████ │ Buffer:             78.3%     │
│ Learning Rate:   1e-04  │ Next Target:     432 steps    │
│ Last Loss:    0.003124  │                              │
│                         │ TREND (8,192 max)             │
│                         │  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅        │
├─────────────────────────┴────────────────────────────────┤
│ [ BOARD ]                          last: ↑               │
│  ·  2  4  8                                                │
│ 16 32 64 128                                              │
│ 256 512 1K 2K                                             │
│  ·  ·  ·  ·                                               │
╰──────────────────────────────────────────────────────────╯
```

## Checkpoint Structure

Checkpoints are saved to `./checkpoints/`:

```
checkpoints/
├── q_net_{tag}.pth          # Q-network weights
├── target_net_{tag}.pth     # Target network weights
└── metadata_{tag}.json      # Episode, step, best score, epsilon, LR
```

Tags: `latest` (auto-save), `emergency` (crash/Ctrl+C), `manual` (press M).

## Logs

Logs are written to `./logs/`:

- `training.log` — per-episode summary with timestamps
- `metrics.csv` — every 10 episodes: episode, steps, score, avg10, avg100, epsilon, loss
- `error.log` — full traceback on crash
