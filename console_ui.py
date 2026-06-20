import threading
import time
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    from pynput import keyboard as pynput_keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

ACTION_ARROWS = {0: "←", 1: "→", 2: "↑", 3: "↓", None: "·"}

TILE_STYLES = {
    0:    ("·", "bright_black"),
    2:    ("2", "bright_white"),
    4:    ("4", "bright_yellow"),
    8:    ("8", "bright_magenta"),
    16:   ("16", "bright_cyan"),
    32:   ("32", "cyan"),
    64:   ("64", "green"),
    128:  ("128", "yellow"),
    256:  ("256", "bright_yellow"),
    512:  ("512", "bright_red"),
    1024: ("1K", "red"),
    2048: ("2K", "bright_red on yellow"),
}


def _num(n) -> str:
    """Format number with thousand separators."""
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def _render_board(grid: list[list[int]], last_action) -> Text:
    t = Text()
    header = Text()
    header.append(" [ BOARD ]", style="bold cyan")
    header.append("  " * 5)
    arrow = ACTION_ARROWS.get(last_action, "·")
    header.append(f"last: {arrow}", style="bold white")
    t.append_text(header)
    t.append("\n")

    for row in grid:
        for val in row:
            label, style = TILE_STYLES.get(val, (str(val), "bright_white"))
            pad = 4 - len(label)
            t.append(f" {label}{' ' * pad}", style=style)
        t.append("\n")
    return t


def _render_mini_chart(scores: list[int]) -> Text:
    BLOCKS = "▁▂▃▄▅▆▇█"
    WIDTH = 20
    HEIGHT = 5
    t = Text()

    if not scores:
        t.append(" TREND\n", style="bold cyan")
        for _ in range(HEIGHT):
            t.append(" " * WIDTH + "\n", style="dim")
        return t

    # Aggregate to WIDTH columns
    cols: list[float] = []
    chunk = max(1, len(scores) // WIDTH)
    for i in range(WIDTH):
        start = i * chunk
        end = min(start + chunk, len(scores))
        if start < len(scores):
            cols.append(sum(scores[start:end]) / (end - start))
        else:
            cols.append(0)

    lo, hi = min(cols), max(cols)
    span = hi - lo if hi > lo else 1.0

    t.append(" TREND", style="bold cyan")
    t.append(f" ({_num(int(hi))} max)\n", style="dim")

    grid_chars = [[" " for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for col_idx in range(min(len(cols), WIDTH)):
        h = round((cols[col_idx] - lo) / span * (HEIGHT - 1))
        h = max(0, min(HEIGHT - 1, h))
        row_idx = HEIGHT - 1 - h
        grid_chars[row_idx][col_idx] = BLOCKS[min(h, len(BLOCKS) - 1)]

    for row in grid_chars:
        t.append(" " + "".join(row) + "\n", style="green")

    return t


def _status_indicator(state: str) -> Text:
    t = Text(" ")
    if state == "running":
        t.append("● RUNNING", style="bold green")
    elif state == "paused":
        t.append("‖ PAUSED", style="bold yellow")
    elif state == "stopped":
        t.append("■ STOPPED", style="bold red")
    else:
        t.append("○ IDLE", style="bold bright_black")
    return t


class ConsoleUI:
    REFRESH_INTERVAL = 0.15  # seconds between UI refreshes

    def __init__(self, trainer):
        self.trainer = trainer
        self.console = Console()
        self._confirm_reset = False
        self._listener = None

    # ── keyboard ────────────────────────────────────────────
    def _start_keyboard_listener(self):
        if not HAS_PYNPUT:
            return

        def on_press(key):
            try:
                ch = key.char
            except AttributeError:
                ch = None

            if ch in ("s", "S"):
                self._confirm_reset = False
                self.trainer.start()
            elif ch in ("p", "P"):
                self._confirm_reset = False
                self.trainer.pause()
            elif ch in ("r", "R"):
                if self._confirm_reset:
                    self.trainer.reset_all()
                    self._confirm_reset = False
                else:
                    self._confirm_reset = True
            elif ch in ("m", "M"):
                self._confirm_reset = False
                self.trainer.save_checkpoint("manual")
            elif ch in ("q", "Q"):
                self._confirm_reset = False
                self.trainer.safe_quit()

        self._listener = pynput_keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    def _stop_keyboard_listener(self):
        if self._listener:
            self._listener.stop()

    # ── render ──────────────────────────────────────────────
    def _build_display(self) -> Panel:
        snap = self.trainer.snapshot()

        # ── Header ──
        now = datetime.now().strftime("%H:%M:%S")
        title = Text()
        title.append("2048 DQN TRAINER", style="bold cyan")
        title.append(f"  {now}", style="dim")

        status_line = _status_indicator(snap["state"])

        hotkeys = Text()
        hotkeys.append(" [S]", style="bold green")
        hotkeys.append(" Start/Resume  ")
        hotkeys.append("[P]", style="bold yellow")
        hotkeys.append(" Pause  ")
        hotkeys.append("[R]", style="bold red")
        hotkeys.append(" Reset  ")
        hotkeys.append("[M]", style="bold blue")
        hotkeys.append(" Save  ")
        hotkeys.append("[Q]", style="bold bright_black")
        hotkeys.append(" Quit")

        if self._confirm_reset:
            hotkeys.append("  ⚠ Press R again to confirm reset!", style="bold red")

        # ── Left panel: current run stats ──
        eps = snap["epsilon"]
        prog = snap["epsilon_progress"]
        filled = int(prog * 20)
        bar = "█" * filled + "░" * (20 - filled)

        left = Table.grid(padding=(0, 2))
        left.add_column(style="bold cyan", width=14)
        left.add_column()
        left.add_row("Episode:", _num(snap["episode"]))
        left.add_row("Ep. Steps:", _num(snap["episode_steps"]))
        left.add_row("Score:", _num(snap["score"]))
        left.add_row("Epsilon:", f"{eps:.4f}  {bar}")
        left.add_row("Learning Rate:", f"{snap['lr']:.2e}")
        left.add_row("Last Loss:", f"{snap['last_loss']:.6f}" if snap["last_loss"] else "—")

        # ── Right panel: aggregate stats ──
        right = Table.grid(padding=(0, 2))
        right.add_column(style="bold magenta", width=16)
        right.add_column()
        right.add_row("Avg (10 ep):", _num(round(snap["avg10"])))
        right.add_row("Best Score:", _num(snap["best_score"]))
        right.add_row("Total Steps:", _num(snap["total_steps"]))
        right.add_row("Buffer:", f"{snap['buffer_usage'] * 100:.1f}%")
        right.add_row("Next Target:", f"{_num(snap['steps_to_target_update'])} steps")

        # ── Mini chart ──
        chart = _render_mini_chart(snap["recent_scores"])

        # ── Board ──
        board = _render_board(snap["grid"], snap["last_action"])

        # ── Compose layout ──
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body"),
            Layout(name="board", size=7),
        )
        layout["header"].split_column(
            Layout(title, size=1),
            Layout(status_line, size=1),
            Layout(hotkeys, size=1),
        )
        layout["body"].split_row(
            Layout(Panel(left, title="Current", border_style="cyan"), ratio=1),
            Layout(name="right_col", ratio=1),
        )
        layout["right_col"].split_column(
            Layout(Panel(right, title="Statistics", border_style="magenta")),
            Layout(Panel(chart, border_style="green")),
        )
        layout["board"].update(Panel(board, border_style="bright_black"))

        return Panel(layout, border_style="bright_black")

    # ── main entry ──────────────────────────────────────────
    def run(self):
        self._start_keyboard_listener()

        try:
            with Live(
                self._build_display(),
                console=self.console,
                refresh_per_second=int(1 / self.REFRESH_INTERVAL),
                screen=True,
            ) as live:
                # Start training in a background thread
                train_thread = threading.Thread(target=self.trainer.train, daemon=True)
                train_thread.start()

                while True:
                    time.sleep(self.REFRESH_INTERVAL)
                    live.update(self._build_display())
                    if self.trainer.state == self.trainer.STOPPED:
                        # Give a moment for final render
                        time.sleep(0.5)
                        live.update(self._build_display())
                        break
        except KeyboardInterrupt:
            self.trainer.safe_quit()
        finally:
            self._stop_keyboard_listener()
