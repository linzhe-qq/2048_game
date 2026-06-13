import tkinter as tk
from tkinter import messagebox
from core_engine import Game2048

TILE_COLORS = {
    0: "#cdc1b4",
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
}

TILE_FONTSIZE = {
    (0, 64): 48,
    (128, 1024): 40,
    (2048, float('inf')): 32,
}


def _get_font_size(value):
    for (low, high), size in TILE_FONTSIZE.items():
        if low <= value <= high:
            return size
    return 28


class GameWindow:
    CELL_SIZE = 100
    GAP = 10
    PADDING = 15

    def __init__(self):
        self.game = Game2048()

        self.root = tk.Tk()
        self.root.title("2048")
        self.root.resizable(False, False)
        self.root.bind("<Key>", self._on_key)

        board_size = self.CELL_SIZE * 4 + self.GAP * 3 + self.PADDING * 2

        # Header
        header = tk.Frame(self.root, bg="#faf8ef")
        header.pack(fill=tk.X, padx=10, pady=(10, 0))

        self.score_label = tk.Label(
            header, text=f"Score: {self.game.get_score()}",
            font=("Arial", 20, "bold"), bg="#faf8ef", fg="#776e65",
        )
        self.score_label.pack(side=tk.LEFT)

        restart_btn = tk.Button(
            header, text="New Game", font=("Arial", 14),
            bg="#8f7a66", fg="white", relief=tk.FLAT,
            command=self._restart,
        )
        restart_btn.pack(side=tk.RIGHT)

        # Board
        self.canvas = tk.Canvas(
            self.root, width=board_size, height=board_size,
            bg="#bbada0", highlightthickness=0,
        )
        self.canvas.pack(pady=10)

        self.tiles = []
        for r in range(4):
            row_tiles = []
            for c in range(4):
                x0 = self.PADDING + c * (self.CELL_SIZE + self.GAP)
                y0 = self.PADDING + r * (self.CELL_SIZE + self.GAP)
                x1 = x0 + self.CELL_SIZE
                y1 = y0 + self.CELL_SIZE
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill="#cdc1b4")
                text = self.canvas.create_text(
                    (x0 + x1) / 2, (y0 + y1) / 2,
                    text="", font=("Arial", 48, "bold"),
                )
                row_tiles.append((rect, text))
            self.tiles.append(row_tiles)

        self._draw()

    def _restart(self):
        self.game.reset()
        self._draw()

    def _draw(self):
        grid = self.game.get_grid()
        for r in range(4):
            for c in range(4):
                value = grid[r][c]
                rect, text = self.tiles[r][c]
                color = TILE_COLORS.get(value, "#3c3a32")
                self.canvas.itemconfig(rect, fill=color)
                if value == 0:
                    self.canvas.itemconfig(text, text="")
                else:
                    fg = "#776e65" if value <= 4 else "#f9f6f2"
                    self.canvas.itemconfig(
                        text, text=str(value), fill=fg,
                        font=("Arial", _get_font_size(value), "bold"),
                    )
        self.score_label.config(text=f"Score: {self.game.get_score()}")

    def _on_key(self, event):
        key_map = {
            "Left": "left", "Right": "right",
            "Up": "up", "Down": "down",
            "a": "left", "d": "right",
            "w": "up", "s": "down",
        }
        direction = key_map.get(event.keysym)
        if direction is None:
            return

        moved, over = self.game.move(direction)
        if moved:
            self._draw()
            if over:
                self.root.unbind("<Key>")
                messagebox.showinfo("Game Over", f"Final Score: {self.game.get_score()}")
                self.root.bind("<Key>", self._on_key)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    GameWindow().run()
