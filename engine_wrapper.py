import numpy as np
from core_engine import Game2048


class Game2048Env:
    ACTION_MAP = {0: "left", 1: "right", 2: "up", 3: "down"}

    def __init__(self):
        self.game = Game2048()

    def reset(self) -> np.ndarray:
        self.game.reset()
        return self._state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        old_score = self.game.score
        direction = self.ACTION_MAP[action]
        moved, game_over = self.game.move(direction)

        reward = float(self.game.score - old_score)
        if not moved:
            reward = -1.0
            if not self.game._has_valid_moves():
                game_over = True

        return self._state(), reward, game_over, {"moved": moved}

    def get_valid_actions(self) -> list[int]:
        grid_np = np.array(self.game.grid, dtype=np.int32)
        saved_score = self.game.score
        saved_over = self.game.game_over
        valid: list[int] = []
        for action, direction in self.ACTION_MAP.items():
            moved, _ = self.game.move(direction)
            if moved:
                valid.append(action)
            # Restore: numpy copy is ~10x faster than deepcopy for 4x4
            self.game.grid = grid_np.tolist()
            self.game.score = saved_score
            self.game.game_over = saved_over
        return valid

    def get_grid(self) -> list[list[int]]:
        return self.game.get_grid()

    def get_score(self) -> int:
        return self.game.score

    # ── internal ────────────────────────────────────────────
    def _state(self) -> np.ndarray:
        grid = self.game.grid
        return np.array(
            [np.log2(v + 1) for row in grid for v in row],
            dtype=np.float32,
        )
