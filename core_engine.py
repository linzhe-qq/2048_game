import random
import copy


class Game2048:
    DIRECTIONS = ('left', 'right', 'up', 'down')

    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.game_over = False
        self._spawn_tile()
        self._spawn_tile()

    def get_grid(self):
        return copy.deepcopy(self.grid)

    def get_score(self):
        return self.score

    def is_game_over(self):
        return self.game_over

    def move(self, direction):
        if direction not in self.DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction}. Must be one of {self.DIRECTIONS}")

        old_grid = copy.deepcopy(self.grid)

        if direction == 'left':
            self._slide_left()
        elif direction == 'right':
            self._slide_right()
        elif direction == 'up':
            self._slide_up()
        elif direction == 'down':
            self._slide_down()

        moved = self.grid != old_grid
        if moved:
            self._spawn_tile()
            self.game_over = not self._has_valid_moves()

        return moved, self.game_over

    def _spawn_tile(self):
        empty = [(r, c) for r in range(4) for c in range(4) if self.grid[r][c] == 0]
        if empty:
            r, c = random.choice(empty)
            self.grid[r][c] = 4 if random.random() < 0.1 else 2

    def _has_valid_moves(self):
        for r in range(4):
            for c in range(4):
                if self.grid[r][c] == 0:
                    return True
                if c < 3 and self.grid[r][c] == self.grid[r][c + 1]:
                    return True
                if r < 3 and self.grid[r][c] == self.grid[r + 1][c]:
                    return True
        return False

    def _merge_line(self, line):
        filtered = [v for v in line if v != 0]
        merged = []
        score_gain = 0
        i = 0
        while i < len(filtered):
            if i + 1 < len(filtered) and filtered[i] == filtered[i + 1]:
                merged.append(filtered[i] * 2)
                score_gain += filtered[i] * 2
                i += 2
            else:
                merged.append(filtered[i])
                i += 1
        merged += [0] * (4 - len(merged))
        return merged, score_gain

    def _slide_left(self):
        for r in range(4):
            self.grid[r], gain = self._merge_line(self.grid[r])
            self.score += gain

    def _slide_right(self):
        for r in range(4):
            reversed_line = self.grid[r][::-1]
            merged, gain = self._merge_line(reversed_line)
            self.grid[r] = merged[::-1]
            self.score += gain

    def _slide_up(self):
        for c in range(4):
            col = [self.grid[r][c] for r in range(4)]
            merged, gain = self._merge_line(col)
            for r in range(4):
                self.grid[r][c] = merged[r]
            self.score += gain

    def _slide_down(self):
        for c in range(4):
            col = [self.grid[r][c] for r in range(4)][::-1]
            merged, gain = self._merge_line(col)
            merged = merged[::-1]
            for r in range(4):
                self.grid[r][c] = merged[r]
            self.score += gain
