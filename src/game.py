# lớp Board, luật Caro, sinh nước đi

import numpy as np

class Board:
    def __init__(self, size=15):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.move_stack = []
        self.last_move = None

    def play(self, r, c, player):
        """Đánh quân cờ"""
        assert self.grid[r, c] == 0, "Ô đã có quân!"
        self.grid[r, c] = player
        self.move_stack.append((r, c, player))
        self.last_move = (r, c)

    def undo(self):
        """Hoàn tác nước đi"""
        r, c, p = self.move_stack.pop()
        self.grid[r, c] = 0
        self.last_move = self.move_stack[-1][:2] if self.move_stack else None

    def is_win(self, last_move=None):
        """Kiểm tra thắng 5 liên tiếp"""
        if last_move is None:
            last_move = self.last_move
        if last_move is None:
            return False
        r, c = last_move
        player = self.grid[r, c]
        if player == 0:
            return False
        dirs = [(0,1),(1,0),(1,1),(1,-1)]
        for dr, dc in dirs:
            cnt = 1
            rr, cc = r+dr, c+dc
            while 0 <= rr < self.size and 0 <= cc < self.size and self.grid[rr, cc] == player:
                cnt += 1; rr += dr; cc += dc
            rr, cc = r-dr, c-dc
            while 0 <= rr < self.size and 0 <= cc < self.size and self.grid[rr, cc] == player:
                cnt += 1; rr -= dr; cc -= dc
            if cnt >= 5:
                return True
        return False
    

    def is_draw(self):
        return np.all(self.grid != 0)

    def generate_candidates(self, radius=1):
        """Sinh nước đi gần các quân cờ đã có"""
        coords = set()
        stones = np.argwhere(self.grid != 0)
        if len(stones) == 0:
            return [(self.size//2, self.size//2)]
        for r, c in stones:
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < self.size and 0 <= cc < self.size:
                        if self.grid[rr, cc] == 0:
                            coords.add((rr, cc))
        return list(coords)
