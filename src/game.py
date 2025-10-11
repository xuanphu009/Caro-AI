# lớp Board, luật Caro, sinh nước đi

# Cài đặt Board cơ bản cho Caro 15x15
# Giá trị ô: 0 = trống, 1 = player1 (X), -1 = player2 (O)

import numpy as np
from typing import List, Tuple, Optional

Coord = Tuple[int,int]

class Board:
    def __init__(self, size: int = 15):
        self.size = size
        # grid dạng numpy để xử lý tiện
        self.grid = np.zeros((size, size), dtype=np.int8)
        # stack lưu từng nước để có thể undo nhanh
        self.move_stack: List[Tuple[int,int,int]] = []  # list of (r,c,player)
        self.last_move: Optional[Coord] = None

        # Zobrist hashing (tạo khi init, dùng cho TT)
        self.zobrist_table = np.random.randint(
            low=0, high=2**63, size=(size, size, 2), dtype=np.int64
        )
        # thêm 1 số random cho side-to-move
        self.zobrist_side = np.random.randint(low=0, high=2**63, dtype=np.int64)
        self.zobrist_hash = np.int64(0)

        self.move_count = 0

    def reset(self):
        """Reset board về trạng thái ban đầu"""
        self.grid.fill(0)
        self.move_stack.clear()
        self.last_move = None
        self.zobrist_hash = np.int64(0)

        self.move_count = 0

    def play(self, r: int, c: int, player: int):
        """Đánh tại (r,c) bởi player (1 hoặc -1). Không kiểm tra hợp lệ ở đây (caller phải đảm bảo)."""
        assert 0 <= r < self.size and 0 <= c < self.size, "Out of bounds"
        assert self.grid[r, c] == 0, "Ô đã có quân"
        self.grid[r, c] = player
        self.move_stack.append((r, c, player))
        self.last_move = (r, c)
        # cập nhật zobrist
        piece_index = 0 if player == 1 else 1
        self.zobrist_hash ^= np.int64(self.zobrist_table[r, c, piece_index])
        # toggle side (nếu bạn dùng side bit)
        self.zobrist_hash ^= np.int64(self.zobrist_side)

        self.move_count += 1

    def undo(self):
        """Hoàn tác nước đi cuối cùng"""
        if not self.move_stack:
            return
        r, c, player = self.move_stack.pop()
        self.grid[r, c] = 0
        self.last_move = (self.move_stack[-1][0], self.move_stack[-1][1]) if self.move_stack else None
        piece_index = 0 if player == 1 else 1
        # undo zobrist: XOR cùng giá trị sẽ phục hồi
        self.zobrist_hash ^= np.int64(self.zobrist_table[r, c, piece_index])
        self.zobrist_hash ^= np.int64(self.zobrist_side)

        self.move_count -= 1

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    #Kiểm tra thắng
    def is_win_from(self, r: int, c: int) -> bool:
        """Kiểm tra thắng bắt đầu từ (r,c) — gọi sau khi đã play tại (r,c)."""
        player = self.grid[r, c]
        if player == 0:
            return False
        directions = [(0,1),(1,0),(1,1),(1,-1)]
        for dr, dc in directions:
            cnt = 1
            # đi về phía dương
            rr, cc = r + dr, c + dc
            while self.in_bounds(rr, cc) and self.grid[rr, cc] == player:
                cnt += 1
                rr += dr; cc += dc
            # đi về phía âm
            rr, cc = r - dr, c - dc
            while self.in_bounds(rr, cc) and self.grid[rr, cc] == player:
                cnt += 1
                rr -= dr; cc -= dc
            if cnt >= 5:
                return True
        return False

    # Kiểm tra hòa
    def is_draw(self) -> bool:
        # Ván hòa khi toàn bộ bàn cờ đã kín mà không ai thắng.
        return not (self.grid == 0).any()

    
    
    # Sinh nước đi

    def legal_moves(self):

        # Trả về toàn bộ các ô trống trên bàn.
        # Cách này dùng cho test cơ bản, nhưng rất chậm khi bàn lớn.
        return [(r, c) for r in range(self.size) for c in range(self.size) if self.grid[r, c] == 0]

    def generate_candidates(self, radius: int = 2) -> List[Coord]:
        """
        Sinh các ô ứng viên: chỉ các ô trống trong bán kính 'radius' xung quanh
        các ô đã có quân. Nếu board trống -> trả center.
        """
        occupied = np.argwhere(self.grid != 0)
        if occupied.shape[0] == 0:
            return [(self.size // 2, self.size // 2)]
        cand = set()
        for (r, c) in occupied:
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    rr, cc = int(r)+dr, int(c)+dc
                    if self.in_bounds(rr, cc) and self.grid[rr, cc] == 0:
                        cand.add((rr, cc))
        return list(cand)

    def to_cnn_input(self, current_player: int) -> np.ndarray:
        """
        Trả về input cho CNN: shape (2, size, size)
        channel0 = 1 where current_player has stone
        channel1 = 1 where opponent has stone
        """
        p = (self.grid == current_player).astype(np.uint8)
        o = (self.grid == -current_player).astype(np.uint8)
        return np.stack([p, o], axis=0)
    
    def print_board(self):
        """In board ra console (debug)"""
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                v = self.grid[r, c]
                if v == 1:
                    row += " X"
                elif v == -1:
                    row += " O"
                else:
                    row += " ."
            print(row)

