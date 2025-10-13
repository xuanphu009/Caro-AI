# xử lý & load dữ liệu


import torch
from torch.utils.data import Dataset
import numpy as np
import json, os

class CaroDataset(Dataset):
    def __init__(self, data_dir="data/selfplay"):
        self.boards = []
        self.results = []
        self._load_data(data_dir)

    def _load_data(self, data_dir):
        for fname in os.listdir(data_dir):
            if not fname.endswith(".json"): continue
            with open(os.path.join(data_dir, fname), "r") as f:
                game = json.load(f)
            moves = game["moves"]
            result = game["result"]

            # reconstruct all intermediate states
            board = np.zeros((15, 15), dtype=np.int8)
            player = 1
            for i, (r, c) in enumerate(moves):
                # state before move
                player_layer = (board == player).astype(np.float32)
                opp_layer = (board == -player).astype(np.float32)
                state = np.stack([player_layer, opp_layer], axis=0)
                self.boards.append(state)
                self.results.append(result if player == 1 else -result)

                board[r, c] = player
                player = -player

        self.boards = np.array(self.boards, dtype=np.float32)
        self.results = np.array(self.results, dtype=np.float32)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        x = torch.tensor(self.boards[idx])
        y = torch.tensor(self.results[idx])
        return x, y

    def is_full(self, index: int = -1) -> bool:
        """
        Kiểm tra bàn cờ có đầy chưa.
        index: chỉ số của ván trong dataset (mặc định là ván cuối cùng)
        """
        if len(self.boards) == 0:
            return False
        board = self.boards[index]
        return not (board == 0).any()

