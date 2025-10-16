import sys, os, json, glob
from pathlib import Path

# Đảm bảo import được module src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def normalize_game(path):
    """Đọc 1 file game, chuẩn hóa key và tái dựng board nếu cần"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Chuẩn hóa key 'result' -> 'winner'
    if "result" in data:
        data["winner"] = data["result"]

    # Nếu chưa có board → tái dựng lại từ moves
    if "board" not in data:
        from src.game import Board
        b = Board(size=15)
        player = 1
        for (r, c) in data["moves"]:
            b.play(r, c, player)
            player *= -1
        data["board"] = b.grid.astype(int).tolist()

    return data


# 🧩 Gom dữ liệu từ nhiều nguồn
all_files = (
    glob.glob("data/run_game/*.json")
    + glob.glob("data/selfplay_round1_500/*.json")
    + glob.glob("data/selfplay_round2_1200/*.json")
    + glob.glob("data/selfplay_round3_500/*.json")
)

# 🗂️ Thư mục đích: data/professional
target_dir = Path("data/professional")
target_dir.mkdir(parents=True, exist_ok=True)

# 🔍 Đếm số file hiện có → đánh số tiếp theo
existing_files = sorted(target_dir.glob("game_*.json"))
start_index = len(existing_files)
count = start_index

print(f"🔎 Đã có {start_index} file cũ trong professional, sẽ thêm tiếp từ game_{count:05d}.json\n")

# 🧠 Lưu từng ván mới vào professional
for path in all_files:
    try:
        data = normalize_game(path)
        out_path = target_dir / f"game_{count:05d}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        count += 1
    except Exception as e:
        print(f"⚠️ Lỗi ở file {path}: {e}")

print(f"\n✅ Đã thêm {count - start_index} ván đấu mới vào data/professional/")
print(f"📦 Tổng số hiện tại: {count} file trong thư mục professional.")
