"""
KIF最終局面とラベルグリッドの対応を検証するスクリプト
data2.kifの最終局面を board の表示と grid の表示で並べて確認する
"""
import shogi
import shogi.KIF
import sys

kif_path = sys.argv[1] if len(sys.argv) > 1 else "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/data2.kif"

PIECE_TO_LABEL = {
    shogi.PAWN:        "fu",
    shogi.LANCE:       "kyo",
    shogi.KNIGHT:      "kei",
    shogi.SILVER:      "gin",
    shogi.GOLD:        "kin",
    shogi.BISHOP:      "kaku",
    shogi.ROOK:        "hi",
    shogi.KING:        "ou",
    shogi.PROM_PAWN:   "tokin",
    shogi.PROM_LANCE:  "nari_kyo",
    shogi.PROM_KNIGHT: "nari_kei",
    shogi.PROM_SILVER: "nari_gin",
    shogi.PROM_BISHOP: "uma",
    shogi.PROM_ROOK:   "ryu",
}

kif = shogi.KIF.Parser.parse_file(kif_path)
game = kif[0]
board = shogi.Board()
for move_str in game['moves']:
    board.push_usi(move_str)

print("=== board標準表示（python-shogi） ===")
print(board)
print()

# ラベルグリッド生成
grid = [["empty"] * 9 for _ in range(9)]
for sq in range(81):
    piece = board.piece_at(sq)
    if piece is None:
        continue
    row = sq % 9
    col = 8 - (sq // 9)
    label = PIECE_TO_LABEL.get(piece.piece_type, "?")
    side = "S" if piece.color == shogi.BLACK else "G"
    grid[row][col] = f"{side}:{label}"

print("=== ラベルグリッド（画像の向き: 左=先手, 上=9筋） ===")
print("    " + "".join([f"col{c:<8}" for c in range(9)]))
for row in range(9):
    筋 = 9 - row
    cells = "".join([f"{grid[row][col]:<11}" for col in range(9)])
    print(f"r{row}({筋}筋) {cells}")
