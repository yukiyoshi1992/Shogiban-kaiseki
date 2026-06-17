"""
square番号と駒の対応を詳細出力
python-shogiのboard.piece_at(square)が返す駒を、
SQUARE_NAMESと突き合わせて確認する
"""
import shogi
import shogi.KIF
import sys

kif_path = sys.argv[1] if len(sys.argv) > 1 else "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/data2.kif"

kif = shogi.KIF.Parser.parse_file(kif_path)
game = kif[0]
board = shogi.Board()
for move_str in game['moves']:
    board.push_usi(move_str)

PIECE_SYMBOLS = ['', 'p', 'l', 'n', 's', 'g', 'b', 'r', 'k',
                 '+p', '+l', '+n', '+s', '+b', '+r']

# 全squareの駒をSQUARE_NAMEと共に出力
print("square | name | piece | color")
print("-" * 40)
for sq in range(81):
    piece = board.piece_at(sq)
    name = shogi.SQUARE_NAMES[sq]
    if piece is None:
        continue
    sym = PIECE_SYMBOLS[piece.piece_type]
    color = "sente(B)" if piece.color == shogi.BLACK else "gote(W)"
    print(f"  {sq:2d}   | {name}  | {sym:3s} | {color}")

print()
print("特定駒の位置確認:")
print("（KIF最終局面で先手の馬は9筋1段='9a'=square0付近にいるはず）")
for sq in range(81):
    piece = board.piece_at(sq)
    if piece and piece.piece_type == shogi.PROM_BISHOP:
        print(f"  馬(uma): square={sq}, name={shogi.SQUARE_NAMES[sq]}, "
              f"color={'sente' if piece.color==shogi.BLACK else 'gote'}")
    if piece and piece.piece_type == shogi.KING:
        print(f"  玉(ou):  square={sq}, name={shogi.SQUARE_NAMES[sq]}, "
              f"color={'sente' if piece.color==shogi.BLACK else 'gote'}")