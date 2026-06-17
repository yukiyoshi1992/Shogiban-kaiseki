"""
駒認識モデル学習スクリプト
- MobileNetV2の転移学習で駒種+先後を分類
- cells/フォルダのラベル別画像を学習データとして使用
- 学習済みモデルをtrain/models/に保存

使い方:
  python train_model.py \
    --cells "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/cells/" \
    --out   "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/models/"

オプション:
  --epochs 20    学習エポック数（デフォルト20）
  --batch  16    バッチサイズ（デフォルト16）
  --val    0.2   検証データ割合（デフォルト0.2）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
import time

# -------------------------------------------------------
# ラベル定義（固定順序）
# -------------------------------------------------------
ALL_LABELS = [
    "empty",
    "sente_fu","sente_kyo","sente_kei","sente_gin","sente_kin",
    "sente_kaku","sente_hi","sente_ou",
    "sente_tokin","sente_nari_kyo","sente_nari_kei","sente_nari_gin",
    "sente_uma","sente_ryu",
    "gote_fu","gote_kyo","gote_kei","gote_gin","gote_kin",
    "gote_kaku","gote_hi","gote_ou",
    "gote_tokin","gote_nari_kyo","gote_nari_kei","gote_nari_gin",
    "gote_uma","gote_ryu",
]
LABEL_TO_IDX = {label: i for i, label in enumerate(ALL_LABELS)}
NUM_CLASSES = len(ALL_LABELS)

# -------------------------------------------------------
# データセット
# -------------------------------------------------------
class CellDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imdecode(
            np.fromfile(str(self.image_paths[idx]), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            # PILに変換してtorchvision transformsを適用
            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, self.labels[idx]

def load_dataset(cells_folder):
    """cells/フォルダから画像パスとラベルを読み込む"""
    cells_folder = Path(cells_folder)
    image_paths = []
    labels = []
    counts = {}

    for label in ALL_LABELS:
        label_dir = cells_folder / label
        if not label_dir.exists():
            continue
        files = list(label_dir.glob("*.jpg"))
        if not files:
            continue
        counts[label] = len(files)
        for f in files:
            image_paths.append(f)
            labels.append(LABEL_TO_IDX[label])

    print(f"総データ数: {len(image_paths)} 枚 / {NUM_CLASSES} クラス")
    print("クラス別枚数:")
    for label, count in counts.items():
        print(f"  {label:<25}: {count:3d}")

    return image_paths, labels

# -------------------------------------------------------
# モデル定義（MobileNetV2転移学習）
# -------------------------------------------------------
def build_model(num_classes, freeze_base=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_base:
        # 最初は特徴抽出層を凍結して分類層だけ学習
        for param in model.features.parameters():
            param.requires_grad = False

    # 分類層を差し替え
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model

# -------------------------------------------------------
# 学習
# -------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nデバイス: {device}")

    # データ読み込み
    print("\n--- データ読み込み ---")
    image_paths, labels = load_dataset(args.cells)

    if len(image_paths) == 0:
        print("[ERROR] 画像が見つかりません")
        sys.exit(1)

    # 学習/検証分割
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=args.val,
        random_state=42
    )
    print(f"\n学習: {len(train_paths)} 枚 / 検証: {len(val_paths)} 枚")

    # Transform定義
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.1),  # 将棋駒は基本左右対称でないので控えめに
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CellDataset(train_paths, train_labels, train_transform)
    val_dataset   = CellDataset(val_paths,   val_labels,   val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch, shuffle=False, num_workers=0)

    # モデル
    print("\n--- モデル構築 ---")
    model = build_model(NUM_CLASSES, freeze_base=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    out_folder = Path(args.out)
    out_folder.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_model_path = out_folder / "best_model.pth"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n--- 学習開始 ({args.epochs} エポック) ---")

    for epoch in range(args.epochs):
        t0 = time.time()

        # --- 学習フェーズ ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, lbls in train_loader:
            images, lbls = images.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(lbls).sum().item()
            train_total += images.size(0)

        # --- 検証フェーズ ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, lbls in val_loader:
                images, lbls = images.to(device), lbls.to(device)
                outputs = model(images)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(lbls).sum().item()
                val_total += images.size(0)

        t_loss = train_loss / train_total
        t_acc  = train_correct / train_total * 100
        v_loss = val_loss / val_total
        v_acc  = val_correct / val_total * 100
        elapsed = time.time() - t0

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(f"Epoch [{epoch+1:2d}/{args.epochs}] "
              f"Train Loss: {t_loss:.4f} Acc: {t_acc:.1f}%  "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.1f}%  "
              f"({elapsed:.1f}s)")

        # ベストモデル保存
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> ベストモデル更新! Val Acc: {v_acc:.1f}%")

        scheduler.step()

        # エポック10でベース層を解凍してfine-tuning
        if epoch == 9:
            print("\n  -> ベース層を解凍してfine-tuning開始")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ラベル情報保存
    meta = {
        "labels": ALL_LABELS,
        "label_to_idx": LABEL_TO_IDX,
        "num_classes": NUM_CLASSES,
        "best_val_acc": best_val_acc,
        "epochs": args.epochs,
        "history": history
    }
    meta_path = out_folder / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n=== 学習完了 ===")
    print(f"ベスト検証精度: {best_val_acc:.1f}%")
    print(f"モデル保存先: {best_model_path}")
    print(f"メタ情報: {meta_path}")
    print("\n次のステップ: predict_cell.pyで駒認識テスト")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="駒認識モデル学習")
    parser.add_argument("--cells",  required=True, help="cells/フォルダのパス")
    parser.add_argument("--out",    required=True, help="モデル出力フォルダ")
    parser.add_argument("--epochs", type=int,   default=20,  help="エポック数")
    parser.add_argument("--batch",  type=int,   default=16,  help="バッチサイズ")
    parser.add_argument("--val",    type=float, default=0.2, help="検証データ割合")
    args = parser.parse_args()
    train(args)