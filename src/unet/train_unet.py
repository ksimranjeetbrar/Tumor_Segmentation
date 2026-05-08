#!/usr/bin/env python
import argparse, glob, os
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ------------------------- Data -------------------------
class SliceDataset(Dataset):
    def __init__(self, root, split="train", img_size=256):
        self.img_dir = Path(root) / split / "images"
        self.msk_dir = Path(root) / split / "masks"
        self.imgs = sorted(glob.glob(str(self.img_dir / "*.png")))
        self.img_size = img_size
        if not self.imgs:
            raise RuntimeError(f"No images found in {self.img_dir}")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        ip = Path(self.imgs[i])
        mp = self.msk_dir / f"{ip.stem}_mask.png"

        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise FileNotFoundError(ip.name)

        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_AREA)
        if msk.shape[:2] != (self.img_size, self.img_size):
            msk = cv2.resize(msk, (self.img_size, self.img_size), cv2.INTER_NEAREST)

        img = (img / 255.0).astype(np.float32)[None, ...]  # (1,H,W)
        msk = ((msk > 0).astype(np.float32))[None, ...]     # (1,H,W)
        return torch.from_numpy(img), torch.from_numpy(msk)

# ------------------------- Model -------------------------
class DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, ic=1, oc=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(ic, base);    self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2);self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4);self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base*4, base*8);self.p4 = nn.MaxPool2d(2)
        self.b  = DoubleConv(base*8, base*16)
        self.u4 = nn.ConvTranspose2d(base*16, base*8, 2, 2); self.c4 = DoubleConv(base*16, base*8)
        self.u3 = nn.ConvTranspose2d(base*8,  base*4, 2, 2); self.c3 = DoubleConv(base*8,  base*4)
        self.u2 = nn.ConvTranspose2d(base*4,  base*2, 2, 2); self.c2 = DoubleConv(base*4,  base*2)
        self.u1 = nn.ConvTranspose2d(base*2,  base,    2, 2); self.c1 = DoubleConv(base*2,  base)
        self.h  = nn.Conv2d(base, oc, 1)

    def forward(self, x):
        c1 = self.d1(x); p1 = self.p1(c1)
        c2 = self.d2(p1); p2 = self.p2(c2)
        c3 = self.d3(p2); p3 = self.p3(c3)
        c4 = self.d4(p3); p4 = self.p4(c4)
        b  = self.b(p4)
        u4 = self.u4(b);  d4 = self.c4(torch.cat([u4, c4], 1))
        u3 = self.u3(d4); d3 = self.c3(torch.cat([u3, c3], 1))
        u2 = self.u2(d3); d2 = self.c2(torch.cat([u2, c2], 1))
        u1 = self.u1(d2); d1 = self.c1(torch.cat([u1, c1], 1))
        return self.h(d1)  # logits

# ------------------------- Loss & Metrics -------------------------
def dice_loss(logits, tgt, smooth=1.0):
    p = torch.sigmoid(logits)
    num = 2.0 * (p * tgt).sum((2, 3)) + smooth
    den = p.sum((2, 3)) + tgt.sum((2, 3)) + smooth
    return (1.0 - num / den).mean()

def bce_dice(logits, tgt):
    return 0.5 * nn.functional.binary_cross_entropy_with_logits(logits, tgt) + 0.5 * dice_loss(logits, tgt)

@torch.no_grad()
def dice_iou_smoothed(logits, tgt, thr=0.5, smooth=1.0):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * tgt).sum((1, 2, 3))
    sum_p  = p.sum((1, 2, 3))
    sum_t  = tgt.sum((1, 2, 3))
    dice = ((2.0 * inter + smooth) / (sum_p + sum_t + smooth)).mean().item()
    union = (p + tgt - p * tgt).sum((1, 2, 3))
    iou  = ((inter + smooth) / (union + smooth)).mean().item()
    return dice, iou

# ------------------------- Train -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()

    tr = SliceDataset(a.data_root, "train", a.img_size)
    va = SliceDataset(a.data_root, "val",   a.img_size)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_ld = DataLoader(tr, batch_size=a.batch_size, shuffle=True,  num_workers=0, pin_memory=(dev.type=="cuda"))
    va_ld = DataLoader(va, batch_size=a.batch_size, shuffle=False, num_workers=0, pin_memory=(dev.type=="cuda"))

    model = UNet(1, 1, base=a.base).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    os.makedirs(a.out_dir, exist_ok=True)
    best = -1.0

    for ep in range(1, a.epochs + 1):
        # train
        model.train(); run = 0.0
        for img, msk in tqdm(tr_ld, desc=f"train {ep}"):
            img, msk = img.to(dev), msk.to(dev)
            opt.zero_grad()
            out = model(img)
            loss = bce_dice(out, msk)
            loss.backward()
            opt.step()
            run += loss.item() * img.size(0)
        tl = run / len(tr)

        # val (smoothed metrics; won’t show hard zeros)
        model.eval(); vl = 0.0; vd = 0.0; vi = 0.0; nimg = 0
        with torch.no_grad():
            for img, msk in tqdm(va_ld, desc="val"):
                img, msk = img.to(dev), msk.to(dev)
                out = model(img)
                vl += bce_dice(out, msk).item() * img.size(0)
                d, i = dice_iou_smoothed(out, msk, thr=a.thr, smooth=1.0)
                vd += d * img.size(0)
                vi += i * img.size(0)
                nimg += img.size(0)
        vl /= len(va)
        vd /= nimg; vi /= nimg
        print(f"[{ep:02d}] train {tl:.4f} | val {vl:.4f} | Dice {vd:.6f} | IoU {vi:.6f}")

        if vd > best:
            best = vd
            torch.save(model.state_dict(), os.path.join(a.out_dir, "best.pt"))

if __name__ == "__main__":
    main()
