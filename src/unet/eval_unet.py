#!/usr/bin/env python
import argparse, csv, glob, os
from pathlib import Path
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ----------------- Data -----------------
class SliceDataset(Dataset):
    def __init__(self, root, split="test", img_size=256):
        self.img_dir = Path(root) / split / "images"
        self.msk_dir = Path(root) / split / "masks"
        self.imgs = sorted(glob.glob(str(self.img_dir / "*.png")))
        self.img_size = img_size
        if not self.imgs:
            raise RuntimeError(f"No images in {self.img_dir}")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        ip = Path(self.imgs[i])
        mp = self.msk_dir / f"{ip.stem}_mask.png"

        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        gt  = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            raise FileNotFoundError(ip.name)

        # resize for model / overlay
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_AREA)
        if gt.shape[:2] != (self.img_size, self.img_size):
            gt  = cv2.resize(gt,  (self.img_size, self.img_size), cv2.INTER_NEAREST)

        img_vis = img.copy()  # uint8 for drawing
        img_t = (img / 255.0).astype(np.float32)[None, ...]   # (1,H,W)
        gt_t  = ((gt > 0).astype(np.float32))[None, ...]      # (1,H,W)
        return ip.stem, img_vis, torch.from_numpy(img_t), torch.from_numpy(gt_t)

# ----------------- Model -----------------
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
        self.d1=DoubleConv(ic,base); self.p1=nn.MaxPool2d(2)
        self.d2=DoubleConv(base,base*2); self.p2=nn.MaxPool2d(2)
        self.d3=DoubleConv(base*2,base*4); self.p3=nn.MaxPool2d(2)
        self.d4=DoubleConv(base*4,base*8); self.p4=nn.MaxPool2d(2)
        self.b =DoubleConv(base*8,base*16)
        self.u4=nn.ConvTranspose2d(base*16,base*8,2,2); self.c4=DoubleConv(base*16,base*8)
        self.u3=nn.ConvTranspose2d(base*8,base*4,2,2);  self.c3=DoubleConv(base*8,base*4)
        self.u2=nn.ConvTranspose2d(base*4,base*2,2,2);  self.c2=DoubleConv(base*4,base*2)
        self.u1=nn.ConvTranspose2d(base*2,base,2,2);    self.c1=DoubleConv(base*2,base)
        self.h =nn.Conv2d(base, oc, 1)
    def forward(self,x):
        c1=self.d1(x); p1=self.p1(c1)
        c2=self.d2(p1); p2=self.p2(c2)
        c3=self.d3(p2); p3=self.p3(c3)
        c4=self.d4(p3); p4=self.p4(c4)
        b =self.b(p4)
        u4=self.u4(b); d4=self.c4(torch.cat([u4,c4],1))
        u3=self.u3(d4); d3=self.c3(torch.cat([u3,c3],1))
        u2=self.u2(d3); d2=self.c2(torch.cat([u2,c2],1))
        u1=self.u1(d2); d1=self.c1(torch.cat([u1,c1],1))
        return self.h(d1)  # logits

# ----------------- Metrics (smoothed) -----------------
@torch.no_grad()
def dice_iou_smoothed(logits, tgt, thr=0.5, smooth=1.0):
    p = torch.sigmoid(logits)
    ph = (p > thr).float()
    inter = (ph * tgt).sum()
    sum_p = ph.sum(); sum_t = tgt.sum()
    dice = float((2.0 * inter + smooth) / (sum_p + sum_t + smooth))
    union = (ph + tgt - ph * tgt).sum()
    iou  = float((inter + smooth) / (union + smooth))
    return dice, iou, ph

# ----------------- Overlay helpers (no alpha channel) -----------------
def draw_contours(gray_u8, pred01, gt01):
    """Return BGR image with GREEN pred outline, RED GT outline."""
    img = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    pred = (pred01.astype(np.uint8))
    gt   = (gt01.astype(np.uint8))

    cnt_p,_ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_g,_ = cv2.findContours(gt,   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = img.copy()
    # draw GT first (RED), then Pred (GREEN) on top
    cv2.drawContours(out, cnt_g, -1, (0, 0, 255), 2)
    cv2.drawContours(out, cnt_p, -1, (0, 255, 0), 2)
    return out

def fill_masks(gray_u8, pred01, gt01, alpha=0.35):
    """Semi-transparent fill: GT=RED, Pred=GREEN. No alpha channel in output."""
    img = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
    overlay = np.zeros_like(img, dtype=np.float32)
    # GT red
    overlay[gt01 == 1]   = (0, 0, 255)
    # Pred green
    overlay[pred01 == 1] = (0, 255, 0)
    out = cv2.addWeighted(img, 1.0, overlay, alpha, 0.0)
    return out.astype(np.uint8)

def save_pred_mask(stem, pred01, out_dir):
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    pm = (pred01.astype(np.uint8)) * 255
    cv2.imwrite(os.path.join(out_dir, "masks", f"{stem}_pred.png"), pm)

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt",      required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--img_size",  type=int, default=256)
    ap.add_argument("--base",      type=int, default=16)
    ap.add_argument("--thr",       type=float, default=0.5)
    ap.add_argument("--overlay",   choices=["contour","mask"], default="contour")
    ap.add_argument("--alpha",     type=float, default=0.35)   # used for mask mode
    a = ap.parse_args()

    ds = SliceDataset(a.data_root, "test", a.img_size)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(1, 1, base=a.base).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "overlays"), exist_ok=True)

    dices, ious = [], []
    with open(os.path.join(a.out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id","dice","iou"])
        for stem, img_vis, img_t, gt_t in tqdm(ld, desc="test"):
            img_t, gt_t = img_t.to(dev), gt_t.to(dev)
            out = model(img_t)
            d, i, ph = dice_iou_smoothed(out, gt_t, thr=a.thr, smooth=1.0)
            dices.append(d); ious.append(i)
            w.writerow([stem[0], f"{d:.6f}", f"{i:.6f}"])

            # save predicted mask + overlay
            pred01 = ph[0,0].cpu().numpy().astype(np.uint8)
            gt01   = gt_t[0,0].cpu().numpy().astype(np.uint8)
            save_pred_mask(stem[0], pred01, a.out_dir)

            if a.overlay == "contour":
                vis = draw_contours(img_vis[0].numpy().astype(np.uint8), pred01, gt01)
            else:
                vis = fill_masks(img_vis[0].numpy().astype(np.uint8), pred01, gt01, alpha=a.alpha)

            cv2.imwrite(os.path.join(a.out_dir, "overlays", f"{stem[0]}_overlay.png"), vis)

    d_mean = float(np.mean(dices)) if len(dices) else 0.0
    i_mean = float(np.mean(ious))  if len(ious)  else 0.0
    print(f"Test Dice: {d_mean:.6f} | IoU: {i_mean:.6f}")

if __name__ == "__main__":
    import numpy as np
    main()

