from pathlib import Path
import csv, shutil

ROOT = Path("Processed_Data/Unet")
MODEL = ROOT / "model"
OVERLAYS = MODEL / "overlays"
PRED_MASKS = MODEL / "masks"
GT_IMAGES = ROOT / "test" / "images"
GT_MASKS  = ROOT / "test" / "masks"
OUT = MODEL / "selection"

def reset_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

reset_dir(OUT / "best")
reset_dir(OUT / "typical")
reset_dir(OUT / "worst")

rows = []
with open(MODEL / "metrics.csv", newline="") as f:
    for r in csv.DictReader(f):
        rows.append({"id": r["id"], "dice": float(r["dice"])})

rows.sort(key=lambda x: x["dice"])
n = len(rows)
worst = rows[:3]
best  = rows[-3:]
mid = n // 2
typical = [rows[max(0, mid-2)], rows[max(0, mid-1)],
           rows[min(n-1, mid)], rows[min(n-1, mid+1)]]

def copy_case(stem: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for src in [
        OVERLAYS / f"{stem}_overlay.png",
        PRED_MASKS / f"{stem}_pred.png",
        GT_IMAGES / f"{stem}.png",
        GT_MASKS  / f"{stem}_mask.png",
    ]:
        if src.exists():
            shutil.copy2(src, dest / src.name)

for name, group in [("best", best), ("typical", typical), ("worst", worst)]:
    for item in group:
        copy_case(item["id"], OUT / name)

print(f"Picked 3 best, 4 typical, 3 worst from {n} cases → {OUT}")
