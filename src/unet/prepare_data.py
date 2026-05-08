import random, csv, re
from pathlib import Path
import nibabel as nib
import numpy as np
import cv2

# Config
BASE_DIR = Path(r"")

DATA_FOLDERS = [
    "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
    "BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData",
]

IMG_SIZE = 256
PLANE = "axial" 
USE_MODALITY = "flair"
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
SEED = 42
MAX_PER_SPLIT = {"train":500, "val": 100, "test": 5000}

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "Processed_Data" / "Unet"

def find_subject_dirs(base_dir: Path):
    subjects = []
    for rel in DATA_FOLDERS:
        root = base_dir / rel
        if not root.exists():
            raise FileNotFoundError(f"Data folder not found: {root}")
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            has_seg = list(sub.glob("*_seg.nii")) + list(sub.glob("*_seg.nii.gz"))
            if has_seg:
                subjects.append(sub)
    subjects = sorted(subjects)
    print(f"Found {len(subjects)} subjects under {base_dir}.")
    return subjects

def pick_paths(subdir: Path, modality: str):
    patt_img = re.compile(fr".*_{modality}\.nii(\.gz)?$")
    patt_seg = re.compile(r".*_seg\.nii(\.gz)?$")
    img = seg = None
    for f in subdir.iterdir():
        s = str(f)
        if patt_img.match(s):
            img = f
        if patt_seg.match(s):
            seg = f
    if img is None or seg is None:
        raise FileNotFoundError(f"Missing {modality} or seg in {subdir}")
    return img, seg

def vol_to_slices(vol, plane: str):
    if plane == "axial":
        return [vol[:, :, k] for k in range(vol.shape[2])]
    elif plane == "sagittal":
        return [vol[k, :, :] for k in range(vol.shape[0])]
    elif plane == "coronal":
        return [vol[:, k, :] for k in range(vol.shape[1])]
    else:
        raise ValueError("PLANE must be axial/sagittal/coronal")

def norm_uint8(img2d):
    x = img2d.astype(np.float32)
    m, M = np.min(x), np.max(x)
    if M <= m:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - m) / (M - m)
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    return x

def resize2d(arr, size=256, is_mask=False):
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(arr, (size, size), interpolation=interp)

def make_out_dirs(root: Path):
    for split in ["train", "val", "test"]:
        for sub in ["images", "masks"]:
            (root / split / sub).mkdir(parents=True, exist_ok=True)

def main():
    random.seed(SEED)

    subjects = find_subject_dirs(BASE_DIR)
    random.shuffle(subjects)
    n = len(subjects)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_subs = set(subjects[:n_train])
    val_subs   = set(subjects[n_train:n_train + n_val])
    test_subs  = set(subjects[n_train + n_val:])

    print(f"Split -> train: {len(train_subs)}, val: {len(val_subs)}, test: {len(test_subs)}")

    make_out_dirs(OUTPUT_ROOT)
    summary_path = OUTPUT_ROOT / "data_summary.csv"
    with open(summary_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["subject","source_set","split","kept_slices","plane","modality"])

        kept_total = {"train": 0, "val": 0, "test": 0}

        for subdir in subjects:
            if subdir in train_subs:
                split = "train"
            elif subdir in val_subs:
                split = "val"
            else:
                split = "test"

            if kept_total[split] >= MAX_PER_SPLIT[split]:
                continue

            source = "val_original" if "Validation" in str(subdir) else "train_original"
            img_path, seg_path = pick_paths(subdir, USE_MODALITY)
            img_vol = nib.load(str(img_path)).get_fdata()
            seg_vol = nib.load(str(seg_path)).get_fdata()
            img_slices = vol_to_slices(img_vol, PLANE)
            seg_slices = vol_to_slices(seg_vol, PLANE)

            subject_id = subdir.name
            kept = 0

            for k, (im, msk) in enumerate(zip(img_slices, seg_slices)):
                if kept_total[split] >= MAX_PER_SPLIT[split]:
                    break
                if np.sum(msk) == 0:
                    continue

                im8  = resize2d(norm_uint8(im), IMG_SIZE, is_mask=False)
                msk8 = resize2d(msk.astype(np.uint8), IMG_SIZE, is_mask=True)
                msk_vis = (msk8 > 0).astype("uint8") * 255

                img_name  = f"{subject_id}_{PLANE}_{k:03d}.png"
                mask_name = f"{subject_id}_{PLANE}_{k:03d}_mask.png"
                cv2.imwrite(str(OUTPUT_ROOT / split / "images" / img_name), im8)
                cv2.imwrite(str(OUTPUT_ROOT / split / "masks"  / mask_name), msk_vis)

                kept += 1
                kept_total[split] += 1

            print(f"{subject_id}: kept {kept} slices -> {split} ({source})")
            writer.writerow([subject_id, source, split, kept, PLANE, USE_MODALITY])

    print(f"Totals per split: {kept_total}.")
    print(f"Done. PNGs + splits in {OUTPUT_ROOT}. Summary: {summary_path}")

if __name__ == "__main__":
    main()
