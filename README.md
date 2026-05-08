# Brain Tumour Segmentation
### Comparing U-Net and Segment Anything Model (SAM) on BraTS2020

A comparative study of task-specific vs. foundation model approaches to brain tumour segmentation on 2D MRI slices. U-Net is trained from scratch on the BraTS2020 dataset; SAM (ViT-B) is evaluated zero-shot and with limited fine-tuning. Results show that domain-specific architecture outperforms scale under conditions of extreme label scarcity.

**Dataset:** BraTS2020 · **Modality:** MRI · **Task:** Segmentation

## Team 
Simranjeet Kaur Brar\
Dhwanan Mirani\
Utsav Patel\
Chen-Yu Lee\
Yizhang Zhu

---

## Results

| Model | Dice (%) | IoU (%) | Notes |
|---|---|---|---|
| U-Net | **13.84** | **7.19** | Trained on full BraTS2020 |
| SAM Fine-tuned | 8.20 | 4.26 | Fine-tuned on full BraTS2020 |
| SAM Zero-shot | 3.77 | 1.93 | No domain adaptation |

> "Scale does not substitute for domain specificity. U-Net's encoder-decoder inductive bias aligns with MRI regularities — SAM's does not."

---

## Project Structure

```
Tumor_Segmentation/
├── src/
│   ├── segmentation/          # SAM pipeline (loader, zero-shot, fine-tune, evaluate)
│   └── unet/                  # U-Net training, evaluation, data preparation
├── frontend/                  # React inference UI
├── scripts/
│   └── preview_overlay.py     # Visualise image–mask overlays
├── Processed_Data/            # Sample BraTS2020 slices (full dataset required for training)
├── output/
│   └── evaluation/            # Comparison images and metrics
├── models/                    # Model weights (not committed, see below)
├── backend.py                 # FastAPI inference server
├── requirements.txt
└── README.md
```
---

## Installation

**1. Clone the repo**
```bash
git clone https://github.com/ksimranjeetbrar/Tumor_Segmentation
cd Tumor_Segmentation
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

**3. Download the SAM checkpoint**
```bash
curl -o models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## Reproducing the Results

The sample data in `Processed_Data/` is sufficient to run the pipeline and verify the code works. For full training, download the complete BraTS2020 dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) and run the data preparation script first.

### Prepare data (full dataset only)
```bash
python src/unet/prepare_data.py
```

### Train U-Net
```bash
python src/unet/train_unet.py \
  --data_root Processed_Data \
  --out_dir Processed_Data/Unet/model \
  --epochs 50 \
  --base 32
```

### Evaluate U-Net
```bash
python src/unet/eval_unet.py \
  --data_root Processed_Data \
  --ckpt Processed_Data/Unet/model/best.pt \
  --out_dir output/unet \
  --base 32
```

### Run SAM Pipeline (zero-shot + fine-tune + evaluate)
```bash
python src/segmentation/pipeline.py
```

---

## Interactive Demo

The project includes a React frontend and FastAPI backend for running live inference using the trained U-Net model.

**Start the backend**
```bash
uvicorn backend:app --reload --port 8000
```

**Start the frontend**
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, upload any BraTS2020 MRI slice, and run segmentation.

---

## Key Finding

SAM's zero-shot performance (Dice 3.77%) confirms a significant domain gap between natural image pretraining and MRI modality. Fine-tuning partially closes this gap (Dice 8.20%) but fails to match U-Net (Dice 13.84%), which benefits from task-specific inductive biases — skip connections, encoder–decoder symmetry — that align with the structural regularities of medical images.

---

## Tech Stack

Python · PyTorch · Segment Anything (SAM ViT-B) · U-Net · FastAPI · React · BraTS2020 · OpenCV

---

## Course

CMPT 340 - Biomedical Computing · Simon Fraser University · 2025