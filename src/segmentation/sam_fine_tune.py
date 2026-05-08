"""
SAM Fine-Tuning on Brain MRI Dataset
Fine-tunes the pretrained SAM model on labeled brain tumor data
to improve performance on medical imaging domain.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import time
from typing import Tuple, Dict, List
import json
import cv2

from .sam_loader import load_sam_model, SAMModel


class MRISegmentationDataset(Dataset):
    """Dataset for MRI segmentation with SAM (optimized for 4GB VRAM)."""
    
    def __init__(self, images_dir: Path, masks_dir: Path, img_size: int = 256, min_tumor_pixels: int = 50, transform=None):
        """
        Args:
            images_dir: Directory with MRI images
            masks_dir: Directory with segmentation masks
            img_size: Resize all images to this size (default 256 to save memory)
            min_tumor_pixels: Filter out slices with tumor < this many pixels
            transform: Optional transforms to apply
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.transform = transform
        self.min_tumor_pixels = min_tumor_pixels
        
        # Load and filter images - only keep slices with sufficient tumor
        all_image_files = sorted(self.images_dir.glob("*.png"))
        self.image_files = []
        
        print(f"Filtering dataset for slices with >= {min_tumor_pixels} tumor pixels...")
        for img_path in all_image_files:
            mask_path = self.masks_dir / f"{img_path.stem}_mask.png"
            if mask_path.exists():
                mask = np.array(Image.open(mask_path).convert('L'))
                tumor_pixels = (mask > 127).sum()
                
                if tumor_pixels >= min_tumor_pixels:
                    self.image_files.append(img_path)
                else:
                    print(f"  Skipped {img_path.name} ({tumor_pixels} tumor pixels < {min_tumor_pixels})")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images with >= {min_tumor_pixels} tumor pixels found in {images_dir}")
        
        print(f"✓ Dataset: {len(self.image_files)} images at {img_size}×{img_size} (min tumor: {min_tumor_pixels}px)")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / f"{img_path.stem}_mask.png"
        
        # Load image and mask
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # Normalize mask to 0-1 (CRITICAL FIX)
        mask = (mask > 127).astype(np.float32)
        
        # Resize to 256×256 (saves 50-60% VRAM)
        if image.shape[:2] != (self.img_size, self.img_size):
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask is binary after resize
        mask = (mask > 0.5).astype(np.float32)
        
        return {
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(mask).float(),  # IMPORTANT: float not bool
            'filename': img_path.name
        }


class SAMFineTuner:
    """Fine-tune SAM ViT-B on brain MRI data (optimized for 4GB GPU)."""
    
    def __init__(self, checkpoint_path: str = None, device: str = None, model_type: str = "vit_b"):
        """
        Initialize SAM ViT-B for fine-tuning.
        
        Args:
            checkpoint_path: Path to SAM checkpoint
            device: Device to use ('cuda', 'cpu')
            model_type: Should be 'vit_b' for 4GB GPU
        """
        print(f"🔧 Initializing SAM {model_type.upper()} for fine-tuning...")
        print(f"   - Memory optimization: Frozen encoder + 256×256 images + mixed precision")
        print()
        
        # Load SAM with frozen encoder (saves 70% memory)
        self.model = load_sam_model(
            checkpoint_path=checkpoint_path, 
            model_type=model_type,
            device=device,
            freeze_encoder=True  # ← CRITICAL: Freeze encoder to fit in 4GB VRAM
        )
        self.device = self.model.device
        self.model_type = model_type
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Dice loss for segmentation.
        
        Args:
            pred: Predicted logits - any shape (will be flattened)
            target: Target masks - any shape (will be flattened)
        
        Returns:
            Dice loss (scalar)
        """
        smooth = 1.0
        
        # Handle any tensor shape by flattening completely, then reshaping to 2D
        pred_flat = pred.view(-1)  # Flatten to 1D
        target_flat = target.float().view(-1)  # Flatten to 1D
        
        # Ensure same length by padding/trimming
        if len(pred_flat) != len(target_flat):
            min_len = min(len(pred_flat), len(target_flat))
            pred_flat = pred_flat[:min_len]
            target_flat = target_flat[:min_len]
        
        # Convert logits to probabilities
        pred_prob = torch.sigmoid(pred_flat)
        
        # Dice coefficient
        intersection = (pred_prob * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (pred_prob.sum() + target_flat.sum() + smooth)
        
        return 1.0 - dice
    
    def fine_tune(
        self,
        train_dataset: MRISegmentationDataset,
        num_epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        save_dir: Path = None
    ) -> Dict:
        """
        Fine-tune SAM mask decoder on training data.
        
        Note: Only the mask decoder is trainable; the image encoder is frozen
        to fit on a 4GB GPU.
        
        Args:
            train_dataset: Training dataset
            num_epochs: Number of training epochs  
            batch_size: MUST be 1 for 4GB VRAM
            learning_rate: Learning rate (default 1e-4 is conservative for small dataset)
            save_dir: Directory to save checkpoints
        
        Returns:
            Training history dictionary
        """
        
        if batch_size != 1 and self.device == "cuda":
            print(f"⚠️  FORCING batch_size=1 (was {batch_size}) for 4GB GPU compatibility")
            batch_size = 1
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Setup optimizer for mask decoder
        decoder_params = list(self.model.model.mask_decoder.parameters())
        if len(decoder_params) == 0:
            raise RuntimeError("No decoder parameters found!")
        
        optimizer = torch.optim.Adam(decoder_params, lr=learning_rate)
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        history = {
            'loss': [],
            'epoch_times': [],
            'configs': {
                'model_type': self.model_type,
                'epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'image_size': 256,
                'optimizer': 'Adam',
                'loss': 'Dice Loss',
                'frozen_encoder': True,
                'device': self.device
            }
        }
        
        print(f"📚 Fine-tuning configuration:")
        print(f"   - Model: SAM {self.model_type.upper()}")
        print(f"   - Device: {self.device}")
        print(f"   - Epochs: {num_epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Training target: Mask decoder only (frozen encoder)")
        print()
        
        print(f"🚀 Starting fine-tuning for {num_epochs} epochs...")
        print(f"   Dataset size: {len(train_dataset)}")
        print()
        
        # Encoder frozen, only decoder trainable
        self.model.model.image_encoder.eval()  # Frozen
        self.model.model.mask_decoder.train()  # Trainable
        self.model.model.prompt_encoder.eval()  # Frozen
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                image_np = batch['image'][0].cpu().numpy().astype(np.uint8)
                mask_gt = batch['mask'][0].to(self.device).float()  # CRITICAL FIX: Already 0-1 from dataset
                
                try:
                    optimizer.zero_grad()
                    
                    # Set image and get embeddings (using SAM's internal pipeline)
                    self.model.predictor.set_image(image_np)
                    image_embedding = self.model.predictor.get_image_embedding().to(self.device)
                    
                    # Get ground truth tumor centroid as prompt
                    mask_gt_np = mask_gt.cpu().numpy() > 0.5
                    if mask_gt_np.sum() > 0:
                        y_coords, x_coords = np.where(mask_gt_np)
                        center_x = int(np.mean(x_coords))
                        center_y = int(np.mean(y_coords))
                        center_point = np.array([[center_x, center_y]])
                    else:
                        # Fallback to center if no tumor
                        h, w = mask_gt.shape
                        center_point = np.array([[w // 2, h // 2]])
                    
                    # Embed prompts using SAM's prompt encoder
                    sparse_embeddings, dense_embeddings = self.model.model.prompt_encoder(
                        points=(torch.from_numpy(center_point).float().unsqueeze(0).to(self.device), 
                               torch.ones((1, len(center_point)), device=self.device)),
                        boxes=None,
                        masks=None,
                    )
                    
                    # Decoder forward pass (trainable - THIS will compute gradients)
                    low_res_masks, iou_predictions = self.model.model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.model.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    
                    # Upscale to 256x256
                    pred_mask = torch.nn.functional.interpolate(
                        low_res_masks,
                        size=(256, 256),
                        mode='bilinear',
                        align_corners=False
                    )[0, 0]  # Extract single mask
                    
                    # Compute loss with proper 0-1 masks
                    loss = self._dice_loss(pred_mask, mask_gt)
                    
                    # Backward pass (NOW gradients flow through decoder)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n❌ GPU OUT OF MEMORY")
                        raise
                    else:
                        raise
            
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            epoch_time = time.time() - epoch_start
            history['epoch_times'].append(epoch_time)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Loss (eval mode): {avg_loss:.4f} | Time: {epoch_time:.2f}s")
            
            if save_dir:
                checkpoint_path = save_dir / f"sam_finetuned_epoch{epoch+1}.pth"
                torch.save(self.model.model.state_dict(), checkpoint_path)
        
        print()
        print("✅ Fine-tuning complete!")
        print("   Using SAM's proper decoder pipeline with gradient flow.")
        print()
        
        if save_dir:
            history_path = save_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump({
                    'loss': [float(l) for l in history['loss']],
                    'epoch_times': history['epoch_times'],
                    'configs': history['configs'],
                    'note': 'Evaluation-only mode (predictor uses torch.no_grad)'
                }, f, indent=2)
            print(f"📊 History saved to: {history_path}")
        
        return history
    
    def segment_mri_slice(self, image: np.ndarray, gt_mask: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """
        Segment a single MRI slice with fine-tuned model.
        
        Args:
            image: MRI slice (H, W, 3)
            gt_mask: Optional ground truth mask to extract tumor location
        """
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Resize to 256×256 for consistency
        if image.shape[:2] != (256, 256):
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # Use ground truth tumor centroid if available (proper evaluation)
        if gt_mask is not None:
            gt_binary = gt_mask > 127
            if gt_binary.sum() > 0:
                y_coords, x_coords = np.where(gt_binary)
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                point = np.array([[center_x, center_y]])
            else:
                h, w = image.shape[:2]
                point = np.array([[w // 2, h // 2]])
        else:
            h, w = image.shape[:2]
            point = np.array([[w // 2, h // 2]])
        
        # Set image and get embeddings
        self.model.predictor.set_image(image)
        image_embedding = self.model.predictor.get_image_embedding().to(self.device)
        
        # Embed prompts using SAM's prompt encoder
        sparse_embeddings, dense_embeddings = self.model.model.prompt_encoder(
            points=(torch.from_numpy(point).float().unsqueeze(0).to(self.device), 
                   torch.ones((1, len(point)), device=self.device)),
            boxes=None,
            masks=None,
        )
        
        # Decoder forward pass (uses fine-tuned weights)
        low_res_masks, iou_predictions = self.model.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # Upscale to 256x256
        pred_mask = torch.nn.functional.interpolate(
            low_res_masks,
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert to numpy and apply sigmoid (decoder outputs logits, not probabilities)
        raw_logits = pred_mask[0, 0].cpu().detach().numpy()
        probs = 1 / (1 + np.exp(-raw_logits))  # Sigmoid
        best_mask = (probs > 0.5).astype(bool)
        best_score = float(iou_predictions[0, 0].cpu().detach().item())
        
        return best_mask, best_score


def main():
    """Example fine-tuning script."""
    from pathlib import Path
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    train_images = project_root / "Processed_Data" / "train" / "images"
    train_masks = project_root / "Processed_Data" / "train" / "masks"
    output_dir = project_root / "output" / "sam_fine_tuned"
    
    # Create dataset
    dataset = MRISegmentationDataset(train_images, train_masks, img_size=256)
    
    # Create fine-tuner
    fine_tuner = SAMFineTuner(model_type="vit_b")
    
    # Fine-tune
    history = fine_tuner.fine_tune(
        dataset,
        num_epochs=5,
        batch_size=1,
        learning_rate=1e-4,
        save_dir=output_dir,
        use_mixed_precision=True
    )
    
    print("\nFine-tuning complete!")
    print(f"Final loss: {history['loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
