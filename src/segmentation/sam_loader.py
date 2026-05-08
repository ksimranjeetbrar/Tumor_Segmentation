"""
SAM (Segment Anything Model) loader and inference utilities.
Provides easy interface to load and use the pretrained SAM model for segmentation.
"""

import os
from pathlib import Path
import numpy as np
import torch
from typing import Tuple, Optional, Union

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    raise ImportError("segment-anything package not found. Install with: pip install segment-anything")


class SAMModel:
    """
    Wrapper class for SAM (Segment Anything Model) inference.
    
    Attributes:
        model_type (str): Type of SAM model (e.g., 'vit_h', 'vit_l', 'vit_b')
        device (str): Device to run model on ('cuda' or 'cpu')
        model: The loaded SAM model
        predictor: SAM predictor for inference
    """
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        model_type: str = "vit_b",
        device: Optional[str] = None,
        freeze_encoder: bool = False
    ):
        """
        Initialize SAM model.
        
        Args:
            checkpoint_path: Path to SAM checkpoint file (.pth)
            model_type: Type of SAM model ('vit_h', 'vit_l', 'vit_b'). Default: 'vit_b' for 4GB GPU.
            device: Device to use ('cuda', 'cpu'). Auto-detects if None.
            freeze_encoder: If True, freeze image encoder to save memory (~70% savings).
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type
        
        # Auto-detect device: try CUDA first, fallback to CPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Verify checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {self.checkpoint_path}")
        
        print(f"Loading SAM model ({model_type}) from {self.checkpoint_path}")
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load model
        self.model = sam_model_registry[model_type](checkpoint=str(self.checkpoint_path))
        self.model.to(device=self.device)
        
        # Optionally freeze encoder (saves 70% memory for fine-tuning)
        if freeze_encoder:
            self._freeze_encoder()
        
        # Create predictor
        self.predictor = SamPredictor(self.model)
    
    def _freeze_encoder(self):
        """Freeze image encoder to save memory during fine-tuning."""
        if hasattr(self.model, 'image_encoder'):
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
            print("✓ Image encoder frozen (saves ~70% memory)")
    
    def segment_image(
        self,
        image: np.ndarray,
        points: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment an image using SAM.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            points: Point prompts (N, 2) in (x, y) format
            boxes: Bounding box prompts (N, 4) in (x1, y1, x2, y2) format
            labels: Labels for points (N,) - 1 for foreground, 0 for background
        
        Returns:
            Tuple of (masks, scores, logits):
                - masks: Predicted masks (N, H, W)
                - scores: Confidence scores for each mask (N,)
                - logits: Raw model logits (N, H, W)
        """
        # Set image for the predictor
        self.predictor.set_image(image)
        
        # Prepare input prompts
        point_coords = points if points is not None else None
        point_labels = labels if labels is not None else None
        box = boxes[0] if boxes is not None and len(boxes) > 0 else None
        
        # Run prediction
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True
        )
        
        return masks, scores, logits
    
    def segment_with_points(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment image using point prompts.
        
        Args:
            image: Input image (H, W, 3) in RGB
            points: Point coordinates (N, 2) in (x, y) format
            labels: Point labels - 1 for foreground, 0 for background (default: all 1)
        
        Returns:
            Tuple of (masks, scores, logits)
        """
        if labels is None:
            labels = np.ones(len(points), dtype=np.int32)
        
        return self.segment_image(image, points=points, labels=labels)
    
    def segment_with_box(
        self,
        image: np.ndarray,
        box: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment image using bounding box prompt.
        
        Args:
            image: Input image (H, W, 3) in RGB
            box: Bounding box (4,) in (x1, y1, x2, y2) format
        
        Returns:
            Tuple of (masks, scores, logits)
        """
        return self.segment_image(image, boxes=np.array([box]))
    
    def get_best_mask(
        self,
        image: np.ndarray,
        points: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get the best mask prediction (highest confidence).
        
        Args:
            image: Input image (H, W, 3)
            points: Point prompts
            boxes: Box prompts
            labels: Point labels
        
        Returns:
            Best mask (H, W) as boolean array
        """
        masks, scores, _ = self.segment_image(image, points=points, boxes=boxes, labels=labels)
        best_idx = np.argmax(scores)
        return masks[best_idx].astype(bool)


def load_sam_model(
    checkpoint_path: Union[str, Path] = None,
    model_type: str = "vit_b",
    device: Optional[str] = None,
    freeze_encoder: bool = False
) -> SAMModel:
    """
    Load a pretrained SAM model.
    
    ⭐ RECOMMENDED: Use model_type='vit_b' for fine-tuning on 4GB GPU.
    
    Args:
        checkpoint_path: Path to checkpoint. If None, looks in 'models/' directory.
        model_type: Type of model ('vit_h', 'vit_l', 'vit_b'). Default: 'vit_b' (memory-efficient).
        device: Device to use ('cuda', 'cpu'). Auto-detects if None.
        freeze_encoder: If True, freeze encoder for fine-tuning (saves 70% memory).
    
    Returns:
        SAMModel instance
    
    Available checkpoints:
        - vit_h: sam_vit_h_4b8939.pth (2.6 GB, requires 8+ GB VRAM) - NOT RECOMMENDED for 4GB GPU
        - vit_b: sam_vit_b_01ec64.pth (375 MB, fits in 4 GB VRAM) - RECOMMENDED
    """
    if checkpoint_path is None:
        # Get absolute path relative to this file
        current_file = Path(__file__).resolve()
        
        # Build list of checkpoint filenames to try
        checkpoint_names = []
        if model_type == "vit_b":
            checkpoint_names.append("sam_vit_b_01ec64.pth")  # ViT-B (375 MB, recommended)
        else:
            checkpoint_names.append(f"sam_{model_type}_4b8939.pth")  # ViT-H or ViT-L
        
        # Try common locations (from various working directories)
        search_paths = []
        for name in checkpoint_names:
            search_paths.extend([
                # Relative to current working directory
                Path.cwd() / "models" / name,
                Path.cwd().parent / "models" / name,
                
                # Relative to this file (src/amazing/)
                current_file.parent.parent.parent / "models" / name,  # project_root/models
            ])
        
        for path in search_paths:
            if path.exists():
                checkpoint_path = path.resolve()
                print(f"Found SAM checkpoint: {checkpoint_path}")
                break
        
        if checkpoint_path is None:
            print(f"Debug info - Current working directory: {Path.cwd()}")
            print(f"Debug info - This file location: {current_file}")
            print(f"Debug info - Searched paths:")
            for p in search_paths:
                print(f"  - {p}")
            raise FileNotFoundError(
                f"SAM checkpoint not found. For 4GB GPU, download ViT-B: "
                f"curl -o models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            )
    
    return SAMModel(checkpoint_path, model_type=model_type, device=device, freeze_encoder=freeze_encoder)
