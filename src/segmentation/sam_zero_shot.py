"""
SAM Zero-Shot Inference on Brain MRI

Runs pretrained SAM (trained on natural images) directly on MRI slices
WITHOUT any fine-tuning to establish baseline performance.

This represents "out-of-the-box" capability of foundation models on medical imaging.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from typing import Tuple, Dict, List
import time

from .sam_loader import load_sam_model


class SAMZeroShotSegmenter:
    """
    Zero-shot SAM segmentation on brain MRI.
    Uses pretrained model without any domain-specific training.
    
    ⭐ Recomm ended: Uses SAM ViT-B (375 MB) by default for 4GB GPU compatibility.
    """
    
    def __init__(self, checkpoint_path: str = None, device: str = None, model_type: str = "vit_b"):
        """
        Initialize pretrained SAM model.
        
        Args:
            checkpoint_path: Path to SAM checkpoint
            device: Device to use ('cuda', 'cpu')
            model_type: Model architecture ('vit_b', 'vit_l', 'vit_h'). Default: 'vit_b' for 4GB GPU.
        """
        self.model = load_sam_model(
            checkpoint_path=checkpoint_path, 
            device=device,
            model_type=model_type
        )
        self.device = self.model.device
        
    def segment_mri_slice(
        self,
        image: np.ndarray,
        gt_mask: np.ndarray = None,
        strategy: str = "gt_centroid"
    ) -> Tuple[np.ndarray, float]:
        """
        Segment an MRI slice using zero-shot SAM.
        
        Args:
            image: MRI slice (H, W, 3) - should be RGB or grayscale converted to RGB
            gt_mask: Optional ground truth mask to extract tumor location
            strategy: Prompt strategy - 'gt_centroid', 'center_point', 'multi_point', 'auto_grid'
        
        Returns:
            Tuple of (best_mask, confidence_score)
        """
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        
        if strategy == "gt_centroid" and gt_mask is not None:
            # Use ground truth tumor centroid as prompt (fair test for SAM capability)
            gt_binary = gt_mask > 127
            if gt_binary.sum() > 0:  # Only if mask has content
                y_coords, x_coords = np.where(gt_binary)
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                point = np.array([[center_x, center_y]])
            else:
                # Fallback to center if no tumor found
                h, w = image.shape[:2]
                point = np.array([[w // 2, h // 2]])
            
            masks, scores, _ = self.model.segment_with_points(image, point)
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx].astype(bool)
            best_score = float(scores[best_idx])
        
        elif strategy == "center_point":
            # Simple: use center point as prompt
            h, w = image.shape[:2]
            center_point = np.array([[w // 2, h // 2]])
            
            masks, scores, _ = self.model.segment_with_points(image, center_point)
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx].astype(bool)
            best_score = float(scores[best_idx])
            
        elif strategy == "multi_point":
            # Multiple points in a grid pattern
            h, w = image.shape[:2]
            points = np.array([
                [w // 2, h // 2],      # center
                [w // 4, h // 4],      # top-left quadrant
                [3 * w // 4, h // 4],  # top-right quadrant
                [w // 4, 3 * h // 4],  # bottom-left quadrant
                [3 * w // 4, 3 * h // 4]  # bottom-right quadrant
            ])
            labels = np.ones(len(points), dtype=np.int32)
            
            masks, scores, _ = self.model.segment_with_points(image, points, labels)
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx].astype(bool)
            best_score = float(scores[best_idx])
            
        elif strategy == "auto_grid":
            # Grid of points across image
            h, w = image.shape[:2]
            grid_size = 3
            points = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    x = (j + 1) * w // (grid_size + 1)
                    y = (i + 1) * h // (grid_size + 1)
                    points.append([x, y])
            
            points = np.array(points)
            labels = np.ones(len(points), dtype=np.int32)
            
            masks, scores, _ = self.model.segment_with_points(image, points, labels)
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx].astype(bool)
            best_score = float(scores[best_idx])
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return best_mask, best_score
    
    def batch_segment(
        self,
        image_dir: Path,
        masks_dir: Path = None,
        strategy: str = "gt_centroid",
        save_dir: Path = None
    ) -> Dict[str, Dict]:
        """
        Segment multiple MRI slices.
        
        Args:
            image_dir: Directory containing MRI images
            masks_dir: Optional directory containing ground truth masks (for gt_centroid strategy)
            strategy: Prompt strategy to use (default: gt_centroid for fair evaluation)
            save_dir: Optional directory to save masks
        
        Returns:
            Dictionary with results for each image
        """
        results = {}
        image_files = sorted(Path(image_dir).glob("*.png"))
        
        print(f"Zero-shot segmentation on {len(image_files)} images using '{strategy}' strategy...")
        
        for idx, img_path in enumerate(image_files):
            try:
                # Load image
                image = np.array(Image.open(img_path).convert('RGB'))
                
                # Load ground truth mask if available (for gt_centroid strategy)
                gt_mask = None
                if masks_dir and strategy == "gt_centroid":
                    mask_path = Path(masks_dir) / f"{img_path.stem}_mask.png"
                    if mask_path.exists():
                        gt_mask = np.array(Image.open(mask_path))
                
                # Segment
                start_time = time.time()
                mask, score = self.segment_mri_slice(image, gt_mask=gt_mask, strategy=strategy)
                elapsed = time.time() - start_time
                
                results[img_path.name] = {
                    'mask': mask,
                    'confidence': score,
                    'time': elapsed,
                    'image_shape': image.shape
                }
                
                # Save mask if requested
                if save_dir:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    mask_path = save_dir / f"{img_path.stem}_mask.png"
                    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                
                print(f"  [{idx+1}/{len(image_files)}] {img_path.name} "
                      f"(conf: {score:.3f}, time: {elapsed:.2f}s)")
                
            except Exception as e:
                print(f"  ERROR on {img_path.name}: {e}")
                results[img_path.name] = {'error': str(e)}
        
        return results


def main():
    """Run zero-shot SAM segmentation on test dataset."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    test_images_dir = project_root / "Processed_Data" / "val" / "images"
    output_dir = project_root / "output" / "sam_zero_shot"
    
    print("=" * 70)
    print("SAM ZERO-SHOT SEGMENTATION")
    print("=" * 70)
    print(f"Test images: {test_images_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Initialize segmenter
    segmenter = SAMZeroShotSegmenter()
    
    # Run segmentation with center point strategy
    results = segmenter.batch_segment(
        test_images_dir,
        strategy="center_point",
        save_dir=output_dir
    )
    
    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results.values() if 'mask' in r)
    failed = len(results) - successful
    
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if successful > 0:
        times = [r['time'] for r in results.values() if 'time' in r]
        confs = [r['confidence'] for r in results.values() if 'confidence' in r]
        
        print(f"Average time per image: {np.mean(times):.2f}s")
        print(f"Average confidence: {np.mean(confs):.3f}")
        print(f"Confidence range: [{np.min(confs):.3f}, {np.max(confs):.3f}]")
    
    print()
    print(f"Masks saved to: {output_dir}")


if __name__ == "__main__":
    main()
