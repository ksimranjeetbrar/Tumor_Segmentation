"""
Evaluation and Metrics for SAM Segmentation

Computes metrics (Dice, IoU) comparing:
- SAM zero-shot predictions
- SAM fine-tuned predictions
- Ground truth masks

Generates visualizations for report.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class SegmentationMetrics:
    """Compute segmentation evaluation metrics."""
    
    @staticmethod
    def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Dice Similarity Coefficient (F1 score).
        
        Range: [0, 1], higher is better
        Interpretation: Overlap between prediction and target
        """
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        intersection = np.logical_and(pred, target).sum()
        
        if pred.sum() + target.sum() == 0:
            return 1.0 if np.array_equal(pred, target) else 0.0
        
        dice = 2.0 * intersection / (pred.sum() + target.sum())
        return float(dice)
    
    @staticmethod
    def iou(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Intersection over Union (Jaccard Index).
        
        Range: [0, 1], higher is better
        Interpretation: Overlap relative to union
        """
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        
        if union == 0:
            return 1.0 if np.array_equal(pred, target) else 0.0
        
        iou = intersection / union
        return float(iou)
    
    @staticmethod
    def sensitivity(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Sensitivity (True Positive Rate / Recall).
        
        How much of the target did we catch?
        """
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        if target.sum() == 0:
            return 1.0
        
        tp = np.logical_and(pred, target).sum()
        return float(tp / target.sum())
    
    @staticmethod
    def specificity(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Specificity (True Negative Rate).
        
        How well do we identify non-tumor regions?
        """
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        # Inverse: non-tumor regions
        pred_neg = ~pred
        target_neg = ~target
        
        if target_neg.sum() == 0:
            return 1.0
        
        tn = np.logical_and(pred_neg, target_neg).sum()
        return float(tn / target_neg.sum())


class EvaluationAnalyzer:
    """Analyze SAM predictions against ground truth."""
    
    def __init__(self, gt_dir: Path, results_dir: Path):
        """
        Args:
            gt_dir: Directory with ground truth masks
            results_dir: Directory with predictions (subdirs: zero_shot, fine_tuned)
        """
        self.gt_dir = Path(gt_dir)
        self.results_dir = Path(results_dir)
        self.metrics = SegmentationMetrics()
    
    def evaluate_predictions(
        self,
        method_name: str,
        predictions_dir: Path
    ) -> Dict:
        """
        Evaluate predictions from a specific method.
        
        Args:
            method_name: Name of method (e.g., 'zero_shot', 'fine_tuned')
            predictions_dir: Directory containing predicted masks
        
        Returns:
            Dictionary with per-image and aggregate metrics
        """
        results = {
            'method': method_name,
            'per_image': {},
            'aggregate': {}
        }
        
        pred_files = sorted(Path(predictions_dir).glob("*_mask.png"))
        
        if len(pred_files) == 0:
            print(f"No predictions found in {predictions_dir}")
            return results
        
        all_dice = []
        all_iou = []
        all_sens = []
        all_spec = []
        
        for pred_path in pred_files:
            # Find corresponding ground truth
            base_name = pred_path.stem.replace('_mask', '')
            gt_path = self.gt_dir / f"{base_name}_mask.png"
            
            if not gt_path.exists():
                print(f"Warning: Ground truth not found for {base_name}")
                continue
            
            # Load masks
            pred_mask = np.array(Image.open(pred_path).convert('L')) > 127
            gt_mask = np.array(Image.open(gt_path).convert('L')) > 127
            
            # Compute metrics
            dice = self.metrics.dice_coefficient(pred_mask, gt_mask)
            iou = self.metrics.iou(pred_mask, gt_mask)
            sens = self.metrics.sensitivity(pred_mask, gt_mask)
            spec = self.metrics.specificity(pred_mask, gt_mask)
            
            results['per_image'][base_name] = {
                'dice': dice,
                'iou': iou,
                'sensitivity': sens,
                'specificity': spec
            }
            
            all_dice.append(dice)
            all_iou.append(iou)
            all_sens.append(sens)
            all_spec.append(spec)
        
        # Aggregate metrics
        if all_dice:
            results['aggregate'] = {
                'dice_mean': np.mean(all_dice),
                'dice_std': np.std(all_dice),
                'dice_min': np.min(all_dice),
                'dice_max': np.max(all_dice),
                
                'iou_mean': np.mean(all_iou),
                'iou_std': np.std(all_iou),
                'iou_min': np.min(all_iou),
                'iou_max': np.max(all_iou),
                
                'sensitivity_mean': np.mean(all_sens),
                'sensitivity_std': np.std(all_sens),
                
                'specificity_mean': np.mean(all_spec),
                'specificity_std': np.std(all_spec),
                
                'num_images': len(all_dice)
            }
        
        return results
    
    def generate_comparison_images(
        self,
        image_dir: Path,
        gt_dir: Path,
        pred_dirs: Dict[str, Path],
        output_dir: Path,
        num_samples: int = 5
    ):
        """
        Generate comparison visualizations.
        
        Args:
            image_dir: Directory with original MRI images
            gt_dir: Directory with ground truth masks
            pred_dirs: Dict mapping method names to prediction directories
            output_dir: Where to save visualizations
            num_samples: Number of sample images to visualize
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = sorted(Path(image_dir).glob("*.png"))[:num_samples]
        
        for idx, img_path in enumerate(image_files):
            base_name = img_path.stem
            
            # Load image and ground truth
            image = np.array(Image.open(img_path).convert('RGB'))
            gt_mask = np.array(Image.open(gt_dir / f"{base_name}_mask.png").convert('L')) > 127
            
            # Create figure
            num_methods = len(pred_dirs)
            fig = plt.figure(figsize=(4 * (num_methods + 2), 4))
            gs = GridSpec(1, num_methods + 2, figure=fig)
            
            # Original image
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(image, cmap='gray')
            ax0.set_title('Original MRI')
            ax0.axis('off')
            
            # Ground truth
            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(image, cmap='gray')
            ax1.imshow(gt_mask, cmap='Reds', alpha=0.5)
            ax1.set_title('Ground Truth')
            ax1.axis('off')
            
            # Predictions
            for col, (method_name, pred_dir) in enumerate(pred_dirs.items()):
                pred_path = pred_dir / f"{base_name}_mask.png"
                
                if pred_path.exists():
                    pred_mask = np.array(Image.open(pred_path).convert('L')) > 127
                    
                    # Compute Dice
                    dice = self.metrics.dice_coefficient(pred_mask, gt_mask)
                    
                    ax = fig.add_subplot(gs[0, col + 2])
                    ax.imshow(image, cmap='gray')
                    ax.imshow(pred_mask, cmap='Blues', alpha=0.5)
                    ax.set_title(f'{method_name}\nDice: {dice:.3f}')
                    ax.axis('off')
            
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / f"comparison_{idx:02d}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved: {output_path}")


def main():
    """Run comprehensive evaluation."""
    
    project_root = Path(__file__).parent.parent.parent
    gt_dir = project_root / "Processed_Data" / "val" / "masks"
    zero_shot_dir = project_root / "output" / "sam_zero_shot"
    fine_tuned_dir = project_root / "output" / "sam_fine_tuned_inference"
    images_dir = project_root / "Processed_Data" / "val" / "images"
    eval_output_dir = project_root / "output" / "evaluation"
    
    print("=" * 70)
    print("SAM EVALUATION")
    print("=" * 70)
    print()
    
    analyzer = EvaluationAnalyzer(gt_dir, project_root / "output")
    
    # Evaluate methods
    print("Evaluating zero-shot predictions...")
    zero_shot_results = analyzer.evaluate_predictions("SAM Zero-Shot", zero_shot_dir)
    
    print("Evaluating fine-tuned predictions...")
    fine_tuned_results = analyzer.evaluate_predictions("SAM Fine-Tuned", fine_tuned_dir)
    
    # Print results
    print()
    print("=" * 70)
    print("RESULTS - DICE COEFFICIENT")
    print("=" * 70)
    
    for results in [zero_shot_results, fine_tuned_results]:
        method = results['method']
        agg = results['aggregate']
        
        if agg:
            print(f"\n{method}:")
            print(f"  Mean:   {agg['dice_mean']:.4f} ± {agg['dice_std']:.4f}")
            print(f"  Range:  [{agg['dice_min']:.4f}, {agg['dice_max']:.4f}]")
            print(f"  Images: {agg['num_images']}")
    
    # Save results
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    results_file = eval_output_dir / "metrics.json"
    with open(results_file, 'w') as f:
        json.dump({
            'zero_shot': zero_shot_results,
            'fine_tuned': fine_tuned_results
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    # Generate comparisons
    print("\nGenerating comparison images...")
    analyzer.generate_comparison_images(
        images_dir,
        gt_dir,
        {
            'Zero-Shot': zero_shot_dir,
            'Fine-Tuned': fine_tuned_dir
        },
        eval_output_dir / "comparisons"
    )


if __name__ == "__main__":
    main()
