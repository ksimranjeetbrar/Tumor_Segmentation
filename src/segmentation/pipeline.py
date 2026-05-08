"""
Complete SAM Pipeline: Zero-Shot -> Fine-Tune -> Evaluate

This is your main script that runs the entire SAM workflow:
1. Zero-shot baseline on test data
2. Fine-tune on training data
3. Fine-tuned inference on test data
4. Evaluation against ground truth
5. Generate report artifacts
"""

import sys
from pathlib import Path
import argparse
import time
import json
import torch
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from .sam_zero_shot import SAMZeroShotSegmenter
from .sam_fine_tune import SAMFineTuner, MRISegmentationDataset
from .sam_evaluate import EvaluationAnalyzer


class SAMPipeline:
    """Complete SAM workflow."""
    
    def __init__(self, project_root: Path = None):
        """Initialize pipeline with project paths."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = project_root
        self.output_root = project_root / "output"
        self.data_root = project_root / "Processed_Data"
        
        # Create output directories
        self.zero_shot_dir = self.output_root / "sam_zero_shot"
        self.fine_tuned_dir = self.output_root / "sam_fine_tuned"
        self.fine_tuned_inference_dir = self.output_root / "sam_fine_tuned_inference"
        self.eval_dir = self.output_root / "evaluation"
        
        for d in [self.zero_shot_dir, self.fine_tuned_dir, self.fine_tuned_inference_dir, self.eval_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def run_zero_shot(self, test_split: str = "val") -> dict:
        """
        Stage 1: Zero-shot SAM on test data (no fine-tuning).
        
        Args:
            test_split: Which split to test on ('val' or 'test')
        
        Returns:
            Results dictionary
        """
        print("\n" + "=" * 80)
        print("STAGE 1: SAM ZERO-SHOT BASELINE")
        print("=" * 80)
        
        test_images_dir = self.data_root / test_split / "images"
        test_masks_dir = self.data_root / test_split / "masks"
        
        print(f"Input images: {test_images_dir}")
        print(f"Ground truth: {test_masks_dir}")
        print(f"Output masks: {self.zero_shot_dir}")
        print()
        
        start_time = time.time()
        
        segmenter = SAMZeroShotSegmenter(model_type="vit_b")
        results = segmenter.batch_segment(
            test_images_dir,
            masks_dir=test_masks_dir,
            strategy="gt_centroid",
            save_dir=self.zero_shot_dir
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Zero-shot completed in {elapsed:.1f}s")
        
        return {
            'stage': 'zero_shot',
            'time': elapsed,
            'results': results
        }
    
    def run_fine_tuning(self) -> dict:
        """
        Stage 2: Fine-tune SAM on training data.
        
        Returns:
            Training history
        """
        print("\n" + "=" * 80)
        print("STAGE 2: SAM FINE-TUNING")
        print("=" * 80)
        
        train_images_dir = self.data_root / "train" / "images"
        train_masks_dir = self.data_root / "train" / "masks"
        
        print(f"Training images: {train_images_dir}")
        print(f"Training masks: {train_masks_dir}")
        print(f"Checkpoints: {self.fine_tuned_dir}")
        print()
        
        dataset = MRISegmentationDataset(train_images_dir, train_masks_dir, img_size=256)
        print(f"Dataset loaded: {len(dataset)} image-mask pairs (resized to 256x256)\n")
        
        start_time = time.time()
        
        # Assumes ViT-B checkpoint is in project_root/models/sam_vit_b_01ec64.pth
        fine_tuner = SAMFineTuner(model_type="vit_b")
        history = fine_tuner.fine_tune(
            dataset,
            num_epochs=5,
            batch_size=1,
            learning_rate=1e-4,
            save_dir=self.fine_tuned_dir
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Fine-tuning completed in {elapsed:.1f}s")
        
        return {
            'stage': 'fine_tuning',
            'time': elapsed,
            'history': history
        }
    
    def run_fine_tuned_inference(self, test_split: str = "val") -> dict:
        """
        Stage 3: Run inference with fine-tuned model.
        
        Args:
            test_split: Which split to test on ('val' or 'test')
        
        Returns:
            Results dictionary
        """
        print("\n" + "=" * 80)
        print("STAGE 3: FINE-TUNED SAM INFERENCE")
        print("=" * 80)
        
        test_images_dir = self.data_root / test_split / "images"
        test_masks_dir = self.data_root / test_split / "masks"
        
        print(f"Input images: {test_images_dir}")
        print(f"Ground truth: {test_masks_dir}")
        print(f"Output masks: {self.fine_tuned_inference_dir}")
        print()
        
        start_time = time.time()
        
        # Create fine-tuned model
        fine_tuner = SAMFineTuner()
        
        # Load fine-tuned model checkpoint into decoder
        latest_checkpoint = list(self.fine_tuned_dir.glob("sam_finetuned_epoch*.pth"))
        if latest_checkpoint:
            latest_checkpoint = sorted(latest_checkpoint)[-1]
            print(f"Loaded checkpoint: {latest_checkpoint}")
            state_dict = torch.load(latest_checkpoint, map_location=fine_tuner.device)
            # Load the full state dict (which includes fine-tuned decoder weights)
            fine_tuner.model.model.load_state_dict(state_dict)
        
        results = {}
        
        for img_path in sorted(test_images_dir.glob("*.png")):
            import numpy as np
            from PIL import Image
            
            image = np.array(Image.open(img_path).convert('RGB'))
            
            # Load ground truth mask for proper prompting
            gt_mask = None
            mask_path = test_masks_dir / f"{img_path.stem}_mask.png"
            if mask_path.exists():
                gt_mask = np.array(Image.open(mask_path))
            
            mask, score = fine_tuner.segment_mri_slice(image, gt_mask=gt_mask)
            
            results[img_path.name] = {
                'mask': mask,
                'confidence': score
            }
            
            # Save mask
            Image.fromarray((mask * 255).astype(np.uint8)).save(
                self.fine_tuned_inference_dir / f"{img_path.stem}_mask.png"
            )
            
            print(f"✓ {img_path.name}")
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Fine-tuned inference completed in {elapsed:.1f}s")
        
        return {
            'stage': 'fine_tuned_inference',
            'time': elapsed,
            'results': results
        }
    
    def run_evaluation(self, test_split: str = "val") -> dict:
        """
        Stage 4: Evaluate predictions.
        
        Args:
            test_split: Which split to evaluate on
        
        Returns:
            Evaluation results
        """
        print("\n" + "=" * 80)
        print("STAGE 4: EVALUATION")
        print("=" * 80)
        print()
        
        gt_dir = self.data_root / test_split / "masks"
        images_dir = self.data_root / test_split / "images"
        
        analyzer = EvaluationAnalyzer(gt_dir, self.output_root)
        
        # Evaluate both methods
        print("Evaluating zero-shot predictions...")
        zero_shot_results = analyzer.evaluate_predictions("SAM Zero-Shot", self.zero_shot_dir)
        
        print("Evaluating fine-tuned predictions...")
        fine_tuned_results = analyzer.evaluate_predictions("SAM Fine-Tuned", self.fine_tuned_inference_dir)
        
        # Print summary
        print("\n" + "=" * 80)
        print("METRICS SUMMARY - DICE COEFFICIENT")
        print("=" * 80)
        
        for results in [zero_shot_results, fine_tuned_results]:
            method = results['method']
            agg = results['aggregate']
            
            if agg:
                print(f"\n{method}:")
                print(f"  Dice:        {agg['dice_mean']:.4f} ± {agg['dice_std']:.4f}")
                print(f"              Range: [{agg['dice_min']:.4f}, {agg['dice_max']:.4f}]")
                print(f"  IoU:         {agg['iou_mean']:.4f} ± {agg['iou_std']:.4f}")
                print(f"  Sensitivity: {agg['sensitivity_mean']:.4f} ± {agg['sensitivity_std']:.4f}")
                print(f"  Specificity: {agg['specificity_mean']:.4f} ± {agg['specificity_std']:.4f}")
                print(f"  Samples:     {agg['num_images']}")
        
        # Save metrics to JSON
        metrics_file = self.eval_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'zero_shot': zero_shot_results,
                'fine_tuned': fine_tuned_results
            }, f, indent=2, default=str)
        print(f"\n✓ Metrics saved to: {metrics_file}")
        
        # Generate visualizations
        print("Generating comparison visualizations...")
        analyzer.generate_comparison_images(
            images_dir,
            gt_dir,
            {
                'Zero-Shot': self.zero_shot_dir,
                'Fine-Tuned': self.fine_tuned_inference_dir
            },
            self.eval_dir / "comparisons"
        )
        
        return {
            'stage': 'evaluation',
            'zero_shot': zero_shot_results,
            'fine_tuned': fine_tuned_results
        }
    
    def run_full_pipeline(self, stages: list = None):
        """
        Run complete SAM pipeline.
        
        Args:
            stages: List of stages to run (default: all)
        """
        if stages is None:
            stages = ['zero_shot', 'fine_tuning', 'fine_tuned_inference', 'evaluation']
        
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "SAM BRAIN TUMOR SEGMENTATION PIPELINE".center(78) + "║")
        print("║" + f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        
        pipeline_start = time.time()
        results = {}
        
        try:
            if 'zero_shot' in stages:
                results['zero_shot'] = self.run_zero_shot()
            
            if 'fine_tuning' in stages:
                results['fine_tuning'] = self.run_fine_tuning()
            
            if 'fine_tuned_inference' in stages:
                results['fine_tuned_inference'] = self.run_fine_tuned_inference()
            
            if 'evaluation' in stages:
                results['evaluation'] = self.run_evaluation()
            
            pipeline_elapsed = time.time() - pipeline_start
            
            # Summary
            print("\n" + "=" * 80)
            print("PIPELINE COMPLETE")
            print("=" * 80)
            print(f"Total time: {pipeline_elapsed:.1f}s")
            print(f"Output directory: {self.output_root}")
            print()
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SAM Brain Tumor Segmentation Pipeline"
    )
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['zero_shot', 'fine_tuning', 'fine_tuned_inference', 'evaluation'],
        default=['zero_shot', 'fine_tuning', 'fine_tuned_inference', 'evaluation'],
        help='Which stages to run'
    )
    parser.add_argument(
        '--test-split',
        choices=['val', 'test'],
        default='val',
        help='Dataset split to test on'
    )
    
    args = parser.parse_args()
    
    pipeline = SAMPipeline()
    pipeline.run_full_pipeline(stages=args.stages)


if __name__ == "__main__":
    main()
