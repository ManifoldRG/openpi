#!/usr/bin/env python3
"""
ODinW Inference with Weight-Injected HuggingFace PaliGemma

This script evaluates the Pi0 weight-injected HuggingFace PaliGemma model on the ODinW dataset.
It uses the weight injection module to create a model with Pi0 weights, then evaluates it
using ODinW-specific evaluation metrics including classification accuracy and F1 scores.

Usage:
    python odinw_hf_inference.py --dataset_dir /path/to/odinw/test --output_dir ./results
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import torch
from PIL import Image
import gc

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

# Import HuggingFace transformers
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# Import evaluation utilities
from src.eval_utils import (
    get_exact_match_rate,
    get_micro_precision_from_counts,
    get_micro_recall_from_counts,
    get_micro_f1,
    calculate_tp_fp_fn_counts,
    get_precision_per_class,
    get_recall_per_class,
    get_f1_per_class,
    get_macro_precision,
    get_macro_recall,
    get_macro_f1,
)

# Import ODinW dataloader and definitions
from src.data_utils.odinw_dataloader import get_odinw_dataloader
from definitions.odinw import ODinWDefinitions

# Import the Pi0 weight injector
from src.v1.modules.openpi.scripts.pi0_weight_injector import get_pi0_injected_model


def _validate_output(output: str, possible_outputs: List[int]) -> bool:
    """Validate that output is a valid integer within the possible outputs"""
    if not isinstance(output, str):
        return False
    try:
        output_int = int(output.strip())
        return output_int in possible_outputs
    except (ValueError, AttributeError):
        return False


def _validate_outputs_and_calculate_metrics(outputs: List[str], labels: List[int], possible_outputs: List[int]):
    """Validate outputs and convert to predictions for exact match calculation"""
    preds = []
    total_invalid_preds = 0
    
    for output in outputs:
        if _validate_output(output, possible_outputs):
            preds.append(int(output.strip()))
        else:
            total_invalid_preds += 1
            preds.append(-1)  # Invalid prediction marker
    
    return total_invalid_preds, preds


def _calculate_final_metrics(preds: List[int], labels: List[int], possible_outputs: List[int]) -> Dict[str, Any]:
    """Calculate comprehensive final metrics for ODinW evaluation"""
    result = {}
    
    valid_preds = []
    valid_trues = []
    invalid_count = 0
    
    for pred, true in zip(preds, labels):
        if pred == -1:  # Invalid prediction
            invalid_count += 1
        else:
            valid_preds.append(pred)
            valid_trues.append(true)
    
    # Convert to numpy arrays for metric calculations
    preds_array = np.array([int(pred) for pred in preds])
    labels_array = np.array([int(true) for true in labels])
    
    # Calculate exact match rate only on valid predictions
    if len(valid_preds) > 0:
        exact_match_rate = get_exact_match_rate(np.array(valid_preds), np.array(valid_trues))
    else:
        exact_match_rate = 0.0
    
    # Calculate exact match rate including invalids as wrong (more conservative metric)
    total_predictions = len(preds)
    correct_predictions = sum(1 for pred, true in zip(preds, labels) if pred == true and pred != -1)
    exact_match_rate_with_invalids = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Calculate metrics counts
    tp, fp, fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
        preds_array, labels_array, possible_outputs
    )
    
    # Calculate micro metrics
    precision = get_micro_precision_from_counts(tp, fp)
    precision_without_invalid = get_micro_precision_from_counts(tp, valid_fp)
    recall = get_micro_recall_from_counts(tp, fn)
    f1 = get_micro_f1(precision, recall)
    f1_without_invalid = get_micro_f1(precision_without_invalid, recall)
    
    # Calculate class-wise metrics
    class_precisions = get_precision_per_class(preds_array, labels_array, possible_outputs)
    class_recalls = get_recall_per_class(preds_array, labels_array, possible_outputs)
    class_f1s = get_f1_per_class(class_precisions, class_recalls)
    
    # Calculate macro metrics
    macro_precision = get_macro_precision(class_precisions)
    macro_recall = get_macro_recall(class_recalls)
    macro_f1 = get_macro_f1(class_f1s)
    
    # Populate result dictionary
    result["exact_match_rate"] = exact_match_rate
    result["exact_match_rate_with_invalids"] = exact_match_rate_with_invalids
    result["recall"] = recall
    result["precision"] = precision
    result["precision_without_invalid"] = precision_without_invalid
    result["f1"] = f1
    result["f1_without_invalid"] = f1_without_invalid
    result["macro_precision"] = macro_precision
    result["macro_recall"] = macro_recall
    result["macro_f1"] = macro_f1
    result["class_precisions"] = class_precisions.tolist() if isinstance(class_precisions, np.ndarray) else class_precisions
    result["class_recalls"] = class_recalls.tolist() if isinstance(class_recalls, np.ndarray) else class_recalls
    result["class_f1s"] = class_f1s.tolist() if isinstance(class_f1s, np.ndarray) else class_f1s
    result["total_invalids"] = int(invalid_fp)
    result["percentage_invalids"] = (invalid_fp / len(preds)) * 100 if len(preds) > 0 else 0.0
    result["total_predictions"] = total_predictions
    result["valid_predictions"] = len(valid_preds)
    result["invalid_predictions"] = invalid_count
    result["preds"] = [int(pred) for pred in preds]
    result["gt_labels"] = [int(true) for true in labels]
    
    return result


def _find_sub_dir(disk_root_dir: str, dataset: str) -> str:
    """Find ODinW sub-dataset directory"""
    dataset_dir = f"{disk_root_dir}/odinw/test/{dataset}"
    if os.path.exists(dataset_dir):
        return dataset_dir
    else:
        raise FileNotFoundError(f"Dataset directory not found for {dataset}. Available datasets: {list(ODinWDefinitions.SUB_DATASET_CATEGORIES.keys())}")


@dataclass
class DatasetResults:
    """Results from ODinW model inference evaluation"""
    all_preds: List[int] = field(default_factory=list)
    all_gt: List[int] = field(default_factory=list)
    
    total_batches: int = 0
    total_samples: int = 0
    eval_time: float = 0
    total_invalid_predictions: int = 0
    
    # Final metrics
    exact_match_rate: float = 0
    exact_match_rate_with_invalids: float = 0
    precision: float = 0
    precision_without_invalid: float = 0
    recall: float = 0
    f1: float = 0
    f1_without_invalid: float = 0
    macro_precision: float = 0
    macro_recall: float = 0
    macro_f1: float = 0
    percentage_invalids: float = 0
    
    # Class-wise metrics
    class_precisions: List[float] = field(default_factory=list)
    class_recalls: List[float] = field(default_factory=list)
    class_f1s: List[float] = field(default_factory=list)
    total_invalid_predictions: int = 0
    invalid_predictions_percentage: float = 0

    invalid_predictions: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }


class ODinWInferenceHF:
    """ODinW inference class using HuggingFace PaliGemma with Pi0 weight injection"""
    
    def __init__(self, model_id: str = "google/paligemma-3b-pt-224", device: str = None):
        """
        Initialize the ODinW inference class.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run inference on (cuda, cpu, etc.)
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        
        print(f"Initializing ODinW inference with device: {self.device}")
    
    def load_model(self):
        """Load the Pi0 weight-injected model and processor"""
        print("Loading Pi0 weight-injected PaliGemma model...")
        try:
            self.model, self.processor = get_pi0_injected_model(
                model_id=self.model_id,
                device=self.device
            )
            print("✓ Model and processor loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    def prepare_inputs(self, questions: Union[str, List[str]], images: Union[Image.Image, List[Image.Image], None] = None) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the HuggingFace model.
        
        Args:
            questions: ODinW question text (single string or list of strings for batch)
            images: Images (single image or list of images, or None for text-only)
            
        Returns:
            Processed inputs for the model
        """
        # Handle both single question and batch of questions
        if isinstance(questions, str):
            questions = [questions]
        
        # Handle images
        if images is None:
            # Create dummy images for text-only questions
            images = [Image.new('RGB', (224, 224), color='white') for _ in questions]
            print(f"Created {len(images)} dummy images for text-only questions")
        elif isinstance(images, Image.Image):
            images = [images]
        elif isinstance(images, list):
            # Ensure all elements are PIL Images, create dummy for None entries
            processed_images = []
            for img in images:
                if img is None:
                    processed_images.append(Image.new('RGB', (224, 224), color='white'))
                    print(f"Created a dummy image for a None entry")
                else:
                    processed_images.append(img)
            images = processed_images
        
        # Ensure we have the same number of questions and images
        assert len(questions) == len(images), f"Mismatch: {len(questions)} questions vs {len(images)} images"
        
        # Format the prompts with the system prompt and questions
        prompts = [f"{ODinWDefinitions.SYSTEM_PROMPT}\n\n{question}" for question in questions]
        
        # Process inputs for batch
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True  # Ensure proper padding for batch processing
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def generate_response(self, inputs: Dict[str, torch.Tensor]) -> Union[str, List[str]]:
        """
        Generate response from the model.
        
        Args:
            inputs: Processed inputs (can be single input or batch)
            
        Returns:
            Generated text response (single string or list of strings for batch)
        """
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,  # We only need 1-2 tokens for classification index
                do_sample=False,  # Use greedy decoding for consistency
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode only the generated part (excluding the input prompt)
        input_len = inputs["input_ids"].shape[-1]
        batch_size = outputs.shape[0]
        
        generated_texts = []
        for i in range(batch_size):
            generated_tokens = outputs[i][input_len:]
            generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text.strip())
        
        # Return single string if batch size is 1, otherwise return list
        if batch_size == 1:
            return generated_texts[0]
        else:
            return generated_texts
    
    def process_batch(self, questions: List[str], images: List[Image.Image] = None) -> List[str]:
        """
        Process a batch of questions using true batch inference.
        
        Args:
            questions: List of ODinW questions
            images: List of images (optional, can contain None entries)
            
        Returns:
            List of generated responses
        """
        try:
            # Prepare inputs for the entire batch
            inputs = self.prepare_inputs(questions, images)
            
            # Generate responses for the entire batch
            responses = self.generate_response(inputs)
            
            # Ensure we return a list even for single questions
            if isinstance(responses, str):
                responses = [responses]
                
            return responses
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Return empty responses for failed cases
            return [""] * len(questions)
    
    def evaluate_model(self, dataset_dir: str, dataset_name: str, batch_size: int = 8) -> Dict[str, Any]:
        """
        Evaluate the model on ODinW dataset.
        
        Args:
            dataset_dir: Directory containing the ODinW dataset
            dataset_name: Name of the ODinW sub-dataset
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Find dataset directory
        sub_dir = _find_sub_dir(dataset_dir, dataset_name)
        if not sub_dir:
            raise ValueError(f"Dataset directory not found for {dataset_name}")
        
        # Get possible outputs for this dataset
        if dataset_name not in ODinWDefinitions.SUB_DATASET_CATEGORIES:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(ODinWDefinitions.SUB_DATASET_CATEGORIES.keys())}")
        
        num_categories = ODinWDefinitions.SUB_DATASET_CATEGORIES[dataset_name]
        possible_outputs = list(range(num_categories))
        
        print(f"Dataset: {dataset_name}")
        print(f"Number of categories: {num_categories}")
        print(f"Possible outputs: {possible_outputs}")
        
        # Create dataloader
        dataset_obj, dataloader = get_odinw_dataloader(
            sub_dir,
            batch_size=batch_size
        )
        
        print(f"Created dataloader with {len(dataset_obj)} samples")
        print(f"Starting ODinW evaluation with {len(dataloader)} batches...")
        
        dataset_results = DatasetResults()
        start_time = time.perf_counter()
        
        all_preds = []
        all_trues = []
        total_invalid_preds = 0
        
        for batch_idx, batch in enumerate(dataloader):
            questions = batch["question"]
            labels = batch["correct_option_idx"]
            images = batch["image"]
            
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)} with {len(questions)} samples...")
            
            # Process batch
            batch_outputs = self.process_batch(questions, images)
            
            # Validate outputs and calculate metrics for this batch
            invalid_preds, preds = _validate_outputs_and_calculate_metrics(batch_outputs, labels, possible_outputs)
            total_invalid_preds += invalid_preds
            all_preds.extend(preds)
            all_trues.extend(labels)
            
            # Update results
            dataset_results.total_batches = batch_idx + 1
            dataset_results.total_samples += len(questions)
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                current_accuracy = sum(1 for p, t in zip(all_preds, all_trues) if p == t and p != -1) / len(all_preds) if all_preds else 0.0
                print(f"Progress: {batch_idx + 1} batches processed. Current accuracy: {current_accuracy:.4f}")
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate final metrics
        final_metrics = _calculate_final_metrics(all_preds, all_trues, possible_outputs)
        
        # Update dataset results
        dataset_results.all_preds = final_metrics["preds"]
        dataset_results.all_gt = final_metrics["gt_labels"]
        dataset_results.exact_match_rate = final_metrics["exact_match_rate"]
        dataset_results.exact_match_rate_with_invalids = final_metrics["exact_match_rate_with_invalids"]
        dataset_results.precision = final_metrics["precision"]
        dataset_results.precision_without_invalid = final_metrics["precision_without_invalid"]
        dataset_results.recall = final_metrics["recall"]
        dataset_results.f1 = final_metrics["f1"]
        dataset_results.f1_without_invalid = final_metrics["f1_without_invalid"]
        dataset_results.macro_precision = final_metrics["macro_precision"]
        dataset_results.macro_recall = final_metrics["macro_recall"]
        dataset_results.macro_f1 = final_metrics["macro_f1"]
        dataset_results.class_precisions = final_metrics["class_precisions"]
        dataset_results.class_recalls = final_metrics["class_recalls"]
        dataset_results.class_f1s = final_metrics["class_f1s"]
        dataset_results.total_invalid_predictions = final_metrics["total_invalids"]
        dataset_results.percentage_invalids = final_metrics["percentage_invalids"]
        dataset_results.total_invalid_predictions = final_metrics["total_invalids"]
        # Calculate evaluation time
        end_time = time.perf_counter()
        dataset_results.eval_time = end_time - start_time
        
        print(f"\nEvaluation completed!")
        print(f"Total samples: {dataset_results.total_samples}")
        print(f"Exact match rate: {dataset_results.exact_match_rate:.4f}")
        print(f"Exact match rate (with invalids): {dataset_results.exact_match_rate_with_invalids:.4f}")
        print(f"Precision: {dataset_results.precision:.4f}")
        print(f"Recall: {dataset_results.recall:.4f}")
        print(f"F1 Score: {dataset_results.f1:.4f}")
        print(f"Macro Precision: {dataset_results.macro_precision:.4f}")
        print(f"Macro Recall: {dataset_results.macro_recall:.4f}")
        print(f"Macro F1: {dataset_results.macro_f1:.4f}")
        print(f"Invalid predictions: {dataset_results.total_invalid_predictions} ({dataset_results.percentage_invalids:.2f}%)")
        print(f"Evaluation time: {dataset_results.eval_time:.2f} seconds")
        
        return dataset_results.to_dict()


def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run ODinW inference with Pi0 weight-injected HuggingFace PaliGemma"
    )
    
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Directory containing the ODinW dataset'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./odinw_hf_inference_results',
        help='Directory to store inference results (default: ./odinw_hf_inference_results)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for inference (default: 8)'
    )
    
    parser.add_argument(
        '--model_id',
        type=str,
        default='google/paligemma-3b-pt-224',
        help='HuggingFace model identifier (default: google/paligemma-3b-pt-224)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run inference on (cuda, cpu, etc.). Auto-detect if not specified.'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (default: all samples)'
    )
    
    args = parser.parse_args()
    
    # Validate dataset directory exists
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    
    return args


def main():
    """Main function to run ODinW inference"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be stored in: {args.output_dir}")
    print(f"Reading ODinW dataset from: {args.dataset_dir}")
    
    try:
        for dataset_name in ODinWDefinitions.SUB_DATASET_CATEGORIES.keys():
            # Initialize inference class
            odinw_inference = ODinWInferenceHF(
                model_id=args.model_id,
                device=args.device
            )
            
            # Load the Pi0 weight-injected model
            odinw_inference.load_model()
            
            # Run evaluation
            results = odinw_inference.evaluate_model(
                dataset_dir=args.dataset_dir,
                dataset_name=dataset_name,
                batch_size=args.batch_size
            )
            
            # Save results
            results_file = os.path.join(args.output_dir, f'odinw_{dataset_name}_hf_inference_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"\nResults saved to: {results_file}")
            
            # Print final summary
            print(f"\n=== ODinW HuggingFace Inference Results Summary ===")
            print(f"Model: {args.model_id}")
            print(f"Device: {odinw_inference.device}")
            print(f"Dataset: {dataset_name}")
            print(f"Total samples: {results.get('total_samples', 0)}")
            print(f"Exact Match Rate: {results.get('exact_match_rate', 0):.4f}")
            print(f"Exact Match Rate (with invalids): {results.get('exact_match_rate_with_invalids', 0):.4f}")
            print(f"Precision: {results.get('precision', 0):.4f}")
            print(f"Recall: {results.get('recall', 0):.4f}")
            print(f"F1 Score: {results.get('f1', 0):.4f}")
            print(f"Macro Precision: {results.get('macro_precision', 0):.4f}")
            print(f"Macro Recall: {results.get('macro_recall', 0):.4f}")
            print(f"Macro F1: {results.get('macro_f1', 0):.4f}")
            print(f"Invalid predictions: {results.get('total_invalid_predictions', 0)} ({results.get('percentage_invalids', 0):.2f}%)")
            print(f"Evaluation time: {results.get('eval_time', 0):.2f} seconds")
            print(f"======================================================")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
