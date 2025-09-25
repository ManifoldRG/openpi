#!/usr/bin/env python3
"""
PIQA Inference with Weight-Injected HuggingFace PaliGemma

This script evaluates the Pi0 weight-injected HuggingFace PaliGemma model on the PIQA dataset.
It uses the weight injection module to create a model with Pi0 weights, then evaluates it
using the standard PIQA evaluation metrics.

Usage:
    python piqa_hf_inference.py --dataset_dir /path/to/piqa/test --output_dir ./results
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
from src.eval_utils import get_exact_match_rate

# Import PIQA dataloader
from src.data_utils.piqa_dataloader import get_piqa_test_dataloader

# Import the Pi0 weight injector
from src.v1.modules.openpi.scripts.pi0_weight_injector import get_pi0_injected_model

# PIQA system prompt (same as used in piqa_module.py)
PIQA_SYSTEM_PROMPT = """<image>You are evaluating physical commonsense reasoning questions. You will be presented with a goal and possible solutions.
    Your task is to determine which solution is more appropriate for achieving the given goal.
    Output only the index of the correct solution, and nothing else.
    Do not provide any explanation, reasoning, or additional text."""


@dataclass
class DatasetResults:
    """Results from PIQA model inference evaluation"""
    all_preds: List[int] = field(default_factory=list)
    all_gt: List[int] = field(default_factory=list)
    
    total_batches: int = 0
    total_samples: int = 0
    eval_time: float = 0
    total_invalid_predictions: int = 0
    invalid_predictions_percentage: float = 0
    
    # Final metrics
    exact_match_rate: float = 0
    exact_match_rate_with_invalids: float = 0
    percentage_invalids: float = 0

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }


def _validate_output(output: str) -> bool:
    """Validate that output is exactly '0' or '1'"""
    if not isinstance(output, str):
        return False
    return output.strip() in ['0', '1']


def _validate_outputs_and_calculate_metrics(outputs: List[str], labels: List[int]):
    """Validate outputs and convert to predictions for exact match calculation"""
    preds = []
    total_invalid_preds = 0
    
    for output in outputs:
        if _validate_output(output):
            preds.append(int(output.strip()))
        else:
            total_invalid_preds += 1
            preds.append(-1)  # Invalid prediction marker
    
    return total_invalid_preds, preds


def _calculate_final_metrics(preds: List[int], trues: List[int]) -> Dict[str, Any]:
    """Calculate final metrics for PIQA evaluation"""
    result = {}
    
    valid_preds = []
    valid_trues = []
    invalid_count = 0
    
    for pred, true in zip(preds, trues):
        if pred == -1:  # Invalid prediction
            invalid_count += 1
        else:
            valid_preds.append(pred)
            valid_trues.append(true)
    
    # Calculate exact match rate only on valid predictions
    if len(valid_preds) > 0:
        exact_match_rate = get_exact_match_rate(np.array(valid_preds), np.array(valid_trues))
    else:
        exact_match_rate = 0.0
    
    # Calculate exact match rate including invalids as wrong (more conservative metric)
    total_predictions = len(preds)
    correct_predictions = sum(1 for pred, true in zip(preds, trues) if pred == true and pred != -1)
    exact_match_rate_with_invalids = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    result["exact_match_rate"] = exact_match_rate
    result["exact_match_rate_with_invalids"] = exact_match_rate_with_invalids
    result["total_predictions"] = total_predictions
    result["valid_predictions"] = len(valid_preds)
    result["invalid_predictions"] = invalid_count
    result["percentage_invalids"] = (invalid_count / total_predictions) * 100 if total_predictions > 0 else 0.0
    result["preds"] = [int(pred) for pred in preds]
    result["gt_labels"] = [int(true) for true in trues]
    
    return result


class PIQAInferenceHF:
    """PIQA inference class using HuggingFace PaliGemma with Pi0 weight injection"""
    
    def __init__(self, model_id: str = "google/paligemma-3b-pt-224", device: str = None, args: argparse.Namespace = None):
        """
        Initialize the PIQA inference class.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run inference on (cuda, cpu, etc.)
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.args = args

        print(f"Initializing PIQA inference with device: {self.device}")
    
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
    
    def prepare_inputs(self, questions: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the HuggingFace model.
        
        Args:
            questions: PIQA question text (single string or list of strings for batch)
            
        Returns:
            Processed inputs for the model
        """
        # Handle both single question and batch of questions
        if isinstance(questions, str):
            questions = [questions]
        
        # Create tensor dummy images (since PIQA is text-only but PaliGemma expects images)
        dummy_images = [Image.new('RGB', (224, 224), color=(0, 0, 0)) for _ in questions]

        # Format the prompts with the system prompt and questions
        prompts = [f"{PIQA_SYSTEM_PROMPT}\n\n{question}" for question in questions]
        
        # Process inputs for batch
        inputs = self.processor(
            images=dummy_images,
            text=prompts,
            return_tensors="pt",
            padding=True  # Ensure proper padding for batch processing
        )

        # Zero out attention on image token *keys*
        img_id = self.model.config.image_token_index  # e.g., 256000
        am = inputs["attention_mask"].clone()      # (B, T)
        print("attention mask values and counts before zeroing out img tokens:\n", torch.unique(am, return_counts=True ))
        img_pos = (inputs["input_ids"] == img_id)  # (B, T) True where <img> placeholders are
        am[img_pos] = 0
        inputs["attention_mask"] = am
        print("attention mask values and counts after zeroing out img tokens:\n", torch.unique(inputs["attention_mask"], return_counts=True ))

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
                max_new_tokens=10,  # We only need 1 token ("0" or "1"), but allow some buffer
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
    
    def process_batch(self, questions: List[str]) -> List[str]:
        """
        Process a batch of questions using true batch inference.
        
        Args:
            questions: List of PIQA questions
            
        Returns:
            List of generated responses
        """
        try:
            # Prepare inputs for the entire batch
            inputs = self.prepare_inputs(questions)
            
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
    
    def evaluate_model(self, dataloader) -> Dict[str, Any]:
        """
        Evaluate the model on PIQA dataset.
        
        Args:
            dataloader: PIQA data loader
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Starting PIQA evaluation...")
        dataset_results = DatasetResults()
        start_time = time.perf_counter()
        
        all_preds = []
        all_trues = []
        total_invalid_preds = 0
        
        for batch_idx, batch in enumerate(dataloader):
            questions = batch["question"]
            labels = batch["label"]
            
            print(f"Processing batch {batch_idx + 1} with {len(questions)} samples...")
            
            # Process batch
            batch_outputs = self.process_batch(questions)
            
            # Validate outputs and calculate metrics for this batch
            invalid_preds, preds = _validate_outputs_and_calculate_metrics(batch_outputs, labels)
            total_invalid_preds += invalid_preds
            all_preds.extend(preds)
            all_trues.extend(labels)
            
            # Update results
            dataset_results.total_batches = batch_idx + 1
            dataset_results.total_samples += len(questions)
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")
            
            if dataset_results.total_samples >= self.args.max_samples:
                print(f"Reached maximum sample limit of {self.args.max_samples}. Stopping evaluation.")
                break
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate final metrics
        final_metrics = _calculate_final_metrics(all_preds, all_trues)
        
        # Update dataset results
        dataset_results.all_preds = final_metrics["preds"]
        dataset_results.all_gt = final_metrics["gt_labels"]
        dataset_results.exact_match_rate = final_metrics["exact_match_rate"]
        dataset_results.exact_match_rate_with_invalids = final_metrics["exact_match_rate_with_invalids"]
        dataset_results.total_invalid_predictions = final_metrics["invalid_predictions"]
        dataset_results.invalid_predictions_percentage = final_metrics["percentage_invalids"]
        dataset_results.percentage_invalids = final_metrics["percentage_invalids"]
        
        # Calculate evaluation time
        end_time = time.perf_counter()
        dataset_results.eval_time = end_time - start_time
        
        print(f"\nEvaluation completed!")
        print(f"Total samples: {dataset_results.total_samples}")
        print(f"Exact match rate: {dataset_results.exact_match_rate:.4f}")
        print(f"Exact match rate (with invalids): {dataset_results.exact_match_rate_with_invalids:.4f}")
        print(f"Invalid predictions: {dataset_results.total_invalid_predictions} ({dataset_results.percentage_invalids:.2f}%)")
        print(f"Evaluation time: {dataset_results.eval_time:.2f} seconds")
        
        return dataset_results.to_dict()


def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run PIQA inference with Pi0 weight-injected HuggingFace PaliGemma"
    )

    parser.add_argument(
        '--mask_image_tokens',
        action='store_true',
        default=True,
        help='Whether to mask dummy image tokens in the input'
    )

    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Directory containing the PIQA test dataset'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./piqa_hf_inference_results',
        help='Directory to store inference results (default: ./piqa_hf_inference_results)'
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
    """Main function to run PIQA inference"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be stored in: {args.output_dir}")
    print(f"Reading PIQA dataset from: {args.dataset_dir}")
    
    try:
        # Initialize inference class
        piqa_inference = PIQAInferenceHF(
            model_id=args.model_id,
            device=args.device,
            args=args
        )
        
        # Load the Pi0 weight-injected model
        piqa_inference.load_model()
        
        # Create PIQA dataloader
        print("Loading PIQA dataset...")
        dataset_obj, dataloader = get_piqa_test_dataloader(
            test_dir=args.dataset_dir,
            batch_size=args.batch_size
        )
        
        print(f"Created dataloader with {len(dataset_obj)} samples")
        
        # Limit samples if specified
        if args.max_samples is not None:
            print(f"Limiting evaluation to {args.max_samples} samples")
    
        
        # Run evaluation
        results = piqa_inference.evaluate_model(dataloader)
        
        # Save results
        results_file = os.path.join(args.output_dir, 'piqa_hf_inference_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to: {results_file}")
        
        # Print final summary
        print(f"\n=== PIQA HuggingFace Inference Results Summary ===")
        print(f"Model: {args.model_id}")
        print(f"Device: {piqa_inference.device}")
        print(f"Total samples: {results.get('total_samples', 0)}")
        print(f"Exact Match Rate: {results.get('exact_match_rate', 0):.4f}")
        print(f"Exact Match Rate (with invalids): {results.get('exact_match_rate_with_invalids', 0):.4f}")
        print(f"Invalid predictions: {results.get('total_invalid_predictions', 0)} ({results.get('percentage_invalids', 0):.2f}%)")
        print(f"Evaluation time: {results.get('eval_time', 0):.2f} seconds")
        print(f"======================================================")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
