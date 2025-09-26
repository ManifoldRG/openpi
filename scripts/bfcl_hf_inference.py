#!/usr/bin/env python3
"""
BFCL Inference with Weight-Injected HuggingFace PaliGemma

This script evaluates the Pi0 weight-injected HuggingFace PaliGemma model on the BFCL dataset.
It uses the weight injection module to create a model with Pi0 weights, then evaluates it
using BFCL-specific evaluation metrics for multi-turn function calling tasks.

Usage:
    python bfcl_hf_inference.py --dataset_dir /path/to/bfcl/test --output_dir ./results
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

# Import BFCL dataloader
from src.data_utils.bfcl_dataloader import get_bfcl_test_dataloader

# Import the Pi0 weight injector
from src.v1.modules.openpi.scripts.pi0_weight_injector import get_pi0_injected_model

# Import similarity model for evaluation
from sentence_transformers import SentenceTransformer, util

# BFCL system prompt for function calling tasks
BFCL_SYSTEM_PROMPT = """You are an AI assistant that can call functions to complete tasks. You will be presented with multi-turn conversations where each turn may require function calls.

For each turn, analyze the user's request and output the exact sequence of function calls needed for each turn.
Format each function call as: function_name(param1=value1, param2=value2, ...)
Use only the exact function names available in the provided set of functions and append appropriate parameters.
Output only the function calls, no explanations or additional text."""


def _validate_output(output: str) -> bool:
    """Validate that output contains content."""
    return isinstance(output, str) and bool(output.strip())


def _validate_outputs_and_calculate_metrics(similarity_model: SentenceTransformer, outputs: List[str], ground_truth_list: List[List[List[str]]]):
    """Validate outputs and calculate exact match and similarity metrics for BFCL.
    
    Args:
        similarity_model: SentenceTransformer model for computing similarity scores
        outputs: List of model outputs (one per sample)
        ground_truth_list: List of ground truth function calls per sample.
                          Each sample contains multiple turns, each turn contains multiple function calls.
                          Format: [[[turn1_func1, turn1_func2], [turn2_func1]], ...]
    """
    exact_matches = []
    similarity_scores = []
    predicted_calls_list = []
    total_invalid_preds = 0
    
    for i, output in enumerate(outputs):
        if _validate_output(output):
            # Split output into lines (predicted function calls)
            predicted_calls = [line.strip() for line in output.strip().split('\n') if line.strip()]
            predicted_calls_list.append(predicted_calls)
            
            # Get ground truth for this sample
            gt_turns = ground_truth_list[i] if i < len(ground_truth_list) else []
            
            # Flatten ground truth turns into a single list of function calls
            flattened_gt_calls = []
            for turn_calls in gt_turns:
                if isinstance(turn_calls, list):
                    for call in turn_calls:
                        if isinstance(call, str) and call.strip():
                            flattened_gt_calls.append(call.strip())
                elif isinstance(turn_calls, str) and turn_calls.strip():
                    flattened_gt_calls.append(turn_calls.strip())
            
            # Calculate exact match (order matters for function sequences)
            exact_match = 1.0 if predicted_calls == flattened_gt_calls else 0.0
            exact_matches.append(exact_match)
            
            # For similarity scoring, join function calls into text
            predicted_text = "\n".join(predicted_calls)
            gt_text = "\n".join(flattened_gt_calls)
            
            # Calculate similarity score using sentence embeddings
            try:
                emb1 = similarity_model.encode(predicted_text, convert_to_tensor=True)
                emb2 = similarity_model.encode(gt_text, convert_to_tensor=True)
                similarity = util.cos_sim(emb1, emb2).item()
                similarity_scores.append(similarity)
            except Exception as e:
                print(f"Warning: Similarity calculation failed for sample {i}: {e}")
                similarity_scores.append(0.0)
        else:
            # Invalid output
            exact_matches.append(0.0)
            similarity_scores.append(0.0)
            predicted_calls_list.append([])
            total_invalid_preds += 1
    
    return exact_matches, similarity_scores, predicted_calls_list, total_invalid_preds


def _calculate_final_metrics(exact_matches: List[float], similarity_scores: List[float], total_invalid_preds: int, 
                           predicted_calls: List[List[str]], ground_truth_calls: List[List[List[str]]]) -> Dict[str, Any]:
    """Calculate comprehensive final metrics for BFCL evaluation.
    
    Args:
        exact_matches: List of exact match scores (0.0 or 1.0) per sample
        similarity_scores: List of similarity scores per sample
        total_invalid_preds: Number of invalid predictions
        predicted_calls: List of predicted function calls per sample
        ground_truth_calls: List of ground truth function calls per sample (nested: sample -> turn -> calls)
    """
    result = {}
    
    total_samples = len(exact_matches)
    
    # Calculate accuracy metrics
    exact_match_accuracy = sum(exact_matches) / total_samples if total_samples > 0 else 0.0
    exact_match_accuracy_without_invalids = sum(exact_matches) / (total_samples - total_invalid_preds) if total_samples - total_invalid_preds > 0 else 0.0
    
    # Flatten ground truth for function-level metrics
    flattened_ground_truth = []
    for sample_gt in ground_truth_calls:
        sample_flattened = []
        for turn_calls in sample_gt:
            if isinstance(turn_calls, list):
                for call in turn_calls:
                    if isinstance(call, str) and call.strip():
                        sample_flattened.append(call.strip())
            elif isinstance(turn_calls, str) and turn_calls.strip():
                sample_flattened.append(turn_calls.strip())
        flattened_ground_truth.append(sample_flattened)
    
    # Calculate function-level metrics
    total_predicted_functions = sum(len(calls) for calls in predicted_calls)
    total_ground_truth_functions = sum(len(calls) for calls in flattened_ground_truth)
    
    # Calculate per-turn metrics
    correct_function_counts = 0
    total_function_comparisons = 0
    
    for pred_calls, gt_calls in zip(predicted_calls, flattened_ground_truth):
        # Count correct functions (regardless of order for this metric)
        for pred_call in pred_calls:
            if pred_call in gt_calls:
                correct_function_counts += 1
        total_function_comparisons += max(len(pred_calls), len(gt_calls))
    
    function_level_accuracy = correct_function_counts / total_function_comparisons if total_function_comparisons > 0 else 0.0
    
    # Calculate similarity metrics
    avg_similarity_score = sum(similarity_scores) / total_samples if total_samples > 0 else 0.0
    max_similarity_score = max(similarity_scores) if similarity_scores else 0.0
    min_similarity_score = min(similarity_scores) if similarity_scores else 0.0
    similarity_std = np.std(similarity_scores) if similarity_scores else 0.0
    
    # Calculate percentage of high similarity matches (threshold-based)
    high_similarity_threshold = 0.8
    high_similarity_count = sum(1 for score in similarity_scores if score >= high_similarity_threshold)
    high_similarity_percentage = (high_similarity_count / total_samples * 100) if total_samples > 0 else 0.0
    
    # Calculate invalid prediction percentage
    invalid_percentage = (total_invalid_preds / total_samples * 100) if total_samples > 0 else 0.0
    
    result['exact_match_accuracy'] = exact_match_accuracy
    result['exact_match_accuracy_without_invalids'] = exact_match_accuracy_without_invalids
    result['function_level_accuracy'] = function_level_accuracy
    result['avg_similarity_score'] = avg_similarity_score
    result['max_similarity_score'] = max_similarity_score
    result['min_similarity_score'] = min_similarity_score
    result['similarity_std'] = similarity_std
    result['high_similarity_percentage'] = high_similarity_percentage
    result['high_similarity_threshold'] = high_similarity_threshold
    result['total_samples'] = total_samples
    result['total_invalid_preds'] = total_invalid_preds
    result['invalid_percentage'] = invalid_percentage
    result['total_predicted_functions'] = total_predicted_functions
    result['total_ground_truth_functions'] = total_ground_truth_functions
    result['avg_predicted_functions_per_sample'] = total_predicted_functions / total_samples if total_samples > 0 else 0.0
    result['avg_ground_truth_functions_per_sample'] = total_ground_truth_functions / total_samples if total_samples > 0 else 0.0
    result['exact_matches'] = exact_matches
    result['similarity_scores'] = similarity_scores
    result['predicted_function_calls'] = predicted_calls
    result['ground_truth_function_calls'] = flattened_ground_truth  # Store flattened version
    
    return result


@dataclass
class DatasetResults:
    """Results from BFCL model inference evaluation"""
    all_exact_matches: List[float] = field(default_factory=list)
    all_similarity_scores: List[float] = field(default_factory=list)
    all_predicted_calls: List[List[str]] = field(default_factory=list)
    all_ground_truth_calls: List[List[List[str]]] = field(default_factory=list)  # Nested: sample -> turn -> calls
    all_original_outputs: List[str] = field(default_factory=list)
    
    total_batches: int = 0
    total_samples: int = 0
    eval_time: float = 0
    total_invalid_predictions: int = 0
    
    # Final metrics
    exact_match_accuracy: float = 0
    exact_match_accuracy_without_invalids: float = 0
    function_level_accuracy: float = 0
    avg_similarity_score: float = 0
    max_similarity_score: float = 0
    min_similarity_score: float = 0
    similarity_std: float = 0
    high_similarity_percentage: float = 0
    high_similarity_threshold: float = 0.8
    invalid_percentage: float = 0
    total_predicted_functions: int = 0
    total_ground_truth_functions: int = 0
    avg_predicted_functions_per_sample: float = 0
    avg_ground_truth_functions_per_sample: float = 0
    similarity_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }


class BFCLInferenceHF:
    """BFCL inference class using HuggingFace PaliGemma with Pi0 weight injection"""
    
    def __init__(self, model_id: str = "google/paligemma-3b-pt-224", device: str = None, args: argparse.Namespace = None):
        """
        Initialize the BFCL inference class.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run inference on (cuda, cpu, etc.)
            args: Command line arguments containing configuration options
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.similarity_model = None
        self.args = args
        
        print(f"Initializing BFCL inference with device: {self.device}")
    
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
    
    def load_similarity_model(self):
        """Load the similarity model for evaluation"""
        if self.similarity_model is None:
            print("Loading similarity model for evaluation...")
            self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            print("✓ Similarity model loaded successfully")
    
    def prepare_inputs(self, prompts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the HuggingFace model.
        
        Args:
            prompts: BFCL conversation prompts (single string or list of strings for batch)
            
        Returns:
            Processed inputs for the model
        """
        # Handle both single prompt and batch of prompts
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Create dummy images (since BFCL is text-only but PaliGemma expects images)
        dummy_images = [Image.new('RGB', (224, 224), color='white') for _ in prompts]
        
        # Format the prompts with the system prompt
        formatted_prompts = [f"{BFCL_SYSTEM_PROMPT}\n\n{prompt}" for prompt in prompts]
        
        # Process inputs for batch
        inputs = self.processor(
            images=dummy_images,
            text=formatted_prompts,
            return_tensors="pt",
            padding=True  # Ensure proper padding for batch processing
        )

        # Zero out attention on image token *keys*
        if self.args and self.args.mask_image_tokens:
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
                max_new_tokens=150,  # Allow more tokens for multiple function calls
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
    
    def process_batch(self, prompts: List[str]) -> List[str]:
        """
        Process a batch of prompts using true batch inference.
        
        Args:
            prompts: List of BFCL conversation prompts
            
        Returns:
            List of generated responses
        """
        try:
            # Prepare inputs for the entire batch
            inputs = self.prepare_inputs(prompts)
            
            # Generate responses for the entire batch
            responses = self.generate_response(inputs)
            
            # Ensure we return a list even for single prompts
            if isinstance(responses, str):
                responses = [responses]
                
            return responses
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Return empty responses for failed cases
            return [""] * len(prompts)
    
    def evaluate_model(self, dataloader, dataset_obj) -> Dict[str, Any]:
        """
        Evaluate the model on BFCL dataset.
        
        Args:
            dataloader: BFCL data loader
            dataset_obj: BFCL dataset object
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Load similarity model for evaluation
        self.load_similarity_model()
        
        print("Starting BFCL evaluation...")
        dataset_results = DatasetResults()
        start_time = time.perf_counter()
        
        all_exact_matches = []
        all_similarity_scores = []
        all_predicted_calls = []
        all_ground_truth_calls = []
        all_original_outputs = []
        total_invalid_preds = 0
        
        for batch_idx, batch in enumerate(dataloader):
            prompts = batch['prompt']
            ground_truth_functions = batch['ground_truth_functions']
            
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)} with {len(prompts)} samples...")
            
            # Process batch
            batch_outputs = self.process_batch(prompts)
            
            # Validate outputs and calculate metrics for this batch
            exact_matches, similarity_scores, predicted_calls, invalid_preds = _validate_outputs_and_calculate_metrics(
                self.similarity_model, batch_outputs, ground_truth_functions
            )
            
            total_invalid_preds += invalid_preds
            all_exact_matches.extend(exact_matches)
            all_similarity_scores.extend(similarity_scores)
            all_predicted_calls.extend(predicted_calls)
            all_ground_truth_calls.extend(ground_truth_functions)
            all_original_outputs.extend(batch_outputs)
            
            # Update results
            dataset_results.total_batches = batch_idx + 1
            dataset_results.total_samples += len(prompts)
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                current_accuracy = sum(all_exact_matches) / len(all_exact_matches) if all_exact_matches else 0.0
                current_similarity = sum(all_similarity_scores) / len(all_similarity_scores) if all_similarity_scores else 0.0
                print(f"Progress: {batch_idx + 1} batches processed. Current accuracy: {current_accuracy:.4f}, Current avg similarity: {current_similarity:.4f}")
            
            # Check if we should stop early due to max_samples limit
            if hasattr(self, 'args') and self.args and self.args.max_samples is not None and dataset_results.total_samples >= self.args.max_samples:
                print(f"Reached maximum sample limit of {self.args.max_samples}. Stopping evaluation.")
                break
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate final metrics
        final_metrics = _calculate_final_metrics(
            all_exact_matches, all_similarity_scores, total_invalid_preds, all_predicted_calls, all_ground_truth_calls
        )
        
        # Update dataset results
        dataset_results.all_exact_matches = final_metrics["exact_matches"]
        dataset_results.all_similarity_scores = final_metrics["similarity_scores"]
        dataset_results.all_predicted_calls = final_metrics["predicted_function_calls"]
        dataset_results.all_ground_truth_calls = final_metrics["ground_truth_function_calls"]
        dataset_results.all_original_outputs = all_original_outputs
        dataset_results.exact_match_accuracy = final_metrics["exact_match_accuracy"]
        dataset_results.exact_match_accuracy_without_invalids = final_metrics["exact_match_accuracy_without_invalids"]
        dataset_results.function_level_accuracy = final_metrics["function_level_accuracy"]
        dataset_results.avg_similarity_score = final_metrics["avg_similarity_score"]
        dataset_results.max_similarity_score = final_metrics["max_similarity_score"]
        dataset_results.min_similarity_score = final_metrics["min_similarity_score"]
        dataset_results.similarity_std = final_metrics["similarity_std"]
        dataset_results.high_similarity_percentage = final_metrics["high_similarity_percentage"]
        dataset_results.high_similarity_threshold = final_metrics["high_similarity_threshold"]
        dataset_results.total_invalid_predictions = final_metrics["total_invalid_preds"]
        dataset_results.invalid_percentage = final_metrics["invalid_percentage"]
        dataset_results.total_predicted_functions = final_metrics["total_predicted_functions"]
        dataset_results.total_ground_truth_functions = final_metrics["total_ground_truth_functions"]
        dataset_results.avg_predicted_functions_per_sample = final_metrics["avg_predicted_functions_per_sample"]
        dataset_results.avg_ground_truth_functions_per_sample = final_metrics["avg_ground_truth_functions_per_sample"]
        dataset_results.similarity_scores = final_metrics["similarity_scores"]
        
        # Calculate evaluation time
        end_time = time.perf_counter()
        dataset_results.eval_time = end_time - start_time
        
        print(f"\nEvaluation completed!")
        print(f"Total samples: {dataset_results.total_samples}")
        print(f"Exact match accuracy: {dataset_results.exact_match_accuracy:.4f}")
        print(f"Exact match accuracy (without invalids): {dataset_results.exact_match_accuracy_without_invalids:.4f}")
        print(f"Function-level accuracy: {dataset_results.function_level_accuracy:.4f}")
        print(f"Average similarity score: {dataset_results.avg_similarity_score:.4f}")
        print(f"High similarity percentage (≥{dataset_results.high_similarity_threshold}): {dataset_results.high_similarity_percentage:.2f}%")
        print(f"Invalid predictions: {dataset_results.total_invalid_predictions} ({dataset_results.invalid_percentage:.2f}%)")
        print(f"Average predicted functions per sample: {dataset_results.avg_predicted_functions_per_sample:.2f}")
        print(f"Average ground truth functions per sample: {dataset_results.avg_ground_truth_functions_per_sample:.2f}")
        print(f"Evaluation time: {dataset_results.eval_time:.2f} seconds")
        
        return dataset_results.to_dict()


def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run BFCL inference with Pi0 weight-injected HuggingFace PaliGemma"
    )
    
    parser.add_argument(
        '--mask_image_tokens',
        type=bool,
        default=False,
        help='Whether to mask dummy image tokens in the input'
    )

    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Directory containing the BFCL test dataset'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./bfcl_hf_inference_results',
        help='Directory to store inference results (default: ./bfcl_hf_inference_results)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for inference (default: 4)'
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
    """Main function to run BFCL inference"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be stored in: {args.output_dir}")
    print(f"Reading BFCL dataset from: {args.dataset_dir}")
    
    try:
        # Initialize inference class
        bfcl_inference = BFCLInferenceHF(
            model_id=args.model_id,
            device=args.device,
            args=args
        )
        
        # Load the Pi0 weight-injected model
        bfcl_inference.load_model()
        
        # Create BFCL dataloader
        print("Loading BFCL dataset...")
        dataset_obj, dataloader = get_bfcl_test_dataloader(
            test_dir=args.dataset_dir,
            batch_size=args.batch_size
        )
        
        print(f"Created dataloader with {len(dataset_obj)} samples")
        
        # Print dataset info
        dataset_info = dataset_obj.get_dataset_info()
        print(f"Dataset info:")
        print(f"  Total conversations: {dataset_info['num_conversations']}")
        print(f"  Total turns: {dataset_info['total_turns']}")
        print(f"  Average turns per conversation: {dataset_info['avg_turns_per_conversation']:.2f}")
        print(f"  Unique tool classes: {dataset_info['num_tool_classes']}")
        
        # Limit samples if specified
        if args.max_samples is not None:
            print(f"Limiting evaluation to {args.max_samples} samples")
        
        # Run evaluation
        results = bfcl_inference.evaluate_model(dataloader, dataset_obj)
        
        # Save results
        results_file = os.path.join(args.output_dir, 'bfcl_hf_inference_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to: {results_file}")
        
        # Print final summary
        print(f"\n=== BFCL HuggingFace Inference Results Summary ===")
        print(f"Model: {args.model_id}")
        print(f"Device: {bfcl_inference.device}")
        print(f"Total samples: {results.get('total_samples', 0)}")
        print(f"Exact Match Accuracy: {results.get('exact_match_accuracy', 0):.4f}")
        print(f"Exact Match Accuracy (without invalids): {results.get('exact_match_accuracy_without_invalids', 0):.4f}")
        print(f"Function-Level Accuracy: {results.get('function_level_accuracy', 0):.4f}")
        print(f"Average Similarity Score: {results.get('avg_similarity_score', 0):.4f}")
        print(f"Max Similarity Score: {results.get('max_similarity_score', 0):.4f}")
        print(f"Min Similarity Score: {results.get('min_similarity_score', 0):.4f}")
        print(f"Similarity Std Dev: {results.get('similarity_std', 0):.4f}")
        print(f"High Similarity (≥{results.get('high_similarity_threshold', 0.8)}): {results.get('high_similarity_percentage', 0):.2f}%")
        print(f"Invalid predictions: {results.get('total_invalid_predictions', 0)} ({results.get('invalid_percentage', 0):.2f}%)")
        print(f"Average predicted functions per sample: {results.get('avg_predicted_functions_per_sample', 0):.2f}")
        print(f"Average ground truth functions per sample: {results.get('avg_ground_truth_functions_per_sample', 0):.2f}")
        print(f"Evaluation time: {results.get('eval_time', 0):.2f} seconds")
        print(f"=====================================================")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
