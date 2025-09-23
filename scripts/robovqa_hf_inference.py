#!/usr/bin/env python3
"""
RoboVQA Inference with Weight-Injected HuggingFace PaliGemma

This script evaluates the Pi0 weight-injected HuggingFace PaliGemma model on the RoboVQA dataset.
It uses the weight injection module to create a model with Pi0 weights, then evaluates it
using RoboVQA-specific evaluation metrics including text similarity scoring.

Usage:
    python robovqa_hf_inference.py --dataset_dir /path/to/robovqa/test --output_dir ./results
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Any, List, Union
import re
from glob import glob
import tensorflow as tf
import numpy as np
import torch
from PIL import Image
import gc

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

# Import HuggingFace transformers
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# Import evaluation utilities and modules
from src.data_utils.openx_dataloader import get_openx_dataloader
from definitions.robovqa_prompt import ROBOVQA_PROMPT

# Import the Pi0 weight injector
from src.v1.modules.openpi.scripts.pi0_weight_injector import get_pi0_injected_model

# Import similarity model for evaluation
from sentence_transformers import SentenceTransformer, util

# Restrict tf to CPU
tf.config.set_visible_devices([], "GPU")
# Configure JAX memory settings
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


def _validate_text_output(output: Any) -> bool:
    """Validate that output is a valid text string."""
    if output is None:
        return False
    if isinstance(output, str) and len(output.strip()) > 0:
        return True
    return False


def _normalize_text(text: str) -> str:
    """Normalize text for comparison by removing punctuation and extra spaces."""
    if not isinstance(text, str):
        return ""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def _validate_outputs_and_calculate_metrics(similarity_model: SentenceTransformer, outputs: List[str], labels: List[str]):
    """Validate outputs and calculate text similarity metrics for RoboVQA."""
    exact_matches = []
    similarity_scores = []
    total_invalid_preds = 0
    
    for i, output in enumerate(outputs):
        if _validate_text_output(output):
            # Normalize both output and label for fair comparison
            normalized_output = _normalize_text(output)
            normalized_label = _normalize_text(labels[i])
            
            # Calculate exact match
            exact_match = 1.0 if normalized_output == normalized_label else 0.0
            exact_matches.append(exact_match)
            
            # Calculate similarity score
            emb1 = similarity_model.encode(output, convert_to_tensor=True)
            emb2 = similarity_model.encode(labels[i], convert_to_tensor=True)
            
            similarity = util.cos_sim(emb1, emb2).item()
            similarity_scores.append(similarity)
        else:
            # Invalid output - assign worst possible scores
            exact_matches.append(0.0)
            similarity_scores.append(0.0)
            total_invalid_preds += 1
    
    return exact_matches, similarity_scores, total_invalid_preds


def _calculate_final_metrics(exact_matches: List[float], similarity_scores: List[float], total_invalid_preds: int) -> Dict[str, Any]:
    """Calculate comprehensive final metrics for RoboVQA evaluation."""
    result = {}
    
    # Calculate accuracy metrics
    total_samples = len(exact_matches)
    exact_match_accuracy = sum(exact_matches) / total_samples if total_samples > 0 else 0.0
    
    # Calculate similarity metrics
    avg_similarity_score = sum(similarity_scores) / total_samples if total_samples > 0 else 0.0
    max_similarity_score = max(similarity_scores) if similarity_scores else 0.0
    min_similarity_score = min(similarity_scores) if similarity_scores else 0.0
    
    # Calculate additional statistics
    similarity_std = np.std(similarity_scores) if similarity_scores else 0.0
    
    # Calculate percentage of high similarity matches (threshold-based)
    high_similarity_threshold = 0.8
    high_similarity_count = sum(1 for score in similarity_scores if score >= high_similarity_threshold)
    high_similarity_percentage = (high_similarity_count / total_samples * 100) if total_samples > 0 else 0.0
    
    # Calculate invalid prediction percentage
    invalid_percentage = (total_invalid_preds / total_samples * 100) if total_samples > 0 else 0.0
    
    result['exact_match_accuracy'] = exact_match_accuracy
    result['avg_similarity_score'] = avg_similarity_score
    result['max_similarity_score'] = max_similarity_score
    result['min_similarity_score'] = min_similarity_score
    result['similarity_std'] = similarity_std
    result['high_similarity_percentage'] = high_similarity_percentage
    result['high_similarity_threshold'] = high_similarity_threshold
    result['total_samples'] = total_samples
    result['total_invalid_preds'] = total_invalid_preds
    result['invalid_percentage'] = invalid_percentage
    result['similarity_scores'] = similarity_scores
    
    return result


def _find_shards(dataset: str, disk_root_dir: str) -> List[str]:
    """Find RoboVQA dataset shard files."""
    try:
        # Look for RoboVQA dataset directory pattern
        dataset_dir = f"{disk_root_dir}/test"
        shard_files = glob(f"{dataset_dir}/translated_shard_*")
        tfds_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
        return tfds_shards
    except (IndexError, ValueError):
        print("Cannot identify the directory to the dataset. Skipping this dataset.")
        return []


@dataclass
class DatasetResults:
    """Results from RoboVQA model inference evaluation"""
    all_exact_matches: List[float] = field(default_factory=list)
    all_similarity_scores: List[float] = field(default_factory=list)
    
    total_batches: int = 0
    total_samples: int = 0
    eval_time: float = 0
    total_invalid_predictions: int = 0
    
    # Final metrics
    exact_match_accuracy: float = 0
    avg_similarity_score: float = 0
    max_similarity_score: float = 0
    min_similarity_score: float = 0
    similarity_std: float = 0
    high_similarity_percentage: float = 0
    high_similarity_threshold: float = 0.8
    invalid_percentage: float = 0
    similarity_scores: List[float] = field(default_factory=list)
    total_invalid_predictions: int = 0
    total_samples: int = 0

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }


class RoboVQAInferenceHF:
    """RoboVQA inference class using HuggingFace PaliGemma with Pi0 weight injection"""
    
    def __init__(self, model_id: str = "google/paligemma-3b-pt-224", device: str = None):
        """
        Initialize the RoboVQA inference class.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run inference on (cuda, cpu, etc.)
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.similarity_model = None
        
        print(f"Initializing RoboVQA inference with device: {self.device}")
    
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
    
    def prepare_inputs(self, questions: Union[str, List[str]], images: Union[Image.Image, List[Image.Image], None] = None) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the HuggingFace model.
        
        Args:
            questions: RoboVQA question text (single string or list of strings for batch)
            images: Images (single PIL Image or list of PIL Images, or None for text-only)
            
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
        
        # Format the prompts with the system prompt and questions
        prompts = [f"{ROBOVQA_PROMPT}\n\nQuestion: {question}" for question in questions]
        
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
                max_new_tokens=50,  # Allow more tokens for descriptive VQA answers
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
            questions: List of RoboVQA questions
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
    
    def evaluate_model(self, dataset_dir: str, dataset_name: str = "openx_multi_embodiment", batch_size: int = 8) -> Dict[str, Any]:
        """
        Evaluate the model on RoboVQA dataset.
        
        Args:
            dataset_dir: Directory containing the RoboVQA dataset
            dataset_name: Name of the dataset
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Load similarity model for evaluation
        self.load_similarity_model()
        
        # Find dataset shards
        tfds_shards = _find_shards(dataset_name, dataset_dir)
        if len(tfds_shards) == 0:
            raise ValueError(f"No dataset shards found in {dataset_dir}")
        
        print(f"Found {len(tfds_shards)} dataset shards")
        
        # Create dataloader
        dataloader_obj, dataloader = get_openx_dataloader(
            tfds_shards,
            batch_size=batch_size,
            dataset_name='robot_vqa',
            by_episode=False
        )
        
        print(f"Starting RoboVQA evaluation with {len(dataloader)} batches...")
        dataset_results = DatasetResults()
        start_time = time.perf_counter()
        
        all_exact_matches = []
        all_similarity_scores = []
        total_invalid_preds = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Process each timestep in the batch
            text_obs = batch['text_observation']
            num_timesteps = len(text_obs)
            
            batch_questions = []
            batch_images = []
            batch_labels = []
            
            # Prepare batch data
            for t in range(num_timesteps):
                # Get the question text
                question_text = text_obs[t]
                batch_questions.append(question_text)
                
                # Get the image if present
                if 'image_observation' in batch and batch['image_observation'][t] is not None:
                    image_obs = batch['image_observation'][t]
                    batch_images.append(image_obs)
                else:
                    batch_images.append(None)
                
                # Get the answer label
                batch_labels.append(batch['text_answer'][t])
            
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)} with {len(batch_questions)} samples...")
            
            # Process batch
            batch_outputs = self.process_batch(batch_questions, batch_images)
            
            # Validate outputs and calculate metrics for this batch
            exact_matches, similarity_scores, invalid_preds = _validate_outputs_and_calculate_metrics(
                self.similarity_model, 
                batch_outputs, 
                batch_labels
            )
            
            total_invalid_preds += invalid_preds
            all_exact_matches.extend(exact_matches)
            all_similarity_scores.extend(similarity_scores)
            
            # Update results
            dataset_results.total_batches = batch_idx + 1
            dataset_results.total_samples += len(batch_questions)
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                current_accuracy = sum(all_exact_matches) / len(all_exact_matches) if all_exact_matches else 0.0
                current_similarity = sum(all_similarity_scores) / len(all_similarity_scores) if all_similarity_scores else 0.0
                print(f"Progress: {batch_idx + 1} batches processed. Current accuracy: {current_accuracy:.4f}, Current avg similarity: {current_similarity:.4f}")
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate final metrics
        final_metrics = _calculate_final_metrics(all_exact_matches, all_similarity_scores, total_invalid_preds)
        
        # Update dataset results
        dataset_results.all_exact_matches = final_metrics["exact_matches"]
        dataset_results.all_similarity_scores = final_metrics["similarity_scores"]
        dataset_results.exact_match_accuracy = final_metrics["exact_match_accuracy"]
        dataset_results.avg_similarity_score = final_metrics["avg_similarity_score"]
        dataset_results.max_similarity_score = final_metrics["max_similarity_score"]
        dataset_results.min_similarity_score = final_metrics["min_similarity_score"]
        dataset_results.similarity_std = final_metrics["similarity_std"]
        dataset_results.high_similarity_percentage = final_metrics["high_similarity_percentage"]
        dataset_results.high_similarity_threshold = final_metrics["high_similarity_threshold"]
        dataset_results.total_invalid_predictions = final_metrics["total_invalid_preds"]
        dataset_results.invalid_percentage = final_metrics["invalid_percentage"]
        
        # Calculate evaluation time
        end_time = time.perf_counter()
        dataset_results.eval_time = end_time - start_time
        
        print(f"\nEvaluation completed!")
        print(f"Total samples: {dataset_results.total_samples}")
        print(f"Exact match accuracy: {dataset_results.exact_match_accuracy:.4f}")
        print(f"Average similarity score: {dataset_results.avg_similarity_score:.4f}")
        print(f"High similarity percentage (≥{dataset_results.high_similarity_threshold}): {dataset_results.high_similarity_percentage:.2f}%")
        print(f"Invalid predictions: {dataset_results.total_invalid_predictions} ({dataset_results.invalid_percentage:.2f}%)")
        print(f"Evaluation time: {dataset_results.eval_time:.2f} seconds")
        
        return dataset_results.to_dict()


def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run RoboVQA inference with Pi0 weight-injected HuggingFace PaliGemma"
    )
    
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Directory containing the RoboVQA dataset'
    )
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='openx_multi_embodiment',
        help='Name of the dataset (default: openx_multi_embodiment)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./robovqa_hf_inference_results',
        help='Directory to store inference results (default: ./robovqa_hf_inference_results)'
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
    """Main function to run RoboVQA inference"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be stored in: {args.output_dir}")
    print(f"Reading RoboVQA dataset from: {args.dataset_dir}")
    
    #try:
        # Initialize inference class
    robovqa_inference = RoboVQAInferenceHF(
        model_id=args.model_id,
        device=args.device
    )
    
    # Load the Pi0 weight-injected model
    robovqa_inference.load_model()
    
    # Run evaluation
    results = robovqa_inference.evaluate_model(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size
    )
    
    # Save results
    results_file = os.path.join(args.output_dir, 'robovqa_hf_inference_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print final summary
    print(f"\n=== RoboVQA HuggingFace Inference Results Summary ===")
    print(f"Model: {args.model_id}")
    print(f"Device: {robovqa_inference.device}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Total samples: {results.get('total_samples', 0)}")
    print(f"Exact Match Accuracy: {results.get('exact_match_accuracy', 0):.4f}")
    print(f"Average Similarity Score: {results.get('avg_similarity_score', 0):.4f}")
    print(f"Max Similarity Score: {results.get('max_similarity_score', 0):.4f}")
    print(f"Min Similarity Score: {results.get('min_similarity_score', 0):.4f}")
    print(f"Similarity Std Dev: {results.get('similarity_std', 0):.4f}")
    print(f"High Similarity (≥{results.get('high_similarity_threshold', 0.8)}): {results.get('high_similarity_percentage', 0):.2f}%")
    print(f"Invalid predictions: {results.get('total_invalid_predictions', 0)} ({results.get('invalid_percentage', 0):.2f}%)")
    print(f"Evaluation time: {results.get('eval_time', 0):.2f} seconds")
    print(f"========================================================")
        
    '''except Exception as e:
        print(f"Error during inference: {e}")
        return 1'''
    
    return 0


if __name__ == "__main__":
    exit(main())
