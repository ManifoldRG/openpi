#!/usr/bin/env python3
"""
SQA3D Inference with Weight-Injected HuggingFace PaliGemma

This script evaluates the Pi0 weight-injected HuggingFace PaliGemma model on the SQA3D dataset.
It uses the weight injection module to create a model with Pi0 weights, then evaluates it
using SQA3D-specific evaluation metrics including text similarity scoring and exact match accuracy.

Usage:
    python sqa3d_hf_inference.py --dataset_dir /path/to/sqa3d/test --output_dir ./results
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
import numpy as np
import torch
from PIL import Image
import gc

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

# Import HuggingFace transformers
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# Import evaluation utilities and SQA3D-specific modules
from src.data_utils.sqa3d_dataloader import get_sqa3d_dataloader
from definitions.sqa3d_prompt import SQA3DDefinitions

# Import the Pi0 weight injector
from src.v1.modules.openpi.scripts.pi0_weight_injector import get_pi0_injected_model

# Import similarity model for evaluation
from sentence_transformers import SentenceTransformer, util


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
    """Validate outputs and calculate text similarity metrics for SQA3D."""
    exact_matches = []
    similarity_scores = []
    total_invalid_preds = 0
    normalized_preds = []
    
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
            
            normalized_preds.append(normalized_output)
        else:
            # Invalid output - assign worst possible scores
            exact_matches.append(0.0)
            similarity_scores.append(0.0)
            total_invalid_preds += 1
            normalized_preds.append("")
    
    return exact_matches, similarity_scores, total_invalid_preds, normalized_preds


def _calculate_final_metrics(exact_matches: List[float], similarity_scores: List[float], total_invalid_preds: int) -> Dict[str, Any]:
    """Calculate comprehensive final metrics for SQA3D evaluation."""
    result = {}
    
    # Calculate accuracy metrics
    total_samples = len(exact_matches)
    exact_match_accuracy = sum(exact_matches) / total_samples if total_samples > 0 else 0.0
    exact_match_accuracy_without_invalids = sum(exact_matches) / (total_samples - total_invalid_preds) if total_samples - total_invalid_preds > 0 else 0.0
    
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
    
    result['exact_match_rate'] = exact_match_accuracy
    result['exact_match_rate_without_invalids'] = exact_match_accuracy_without_invalids
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


def _find_data_files(dataset_dir: str) -> Dict[str, str]:
    """Find SQA3D data files (questions, annotations, images)"""
    test_dir = Path(dataset_dir)
    
    # Look for standard test files
    questions_file = test_dir / "v1_balanced_questions_test_scannetv2.json"
    annotations_file = test_dir / "v1_balanced_sqa_annotations_test_scannetv2.json"
    
    if not questions_file.exists():
        raise FileNotFoundError(f"Test questions file not found: {questions_file}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Test annotations file not found: {annotations_file}")
    
    # Images should be in scene subdirectories
    images_dir = test_dir
    return {
        "questions_file": str(questions_file),
        "annotations_file": str(annotations_file),
        "images_dir": str(images_dir)
    }


@dataclass
class DatasetResults:
    """Results from SQA3D model inference evaluation"""
    all_exact_matches: List[float] = field(default_factory=list)
    all_similarity_scores: List[float] = field(default_factory=list)
    normalized_preds: List[str] = field(default_factory=list)
    all_original_outputs: List[str] = field(default_factory=list)
    all_labels: List[str] = field(default_factory=list)
    
    total_batches: int = 0
    total_samples: int = 0
    eval_time: float = 0
    total_invalid_predictions: int = 0
    
    # Final metrics
    exact_match_rate: float = 0
    exact_match_rate_without_invalids: float = 0
    avg_similarity_score: float = 0
    max_similarity_score: float = 0
    min_similarity_score: float = 0
    similarity_std: float = 0
    high_similarity_percentage: float = 0
    high_similarity_threshold: float = 0.8
    invalid_percentage: float = 0
    similarity_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }


class SQA3DInferenceHF:
    """SQA3D inference class using HuggingFace PaliGemma with Pi0 weight injection"""
    
    def __init__(self, model_id: str = "google/paligemma-3b-pt-224", device: str = None):
        """
        Initialize the SQA3D inference class.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run inference on (cuda, cpu, etc.)
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.similarity_model = None
        
        print(f"Initializing SQA3D inference with device: {self.device}")
    
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
            questions: SQA3D question text (single string or list of strings for batch)
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
        prompts = [f"{SQA3DDefinitions.SYSTEM_PROMPT}\n\n{question}" for question in questions]
        
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
            questions: List of SQA3D questions
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
    
    def evaluate_model(self, dataset_dir: str, batch_size: int = 8) -> Dict[str, Any]:
        """
        Evaluate the model on SQA3D dataset.
        
        Args:
            dataset_dir: Directory containing the SQA3D dataset
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Load similarity model for evaluation
        self.load_similarity_model()
        
        # Find dataset files
        data_files = _find_data_files(dataset_dir)
        print(f"Found SQA3D data files:")
        print(f"  Questions: {data_files['questions_file']}")
        print(f"  Annotations: {data_files['annotations_file']}")
        print(f"  Images: {data_files['images_dir']}")
        
        # Create dataloader
        dataset_obj, dataloader = get_sqa3d_dataloader(
            questions_file=data_files['questions_file'],
            annotations_file=data_files['annotations_file'],
            images_dir=data_files['images_dir'],
            batch_size=batch_size
        )
        
        print(f"Created dataloader with {len(dataset_obj)} samples")
        print(f"Starting SQA3D evaluation with {len(dataloader)} batches...")
        
        dataset_results = DatasetResults()
        start_time = time.perf_counter()
        
        all_exact_matches = []
        all_similarity_scores = []
        all_normalized_preds = []
        all_original_outputs = []
        all_labels = []
        total_invalid_preds = 0
        
        for batch_idx, batch in enumerate(dataloader):
            questions = batch['question']
            answers = batch['answer']
            scene_images = batch.get('scene_image', [None] * len(questions))
            
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)} with {len(questions)} samples...")
            
            # Process batch
            batch_outputs = self.process_batch(questions, scene_images)
            
            # Validate outputs and calculate metrics for this batch
            exact_matches, similarity_scores, invalid_preds, normalized_preds = _validate_outputs_and_calculate_metrics(
                self.similarity_model, 
                batch_outputs, 
                answers
            )
            
            total_invalid_preds += invalid_preds
            all_exact_matches.extend(exact_matches)
            all_similarity_scores.extend(similarity_scores)
            all_normalized_preds.extend(normalized_preds)
            all_original_outputs.extend(batch_outputs)
            all_labels.extend(answers)
            
            # Update results
            dataset_results.total_batches = batch_idx + 1
            dataset_results.total_samples += len(questions)
            
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
        dataset_results.all_exact_matches = all_exact_matches
        dataset_results.all_similarity_scores = final_metrics["similarity_scores"]
        dataset_results.normalized_preds = all_normalized_preds
        dataset_results.all_original_outputs = all_original_outputs
        dataset_results.all_labels = all_labels
        dataset_results.exact_match_rate = final_metrics["exact_match_rate"]
        dataset_results.exact_match_rate_without_invalids = final_metrics["exact_match_rate_without_invalids"]
        dataset_results.avg_similarity_score = final_metrics["avg_similarity_score"]
        dataset_results.max_similarity_score = final_metrics["max_similarity_score"]
        dataset_results.min_similarity_score = final_metrics["min_similarity_score"]
        dataset_results.similarity_std = final_metrics["similarity_std"]
        dataset_results.high_similarity_percentage = final_metrics["high_similarity_percentage"]
        dataset_results.high_similarity_threshold = final_metrics["high_similarity_threshold"]
        dataset_results.total_invalid_predictions = final_metrics["total_invalid_preds"]
        dataset_results.invalid_percentage = final_metrics["invalid_percentage"]
        dataset_results.similarity_scores = final_metrics["similarity_scores"]
        
        # Calculate evaluation time
        end_time = time.perf_counter()
        dataset_results.eval_time = end_time - start_time
        
        print(f"\nEvaluation completed!")
        print(f"Total samples: {dataset_results.total_samples}")
        print(f"Exact match rate: {dataset_results.exact_match_rate:.4f}")
        print(f"Exact match rate (without invalids): {dataset_results.exact_match_rate_without_invalids:.4f}")
        print(f"Average similarity score: {dataset_results.avg_similarity_score:.4f}")
        print(f"High similarity percentage (≥{dataset_results.high_similarity_threshold}): {dataset_results.high_similarity_percentage:.2f}%")
        print(f"Invalid predictions: {dataset_results.total_invalid_predictions} ({dataset_results.invalid_percentage:.2f}%)")
        print(f"Evaluation time: {dataset_results.eval_time:.2f} seconds")
        
        return dataset_results.to_dict()


def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run SQA3D inference with Pi0 weight-injected HuggingFace PaliGemma"
    )
    
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Directory containing the SQA3D test dataset'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./sqa3d_hf_inference_results',
        help='Directory to store inference results (default: ./sqa3d_hf_inference_results)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
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
    """Main function to run SQA3D inference"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be stored in: {args.output_dir}")
    print(f"Reading SQA3D dataset from: {args.dataset_dir}")
    
    try:
        # Initialize inference class
        sqa3d_inference = SQA3DInferenceHF(
            model_id=args.model_id,
            device=args.device
        )
        
        # Load the Pi0 weight-injected model
        sqa3d_inference.load_model()
        
        # Run evaluation
        results = sqa3d_inference.evaluate_model(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size
        )
        
        # Save results
        results_file = os.path.join(args.output_dir, 'sqa3d_pi0_hf_inference_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to: {results_file}")
        
        # Print final summary
        print(f"\n=== SQA3D HuggingFace Inference Results Summary ===")
        print(f"Model: {args.model_id}")
        print(f"Device: {sqa3d_inference.device}")
        print(f"Total samples: {results.get('total_samples', 0)}")
        print(f"Exact Match Rate: {results.get('exact_match_rate', 0):.4f}")
        print(f"Exact Match Rate (without invalids): {results.get('exact_match_rate_without_invalids', 0):.4f}")
        print(f"Average Similarity Score: {results.get('avg_similarity_score', 0):.4f}")
        print(f"Max Similarity Score: {results.get('max_similarity_score', 0):.4f}")
        print(f"Min Similarity Score: {results.get('min_similarity_score', 0):.4f}")
        print(f"Similarity Std Dev: {results.get('similarity_std', 0):.4f}")
        print(f"High Similarity (≥{results.get('high_similarity_threshold', 0.8)}): {results.get('high_similarity_percentage', 0):.2f}%")
        print(f"Invalid predictions: {results.get('total_invalid_predictions', 0)} ({results.get('invalid_percentage', 0):.2f}%)")
        print(f"Evaluation time: {results.get('eval_time', 0):.2f} seconds")
        print(f"=======================================================")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
