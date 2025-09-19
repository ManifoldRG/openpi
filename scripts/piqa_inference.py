import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import gc

import torch

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
# Add the OpenPI src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# Add the openpi source directory to sys.path (same as pi0.py does)
openpi_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if openpi_src_path not in sys.path:
    sys.path.insert(0, openpi_src_path)

from src.eval_utils import (get_exact_match_rate,
                            calculate_tp_fp_fn_counts,
                            get_micro_precision_from_counts, 
                            get_micro_recall_from_counts, 
                            get_micro_f1)

# Import PIQA dataloader
from src.data_utils.piqa_dataloader import get_piqa_test_dataloader
# import the pi0 policy
from src.v1.modules.openpi.src.openpi.models import pi0
from src.v1.modules.openpi.src.openpi.models import model as _model
from src.v1.modules.openpi.src.openpi.transforms import TokenizePrompt, _tokenizer
from src.v1.modules.openpi.src.openpi.policies.piqa_policy import PiqaInputs
from src.v1.modules.openpi.scripts.serve_policy import create_policy
from src.v1.modules.openpi.src.openpi.shared import download
from src.v1.modules.openpi.src.openpi.policies import policy as _policy
from src.v1.modules.openpi.src.openpi.policies import policy_config as _policy_config
from src.v1.modules.openpi.src.openpi.training import config as _config
from replace_paligemma_weights_wpi0 import replace_paligemma_weights_with_pi0_weights
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


# Restrict tf to CPU
#tf.config.set_visible_devices([], "GPU")


@dataclass
class DatasetResults:
    """Results from PIQA model inference evaluation"""
    all_preds: list[list[float]] = field(default_factory=list)
    all_gt: list[list[float]] = field(default_factory=list)
    
    total_batches: int = 0
    total_timesteps: int = 0
    eval_time: float = 0
    total_invalid_predictions: int = 0
    invalid_predictions_percentage: float = 0
    total_emr: float = 0
    total_micro_precision: float = 0
    total_micro_recall: float = 0
    total_micro_f1: float = 0
    avg_emr: float = 0
    avg_micro_precision: float = 0
    avg_micro_recall: float = 0
    avg_micro_f1: float = 0
    total_clipped_emr: float = 0
    total_clipped_micro_precision: float = 0
    total_clipped_micro_recall: float = 0
    total_clipped_micro_f1: float = 0
    avg_clipped_emr: float = 0
    avg_clipped_micro_precision: float = 0
    avg_clipped_micro_recall: float = 0
    avg_clipped_micro_f1: float = 0
    total_micro_precision_without_invalids: float = 0
    total_micro_f1_without_invalids: float = 0
    avg_micro_precision_without_invalids: float = 0
    avg_micro_f1_without_invalids: float = 0

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }


class PIQAInference:
    """PIQA inference class for model evaluation"""
    
    def __init__(self):
        pass


    def evaluate_model(self, processor: AutoProcessor, model: PaliGemmaForConditionalGeneration, dataloader) -> dict:
        """Evaluate the model on PIQA dataset

        Args:
            processor: HuggingFace processor for tokenization
            model: PaliGemma model with Pi0 weights
            dataloader: DataLoader for PIQA dataset

        Returns:
            Dictionary containing evaluation results
        """  
        counter = 0
        dataset_results = DatasetResults()

        start_time = time.perf_counter()

        for batch in dataloader:
            # Process entire batch at once
            actual_batch_size = len(batch['goal'])  # PIQA dataloader returns goal, sol1, sol2, etc.
            
            # Format PIQA prompts for each sample in the batch
            predictions = np.zeros((actual_batch_size,), dtype=int) - 1  # Initialize with -1 (invalid)

            # Batch processing
            # create a dummy image for each sample in the batch
            dummy_images = tf.zeros((actual_batch_size, 3, 224, 224), dtype=tf.float32)
            try:
                inputs = processor(text=batch, images=dummy_images,
                  padding="longest", do_convert_rgb=True, return_tensors="pt")
                inputs = inputs.to(dtype=model.dtype)
                with torch.no_grad():
                    output = model.generate(**inputs, max_length=496)
                generated_texts = [processor.decode(out, skip_special_tokens=True) for out in output]
                predictions = self.process_batch_text_response(generated_texts)

            except Exception as e:
                print(f"Error during inference for batch {counter}: {e}")
                # throw exception to avoid silent failures
                raise e
            
            print(f"Batch {counter} processed {actual_batch_size} samples")
            print(f"Sample prompt: {batch['question'][:5]}...")
            
            counter += 1

            # Get ground truth labels from batch
            gt_labels = np.array(batch['label'])
            
            print(f'Ground truth labels: {gt_labels}')
            print(f'Predicted labels: {predictions}')

            # Calculate metrics
            emr = get_exact_match_rate(predictions, gt_labels)
            action_space = [0, 1]  # Binary classification
            
            # Calculate metrics counts
            total_tp, total_fp, total_fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
                predictions, gt_labels, action_space
            )

            # Calculate all metrics
            micro_precision = get_micro_precision_from_counts(total_tp, total_fp)
            micro_recall = get_micro_recall_from_counts(total_tp, total_fn)
            micro_f1 = get_micro_f1(micro_precision, micro_recall)
            
            print(f"Micro Precision: {micro_precision}, Micro Recall: {micro_recall}, Micro F1: {micro_f1}")
            
            # Store results for this batch
            dataset_results.all_preds.extend(predictions.tolist() if hasattr(predictions, 'tolist') else predictions)
            dataset_results.all_gt.extend(gt_labels.tolist())
            dataset_results.total_invalid_predictions += int(invalid_fp)
            dataset_results.total_batches = counter
            dataset_results.total_timesteps += len(predictions)
            dataset_results.total_emr += emr
            dataset_results.total_micro_precision += micro_precision
            dataset_results.total_micro_recall += micro_recall
            dataset_results.total_micro_f1 += micro_f1

            # Memory management
            gc.collect()
            print(f"Processed {counter} batches, cleared memory")


        end_time = time.perf_counter()
        eval_duration = end_time - start_time
        dataset_results.eval_time = eval_duration
        dataset_results.avg_emr = dataset_results.total_emr / dataset_results.total_timesteps if dataset_results.total_timesteps > 0 else 0
        dataset_results.invalid_predictions_percentage = dataset_results.total_invalid_predictions / dataset_results.total_timesteps * 100 if dataset_results.total_timesteps > 0 else 0
        dataset_results.avg_micro_precision = dataset_results.total_micro_precision / dataset_results.total_timesteps if dataset_results.total_timesteps > 0 else 0
        dataset_results.avg_micro_recall = dataset_results.total_micro_recall / dataset_results.total_timesteps if dataset_results.total_timesteps > 0 else 0
        dataset_results.avg_micro_f1 = dataset_results.total_micro_f1 / dataset_results.total_timesteps if dataset_results.total_timesteps > 0 else 0

        return dataset_results.to_dict()

    def prepare_observation(self, element: dict, max_token_len:int=300) -> dict:
        """Prepare observation dictionary for model inference
        
        Args:
            element: Dictionary containing input data (e.g., prompt)
            max_token_len: Maximum token length for prompt tokenization

        Returns:
            Prepared observation dictionary
        """
        # Ensure prompt is a list for consistent processing
        if isinstance(element['prompt'], str):
            element['prompt'] = [element['prompt']]
        # If it's already a list but not numpy array, keep as is
        elif isinstance(element['prompt'], list):
            pass
        else:
            # Convert other types to list
            element['prompt'] = list(element['prompt'])

        #element = jax.tree.map(lambda x: x, element)
        element = PiqaInputs()(element)
        # tokenize the prompt
        element = TokenizePrompt(_tokenizer.PaligemmaTokenizer(max_token_len))(element)
        # convert to jax.Array.
        element = jax.tree.map(lambda x: jnp.asarray(x)[...], element)
        # Use the Observation from the pi0 module to ensure type compatibility
        observation = pi0._model.Observation.from_dict(element)
        return observation

    def process_batch_text_response(self, generated_texts: list[str]) -> int:
        """Process a batch of generated texts response to extract binary choice
        
        Args:
            generated_texts: Generated text responses from the model

        Returns:
            Binary predictions (0 or 1, -1 for invalid)
        """
        # compare with labels to determine correctness. Generated text should either be 0 or 1, and must match the label
        def safe_int(x):
            try:
                return int(x)
            except ValueError:
                return -1
        vec_safe_int = np.vectorize(safe_int)
        # Convert generated texts to integers safely
        generated_texts = vec_safe_int(generated_texts)
        return generated_texts


def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run inference on PIQA dataset with model evaluation"
    )
    
    parser.add_argument(
        '--dataset_dir',
        type=str,
        #default= '../../../processed_datasets/piqa/test/', 
        default = 'src/v1/processed_datasets/piqa/test/',
        help='Directory containing the PIQA dataset (default: ../../../processed_datasets/piqa/test/)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run the model on (default: cuda if available else cpu)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./piqa_inference_results',
        help='Directory to store inference results (default: ./piqa_inference_results)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for inference (default: 32)'
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
    """Main function to run PIQA inference or analysis"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be stored in: {args.output_dir}")
    print(f"Reading PIQA dataset from: {args.dataset_dir}")

    processor, model = replace_paligemma_weights_with_pi0_weights(args.device)

    print('Model loaded')
    try:
        piqa_inference = PIQAInference()
        
        # Create PIQA dataloader using the piqa_dataloader module
        dataset_obj, dataloader = get_piqa_test_dataloader(
            test_dir=args.dataset_dir, 
            batch_size=args.batch_size
        )
        
        print(f"Created dataloader with {len(dataset_obj)} samples")
        
        # Run inference
        results = piqa_inference.evaluate_model(
            processor, model, dataloader
        )
        
        results_file = os.path.join(args.output_dir, 'piqa_inference_results.json')
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"Inference results saved to: {results_file}")
        
        # Print summary
        print(f"\n=== PIQA Inference Results Summary ===")
        print(f"Total timesteps: {results.get('total_timesteps', 0)}")
        print(f"Average EMR: {results.get('avg_emr', 0):.3f}")
        print(f"Average Micro F1: {results.get('avg_micro_f1', 0):.3f}")
        print(f"Average Micro Precision: {results.get('avg_micro_precision', 0):.3f}")
        print(f"Average Micro Recall: {results.get('avg_micro_recall', 0):.3f}")
        print(f"Evaluation time: {results.get('eval_time', 0):.2f} seconds")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
