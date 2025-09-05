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
from src.v1.modules.openpi.src.openpi.serving import websocket_policy_server
from src.v1.modules.openpi.src.openpi.training import config as _config

# Import websocket client for model inference
# try:
#     from openpi_client import websocket_client_policy as _websocket_client_policy
#     WEBSOCKET_CLIENT_AVAILABLE = True
# except ImportError as e:
#     print(f"Warning: websocket client not available: {e}")
#     WEBSOCKET_CLIENT_AVAILABLE = False

# Restrict tf to CPU
tf.config.set_visible_devices([], "GPU")


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
    """PIQA inference class for model evaluation using websocket client"""
    
    def __init__(self):
        pass


    def evaluate_model(self, model: pi0.Pi0, dataloader) -> dict:
        """Evaluate the model on PIQA dataset using websocket client
        
        Args:
            model: The pi0 model to evaluate
            dataloader: Data loader for PIQA dataset

        Returns:
            Dictionary containing evaluation results
        """
        # if not WEBSOCKET_CLIENT_AVAILABLE:
        #     print("Warning: Websocket client not available, running analysis mode instead")
        #     # Fall back to dataset analysis
        #     return self.analyze_dataset_from_loader(dataloader)
            
        counter = 0
        dataset_results = DatasetResults()

        start_time = time.perf_counter()

        for batch in dataloader:
            # Process entire batch at once
            actual_batch_size = len(batch['goal'])  # PIQA dataloader returns goal, sol1, sol2, etc.
            
            # Format PIQA prompts for each sample in the batch
            text_prompts = batch['question']
            labels = batch['label']
            predictions = []
            
            # 
            for i in range(actual_batch_size):

                
                # For each sample, prepare element for client inference
                try:
                    element = {
                        "prompt": text_prompts[i], 
                    }

                    
                    #element = jax.tree.map(lambda x: x, element)
                    element = PiqaInputs()(element)
                    # tokenize the prompt
                    element = TokenizePrompt(_tokenizer.PaligemmaTokenizer(200))(element) # increase max length for piqa
                    # Make a batch and convert to jax.Array.
                    element = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], element)
                    # Use the Observation from the pi0 module to ensure type compatibility
                    observation = pi0._model.Observation.from_dict(element)
                    # Query pi0 model using websocket client
                    response = model.vlm_autoregress(rng=jax.random.PRNGKey(0), observation=observation)

                    # Process the response to extract A/B choice
                    generated_text = str(response)
                    
                    # Extract binary choice from generated text
                    # For now, use a simple extraction since we don't have label/correct_solution here
                    prediction = self.process_single_text_response(generated_text, labels[i], batch['correct_solution'][i])
                    predictions.append(generated_text)
                    
                except Exception as e:
                    print(f"Error during inference for sample {i}: {e}")
                    # Fallback to random prediction
                    predictions.append(-1) # -1 indicates invalid prediction
            
            print(f"Batch {counter} processed {actual_batch_size} samples")
            print(f"Sample prompt: {text_prompts[0][:200]}...")  # Show first 200 chars of first prompt
            
            counter += 1

            # Get ground truth labels from batch
            gt_labels = np.array(labels)
            processed_responses = np.array(predictions)
            
            print(f'Ground truth labels: {gt_labels}')
            print(f'Predicted labels: {processed_responses}')
            
            # Calculate metrics
            emr = get_exact_match_rate(processed_responses, gt_labels)
            action_space = [0, 1]  # Binary classification
            
            # Calculate metrics counts
            total_tp, total_fp, total_fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
                processed_responses, gt_labels, action_space
            )

            # Calculate all metrics
            micro_precision = get_micro_precision_from_counts(total_tp, total_fp)
            micro_recall = get_micro_recall_from_counts(total_tp, total_fn)
            micro_f1 = get_micro_f1(micro_precision, micro_recall)
            
            print(f"Micro Precision: {micro_precision}, Micro Recall: {micro_recall}, Micro F1: {micro_f1}")
            
            # Store results for this batch
            dataset_results.all_preds.extend(processed_responses.tolist() if hasattr(processed_responses, 'tolist') else processed_responses)
            dataset_results.all_gt.extend(gt_labels.tolist())
            dataset_results.total_invalid_predictions += int(invalid_fp)
            dataset_results.total_batches = counter
            dataset_results.total_timesteps += len(processed_responses)
            dataset_results.total_emr += emr
            dataset_results.total_micro_precision += micro_precision
            dataset_results.total_micro_recall += micro_recall
            dataset_results.total_micro_f1 += micro_f1

            # Memory management
            gc.collect()
            print(f"Processed {counter} batches, cleared memory")

            # Uncomment to stop after a few batches for testing
            if counter == 2:
                break

        end_time = time.perf_counter()
        eval_duration = end_time - start_time
        dataset_results.eval_time = eval_duration
        dataset_results.avg_emr = dataset_results.total_emr / dataset_results.total_timesteps if dataset_results.total_timesteps > 0 else 0
        dataset_results.invalid_predictions_percentage = dataset_results.total_invalid_predictions / dataset_results.total_timesteps * 100 if dataset_results.total_timesteps > 0 else 0
        dataset_results.avg_micro_precision = dataset_results.total_micro_precision / dataset_results.total_timesteps if dataset_results.total_timesteps > 0 else 0
        dataset_results.avg_micro_recall = dataset_results.total_micro_recall / dataset_results.total_timesteps if dataset_results.total_timesteps > 0 else 0
        dataset_results.avg_micro_f1 = dataset_results.total_micro_f1 / dataset_results.total_timesteps if dataset_results.total_timesteps > 0 else 0

        return dataset_results.to_dict()


    def process_single_text_response(self, generated_text: str, label: int, correct_solution_text: str) -> int:
        """Process a single generated text response to extract binary choice
        
        Args:
            generated_text: Generated text response from the model
            label: Ground truth label (0 or 1)
            correct_solution_text: The correct solution text for comparison

        Returns:
            Binary prediction (0 or 1, -1 for invalid)
        """
        # If the generated text matches the correct solution text, return the label
        if correct_solution_text.strip().lower() in generated_text.strip().lower():
            return label
        # else if the generated text is a only numeric 0 or 1, return that
        elif generated_text.strip() == '0':
            return 0
        elif generated_text.strip() == '1':
            return 1
        else:
            # If ambiguous, return invalid prediction
            return -1


    def analyze_dataset_from_loader(self, dataloader) -> dict:
        """Analyze dataset when model inference is not available"""
        print("Running dataset analysis mode...")
        results = {
            "mode": "analysis_only",
            "total_samples": 0,
            "sample_examples": []
        }
        
        counter = 0
        for batch in dataloader:
            batch_size = len(batch['goal'])
            results["total_samples"] += batch_size
            
            # Store a few examples
            if counter < 5:  # Store first 5 batches as examples
                for i in range(min(batch_size, 3)):  # Up to 3 samples per batch
                    example = {
                        "goal": batch['goal'][i],
                        "solution_1": batch['sol1'][i],
                        "solution_2": batch['sol2'][i],
                        "correct_answer": batch['correct_solution'][i]
                    }
                    results["sample_examples"].append(example)
            
            counter += 1
            if counter == 10:  # Limit analysis to first 10 batches
                break
        
        return results


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
        '--output_dir',
        type=str,
        default='./piqa_inference_results',
        help='Directory to store inference results (default: ./piqa_inference_results)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1)'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (default: all samples)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['analyze', 'inference'],
        default='inference',
        help='Mode: analyze dataset structure or run model inference (default: inference)'
    )
    
    # Websocket client arguments for inference mode
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host address for the model server (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for the model server (default: 8000)'
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
    
    if args.mode == 'analyze':
        # Initialize analyzer for dataset analysis
        analyzer = PIQAInference()
        
        try:
            # Use PIQA dataloader for analysis
            dataset_obj, dataloader = get_piqa_test_dataloader(
                test_dir=args.dataset_dir,
                batch_size=args.batch_size
            )
            
            # Run analysis using dataloader
            results = analyzer.analyze_dataset_from_loader(dataloader)
            
            # Print results to console
            print("\n=== PIQA Dataset Analysis Results ===")
            print(f"Mode: {results.get('mode', 'analysis')}")
            print(f"Total samples: {results.get('total_samples', 0)}")
            
            if 'sample_examples' in results:
                print(f"\nSample examples ({len(results['sample_examples'])}):")
                for i, example in enumerate(results['sample_examples'][:5]):
                    print(f"\n  Example {i+1}:")
                    print(f"    Goal: {example['goal']}")
                    print(f"    Solution 1: {example['solution_1']}")
                    print(f"    Solution 2: {example['solution_2']}")
                    print(f"    Correct Answer: {example['correct_answer']}")
            
            # Save results to JSON file
            results_file = os.path.join(args.output_dir, 'piqa_analysis_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"\nDetailed results saved to: {results_file}")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return 1
    
    elif args.mode == 'inference':
        # Initialize websocket client for model evaluation
        # if not WEBSOCKET_CLIENT_AVAILABLE:
        #     print("Error: websocket client is required for inference mode")
        #     return 1
        config = pi0.Pi0Config(action_horizon=1)
        #key = jax.random.key(0)
        model = config.load(_model.restore_params(download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params")))
        # from src.v1.modules.openpi.scripts.serve_policy import Args
        # policy_args = Args()
        # policy = create_policy(policy_args)
        print('Model loaded')
        try:
            # Create websocket client to connect to model server
            # client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
            # print(f'Connected to model server at {args.host}:{args.port}')
            piqa_inference = PIQAInference()
            
            # Get dataset stats (for PIQA, this is minimal since it's text-only)
            dataset_stats = {}  # Not needed for websocket client
            
            # Create PIQA dataloader using the piqa_dataloader module
            dataset_obj, dataloader = get_piqa_test_dataloader(
                test_dir=args.dataset_dir, 
                batch_size=args.batch_size
            )
            
            print(f"Created dataloader with {len(dataset_obj)} samples")
            
            # Run inference
            results = piqa_inference.evaluate_model(
                model, dataloader
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
