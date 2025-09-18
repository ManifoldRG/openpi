#!/usr/bin/env python3
"""
PaliGemma Weight Comparison Test Suite

This script compares the text generation capabilities of base PaliGemma weights
versus Pi0-trained PaliGemma weights using the HuggingFace transformers
implementation. The test isolates weight-level differences by using identical
inference pipelines.

Test Methodology:
1. Load base PaliGemma weights and Pi0-trained weights from checkpoints
2. Inject embedding tables into separate transformers model instances
3. Test both models on 25 diverse COCO validation images
4. Use three different prompt types to evaluate text generation robustness
5. Generate comparative HTML report with visual analysis

Key Features:
- Weight injection verification to ensure proper loading
- Multiple prompt types: basic caption, detailed description, creative pun
- Comprehensive HTML report with image visualization
- Statistical analysis of text generation patterns
- Detection of repetitive loops and generation failures
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import torch
import requests
from io import BytesIO

# JAX imports for loading weights
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Transformers imports
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

# OpenPI imports
from openpi.models import model as _model
from openpi.models import pi0
from openpi.training.weight_loaders import PaliGemmaWeightLoader
from openpi.shared import download

# Test dataset: 25 diverse COCO validation images
# Selected for variety across object categories, scenes, and visual complexity
COCO_TEST_IMAGES = [
    {"id": "000000397133", "url": "http://images.cocodataset.org/val2017/000000397133.jpg"},
    {"id": "000000037777", "url": "http://images.cocodataset.org/val2017/000000037777.jpg"},
    {"id": "000000252219", "url": "http://images.cocodataset.org/val2017/000000252219.jpg"},
    {"id": "000000087038", "url": "http://images.cocodataset.org/val2017/000000087038.jpg"},
    {"id": "000000174482", "url": "http://images.cocodataset.org/val2017/000000174482.jpg"},
    {"id": "000000403385", "url": "http://images.cocodataset.org/val2017/000000403385.jpg"},
    {"id": "000000006818", "url": "http://images.cocodataset.org/val2017/000000006818.jpg"},
    {"id": "000000480985", "url": "http://images.cocodataset.org/val2017/000000480985.jpg"},
    {"id": "000000458755", "url": "http://images.cocodataset.org/val2017/000000458755.jpg"},
    {"id": "000000331352", "url": "http://images.cocodataset.org/val2017/000000331352.jpg"},
    {"id": "000000579158", "url": "http://images.cocodataset.org/val2017/000000579158.jpg"},
    {"id": "000000578922", "url": "http://images.cocodataset.org/val2017/000000578922.jpg"},
    {"id": "000000472375", "url": "http://images.cocodataset.org/val2017/000000472375.jpg"},
    {"id": "000000013177", "url": "http://images.cocodataset.org/val2017/000000013177.jpg"},
    {"id": "000000544519", "url": "http://images.cocodataset.org/val2017/000000544519.jpg"},
    {"id": "000000334555", "url": "http://images.cocodataset.org/val2017/000000334555.jpg"},
    {"id": "000000502136", "url": "http://images.cocodataset.org/val2017/000000502136.jpg"},
    {"id": "000000520077", "url": "http://images.cocodataset.org/val2017/000000520077.jpg"},
    {"id": "000000184791", "url": "http://images.cocodataset.org/val2017/000000184791.jpg"},
    {"id": "000000050326", "url": "http://images.cocodataset.org/val2017/000000050326.jpg"},
    {"id": "000000315257", "url": "http://images.cocodataset.org/val2017/000000315257.jpg"},
    {"id": "000000226111", "url": "http://images.cocodataset.org/val2017/000000226111.jpg"},
    {"id": "000000017627", "url": "http://images.cocodataset.org/val2017/000000017627.jpg"},
    {"id": "000000025560", "url": "http://images.cocodataset.org/val2017/000000025560.jpg"},
    {"id": "000000036494", "url": "http://images.cocodataset.org/val2017/000000036494.jpg"}
]

# Three distinct prompt types for comprehensive evaluation
PROMPT_TYPES = {
    "basic_caption": {
        "template": "<image>caption en",
        "description": "Basic image captioning task",
        "max_tokens": 25
    },
    "detailed_description": {
        "template": "<image>Provide a detailed description of all objects, people, and activities visible in this image. Include spatial relationships, colors, and any notable details in a comprehensive paragraph.",
        "description": "Detailed object and scene description",
        "max_tokens": 75
    },
    "creative_pun": {
        "template": "<image>Create a clever pun or wordplay based on what you see in this image. Be creative and humorous.",
        "description": "Creative text generation requiring wordplay",
        "max_tokens": 40
    }
}

def verify_weight_loading():
    """
    Verify that base and Pi0 weights are actually different.

    This function loads both weight sets and compares their embedding tables
    to ensure we're not accidentally loading identical weights, which would
    invalidate the comparison test.

    Returns:
        bool: True if weights are different, False if identical
    """
    print("STEP 1: Verifying weight loading...")

    # Load base PaliGemma weights from official checkpoint
    print("  Loading base PaliGemma weights...")
    weight_loader = PaliGemmaWeightLoader()
    rng = jax.random.key(42)
    config = pi0.Pi0Config()
    dummy_model = config.create(rng)
    dummy_params = nnx.state(dummy_model).to_pure_dict()
    base_params = weight_loader.load(dummy_params)
    base_weights = base_params.get("PaliGemma", {})

    # Load Pi0-trained weights from checkpoint
    print("  Loading Pi0-trained weights...")
    pi0_params = _model.restore_params(
        download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params")
    )
    pi0_weights = pi0_params.get("PaliGemma", {})

    # Compare embedding tables as a representative sample
    base_embeddings = base_weights['llm']['embedder']['input_embedding']
    pi0_embeddings = pi0_weights['llm']['embedder']['input_embedding']

    print(f"  Base embeddings shape: {base_embeddings.shape}")
    print(f"  Pi0 embeddings shape: {pi0_embeddings.shape}")

    # Statistical comparison
    base_mean = float(np.mean(base_embeddings))
    pi0_mean = float(np.mean(pi0_embeddings))
    base_std = float(np.std(base_embeddings))
    pi0_std = float(np.std(pi0_embeddings))

    print(f"  Base embeddings - mean: {base_mean:.6f}, std: {base_std:.6f}")
    print(f"  Pi0 embeddings  - mean: {pi0_mean:.6f}, std: {pi0_std:.6f}")

    # Numerical difference check
    are_identical = np.allclose(base_embeddings, pi0_embeddings, rtol=1e-6)
    max_diff = float(np.max(np.abs(base_embeddings - pi0_embeddings)))

    print(f"  Embeddings identical: {are_identical}")
    print(f"  Maximum difference: {max_diff:.6f}")

    if are_identical:
        print("  WARNING: Weights appear identical - test would be invalid")
        return False
    else:
        print("  PASS: Weights are different - proceeding with test")
        return True

def inject_embedding_with_verification(model, jax_embeddings, name="unknown"):
    """
    Inject JAX embedding weights into a PyTorch transformers model.

    This function replaces the embedding table in a PyTorch PaliGemma model
    with weights loaded from JAX checkpoints. It handles vocabulary size
    mismatches and verifies successful injection.

    Args:
        model: PyTorch PaliGemmaForConditionalGeneration model
        jax_embeddings: JAX embedding array from checkpoint
        name: Descriptive name for logging

    Returns:
        bool: True if injection successful, False otherwise
    """
    print(f"\nSTEP 2: Injecting {name} embeddings...")

    state_dict = model.state_dict()

    # Find embedding parameter
    embedding_param = None
    embedding_name = None

    for param_name, param in state_dict.items():
        if 'embed_tokens.weight' in param_name:
            embedding_param = param
            embedding_name = param_name
            break

    if embedding_param is None:
        print("  Could not find embedding parameter")
        return False

    # Store original for verification
    original_embedding = embedding_param.clone()

    print(f"  Found PyTorch embedding: {embedding_name}")
    print(f"  PyTorch shape: {embedding_param.shape}")
    print(f"  JAX shape: {jax_embeddings.shape}")

    # Handle vocab size mismatch
    pytorch_vocab_size, embed_dim = embedding_param.shape
    jax_vocab_size, jax_embed_dim = jax_embeddings.shape

    if embed_dim != jax_embed_dim:
        print(f"  ERROR: Embedding dimension mismatch: {embed_dim} vs {jax_embed_dim}")
        return False

    # Create injection tensor
    if pytorch_vocab_size >= jax_vocab_size:
        padded_embeddings = np.zeros((pytorch_vocab_size, embed_dim), dtype=jax_embeddings.dtype)
        padded_embeddings[:jax_vocab_size] = jax_embeddings
        injection_tensor = torch.from_numpy(padded_embeddings)
        print(f"  Vocabulary handling: Padded from {jax_vocab_size} to {pytorch_vocab_size}")
    else:
        truncated_embeddings = jax_embeddings[:pytorch_vocab_size]
        injection_tensor = torch.from_numpy(np.array(truncated_embeddings))
        print(f"  Vocabulary handling: Truncated from {jax_vocab_size} to {pytorch_vocab_size}")

    # Inject
    with torch.no_grad():
        embedding_param.copy_(injection_tensor)

    # Verify injection worked
    new_embedding = embedding_param.clone()
    injection_diff = torch.max(torch.abs(new_embedding - original_embedding))
    injection_mean = torch.mean(new_embedding)

    print(f"  Injection verification:")
    print(f"    Max parameter change: {injection_diff:.6f}")
    print(f"    New embedding mean: {injection_mean:.6f}")
    print(f"    Target JAX mean: {float(np.mean(jax_embeddings)):.6f}")

    if injection_diff > 0.001:  # Significant change indicates successful injection
        print(f"  SUCCESS: Embedding injection verified")
        return True
    else:
        print(f"  WARNING: Injection may have failed - minimal parameter change")
        return False

def test_model_on_all_images(model, processor, name, output_dir):
    """
    Test model on all images using multiple prompt types.

    This function evaluates the model's text generation capabilities across
    three different prompt types: basic captioning, detailed description,
    and creative pun generation. Each prompt type tests different aspects
    of the model's linguistic and creative abilities.

    Args:
        model: PyTorch PaliGemma model to test
        processor: HuggingFace processor for the model
        name: Descriptive name for this test run
        output_dir: Directory to save images and results

    Returns:
        dict: Results organized by prompt type and image
    """
    print(f"\nSTEP 3: Testing {name} on all 25 images with multiple prompts")
    print("=" * 80)

    # Create output directory for images
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)

    results = {}

    for i, img_info in enumerate(COCO_TEST_IMAGES):
        print(f"\nImage {i+1:2d}/25: {img_info['id']}")

        image_results = {}

        try:
            # Download and save image once
            response = requests.get(img_info['url'], timeout=15)
            response.raise_for_status()
            pil_image = Image.open(BytesIO(response.content)).convert("RGB")

            image_path = image_dir / f"{img_info['id']}.jpg"
            pil_image.save(image_path)

            # Test each prompt type
            for prompt_key, prompt_config in PROMPT_TYPES.items():
                print(f"  Testing {prompt_config['description']}...")

                # Generate text with this prompt
                inputs = processor(prompt_config['template'], pil_image, return_tensors="pt")

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=prompt_config['max_tokens'],
                        do_sample=False
                    )

                input_len = inputs["input_ids"].shape[-1]
                generated_text = processor.decode(output[0][input_len:], skip_special_tokens=True)

                # Analyze text quality
                words = generated_text.split()
                unique_words = len(set(words))
                is_repetitive = unique_words <= 2 and len(words) > 2
                is_single_char = all(len(word) == 1 for word in words[:5]) if words else False
                is_empty = len(generated_text.strip()) == 0

                # Detect repetitive patterns (same phrase repeated)
                has_loops = False
                if len(words) > 4:
                    # Check for immediate repetitions
                    for j in range(len(words) - 2):
                        if words[j] == words[j + 2] and words[j + 1] == words[j + 3]:
                            has_loops = True
                            break

                print(f"    Generated: '{generated_text}'")

                # Flag quality issues
                issues = []
                if is_repetitive:
                    issues.append(f"REPETITIVE ({unique_words} unique words)")
                if is_single_char:
                    issues.append("SINGLE CHARACTERS")
                if is_empty:
                    issues.append("EMPTY OUTPUT")
                if has_loops:
                    issues.append("REPETITIVE LOOPS")

                if issues:
                    print(f"    ISSUES: {', '.join(issues)}")

                image_results[prompt_key] = {
                    'generated_text': generated_text,
                    'word_count': len(words),
                    'unique_words': unique_words,
                    'is_repetitive': is_repetitive,
                    'is_single_char': is_single_char,
                    'is_empty': is_empty,
                    'has_loops': has_loops,
                    'prompt_template': prompt_config['template'],
                    'success': True
                }

            image_results['image_id'] = img_info['id']
            image_results['image_path'] = str(image_path)
            results[img_info['id']] = image_results

        except Exception as e:
            print(f"  ERROR: {e}")
            error_result = {
                'image_id': img_info['id'],
                'success': False,
                'error': str(e)
            }
            for prompt_key in PROMPT_TYPES.keys():
                error_result[prompt_key] = {
                    'generated_text': '',
                    'success': False,
                    'error': str(e)
                }
            results[img_info['id']] = error_result

    return results

def create_html_report(base_results, pi0_results, output_dir):
    """
    Generate comprehensive HTML report comparing base vs Pi0 results.

    Creates a detailed HTML report showing side-by-side comparisons of
    text generation results across all prompt types and images. Includes
    statistical analysis and visual highlighting of issues.

    Args:
        base_results: Results from base PaliGemma model
        pi0_results: Results from Pi0-trained model
        output_dir: Directory to save the report

    Returns:
        Path: Path to the generated HTML report
    """
    print(f"\nSTEP 4: Creating comprehensive HTML report...")

    # Calculate overall statistics
    total_images = len(COCO_TEST_IMAGES)
    stats = {
        'base': {'success': 0, 'repetitive': 0, 'loops': 0, 'empty': 0},
        'pi0': {'success': 0, 'repetitive': 0, 'loops': 0, 'empty': 0}
    }

    for img_id in base_results.keys():
        for prompt_type in PROMPT_TYPES.keys():
            # Base statistics
            if img_id in base_results and prompt_type in base_results[img_id]:
                base_result = base_results[img_id][prompt_type]
                if base_result.get('success', False):
                    stats['base']['success'] += 1
                    if base_result.get('is_repetitive', False):
                        stats['base']['repetitive'] += 1
                    if base_result.get('has_loops', False):
                        stats['base']['loops'] += 1
                    if base_result.get('is_empty', False):
                        stats['base']['empty'] += 1

            # Pi0 statistics
            if img_id in pi0_results and prompt_type in pi0_results[img_id]:
                pi0_result = pi0_results[img_id][prompt_type]
                if pi0_result.get('success', False):
                    stats['pi0']['success'] += 1
                    if pi0_result.get('is_repetitive', False):
                        stats['pi0']['repetitive'] += 1
                    if pi0_result.get('has_loops', False):
                        stats['pi0']['loops'] += 1
                    if pi0_result.get('is_empty', False):
                        stats['pi0']['empty'] += 1

    total_tests = total_images * len(PROMPT_TYPES)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PaliGemma Weight Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .stats {{ background: #e8e8f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .image-comparison {{ margin: 30px 0; border: 2px solid #ddd; border-radius: 8px; overflow: hidden; }}
        .image-header {{ background: #f8f8f8; padding: 15px; border-bottom: 1px solid #ddd; }}
        .image-content {{ display: flex; }}
        .image-section {{ width: 200px; padding: 15px; border-right: 1px solid #ddd; }}
        .image-section img {{ max-width: 180px; height: auto; border-radius: 5px; }}
        .results-section {{ flex: 1; padding: 15px; }}
        .prompt-results {{ margin-bottom: 25px; }}
        .prompt-title {{ font-weight: bold; margin-bottom: 10px; padding: 5px; background: #f5f5f5; }}
        .model-result {{ margin: 8px 0; padding: 12px; border-radius: 5px; border-left: 4px solid; }}
        .base-result {{ background: #e8f5e8; border-color: #4CAF50; }}
        .pi0-result {{ background: #f5e8e8; border-color: #f44336; }}
        .issue {{ background: #ffebee !important; border-color: #f44336 !important; }}
        .repetitive {{ background: #ffcccc !important; }}
        .loops {{ background: #ffdddd !important; }}
        .empty {{ background: #ffeeee !important; }}
        .prompt-text {{ font-size: 0.9em; color: #666; margin-bottom: 8px; }}
        .generated-text {{ font-weight: bold; color: #333; }}
        .issue-flag {{ color: #d32f2f; font-weight: bold; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PaliGemma Weight Comparison Test Report</h1>
        <p><strong>Test Date:</strong> {time.ctime()}</p>
        <p><strong>Base Model:</strong> google/paligemma-3b-pt-224</p>
        <p><strong>Test Type:</strong> Base PaliGemma vs Pi0-trained weights comparison</p>
        <p><strong>Test Images:</strong> {total_images} COCO validation images</p>
        <p><strong>Prompt Types:</strong> {len(PROMPT_TYPES)} (Basic Caption, Detailed Description, Creative Pun)</p>
        <p><strong>Total Tests:</strong> {total_tests} individual text generations</p>
    </div>

    <div class="stats">
        <h2>Summary Statistics</h2>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Model</th>
                <th>Success Rate</th>
                <th>Repetitive Issues</th>
                <th>Loop Issues</th>
                <th>Empty Outputs</th>
            </tr>
            <tr>
                <td><strong>Base PaliGemma</strong></td>
                <td>{stats['base']['success']}/{total_tests} ({stats['base']['success']/total_tests*100:.1f}%)</td>
                <td>{stats['base']['repetitive']}/{total_tests} ({stats['base']['repetitive']/total_tests*100:.1f}%)</td>
                <td>{stats['base']['loops']}/{total_tests} ({stats['base']['loops']/total_tests*100:.1f}%)</td>
                <td>{stats['base']['empty']}/{total_tests} ({stats['base']['empty']/total_tests*100:.1f}%)</td>
            </tr>
            <tr>
                <td><strong>Pi0 Weights</strong></td>
                <td>{stats['pi0']['success']}/{total_tests} ({stats['pi0']['success']/total_tests*100:.1f}%)</td>
                <td>{stats['pi0']['repetitive']}/{total_tests} ({stats['pi0']['repetitive']/total_tests*100:.1f}%)</td>
                <td>{stats['pi0']['loops']}/{total_tests} ({stats['pi0']['loops']/total_tests*100:.1f}%)</td>
                <td>{stats['pi0']['empty']}/{total_tests} ({stats['pi0']['empty']/total_tests*100:.1f}%)</td>
            </tr>
        </table>
    </div>

    <h2>Detailed Results by Image</h2>
"""

    # Generate detailed image comparisons
    for i, img_id in enumerate(COCO_TEST_IMAGES):
        img_id = img_id['id']

        if img_id not in base_results or img_id not in pi0_results:
            continue

        base_data = base_results[img_id]
        pi0_data = pi0_results[img_id]

        image_path = base_data.get('image_path', f"images/{img_id}.jpg")
        relative_image_path = os.path.relpath(image_path, output_dir)

        html_content += f"""
        <div class="image-comparison">
            <div class="image-header">
                <h3>Image {i+1}: {img_id}</h3>
            </div>
            <div class="image-content">
                <div class="image-section">
                    <img src="{relative_image_path}" alt="COCO Image {img_id}">
                </div>
                <div class="results-section">
"""

        # Add results for each prompt type
        for prompt_key, prompt_config in PROMPT_TYPES.items():
            html_content += f"""
                    <div class="prompt-results">
                        <div class="prompt-title">{prompt_config['description'].upper()}</div>
                        <div class="prompt-text">Prompt: {prompt_config['template'][:100]}{'...' if len(prompt_config['template']) > 100 else ''}</div>
"""

            # Base result
            base_result = base_data.get(prompt_key, {})
            base_text = base_result.get('generated_text', 'FAILED')
            base_class = "base-result"

            html_content += f"""
                        <div class="model-result {base_class}">
                            <strong>Base PaliGemma:</strong>
                            <div class="generated-text">"{base_text}"</div>
                        </div>
"""

            # Pi0 result
            pi0_result = pi0_data.get(prompt_key, {})
            pi0_text = pi0_result.get('generated_text', 'FAILED')
            pi0_class = "pi0-result"

            # Add issue classes
            issues = []
            if pi0_result.get('is_repetitive', False):
                pi0_class += " repetitive"
                issues.append("REPETITIVE")
            if pi0_result.get('has_loops', False):
                pi0_class += " loops"
                issues.append("LOOPS")
            if pi0_result.get('is_empty', False):
                pi0_class += " empty"
                issues.append("EMPTY")

            html_content += f"""
                        <div class="model-result {pi0_class}">
                            <strong>Pi0 Weights:</strong>
                            <div class="generated-text">"{pi0_text}"</div>
                            {f'<div class="issue-flag">ISSUES: {", ".join(issues)}</div>' if issues else ''}
                        </div>
                    </div>
"""

        html_content += """
                </div>
            </div>
        </div>
"""

    html_content += """
</body>
</html>
"""

    # Save HTML report
    html_file = output_dir / "comprehensive_comparison_report.html"
    with open(html_file, 'w') as f:
        f.write(html_content)

    print(f"  SUCCESS: HTML report saved to: {html_file}")
    return html_file

def main():
    """
    Main test execution function.

    This function orchestrates the complete test workflow:
    1. Verify weight loading to ensure we have different base vs Pi0 weights
    2. Load and inject embedding weights into separate model instances
    3. Test both models on all images with multiple prompt types
    4. Generate comprehensive HTML report with visual comparisons
    5. Save all results and create final documentation

    The test isolates weight-level differences by using identical inference
    pipelines, ensuring any differences in output are due to weight corruption
    rather than implementation differences.
    """
    print("PaliGemma Weight Comparison Test Suite")
    print("=" * 80)
    print("Comparing base PaliGemma weights vs Pi0-trained weights")
    print("Testing text generation capabilities across multiple prompt types")
    print("=" * 80)

    # Create output directory
    timestamp = int(time.time())
    output_dir = Path(f"/tmp/comprehensive_verification_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    # Create output directory with timestamp
    if not verify_weight_loading():
        print("ERROR: Weight verification failed - cannot proceed with invalid test")
        return

    # Load transformers components
    model_id = "google/paligemma-3b-pt-224"
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"SUCCESS: Loaded processor for {model_id}")

    # Load both weight sets
    print("\nLoading weight sets from checkpoints...")

    weight_loader = PaliGemmaWeightLoader()
    rng = jax.random.key(42)
    config = pi0.Pi0Config()
    dummy_model = config.create(rng)
    dummy_params = nnx.state(dummy_model).to_pure_dict()
    base_params = weight_loader.load(dummy_params)
    base_weights = base_params.get("PaliGemma", {})
    base_embeddings = base_weights['llm']['embedder']['input_embedding']

    pi0_params = _model.restore_params(
        download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params")
    )
    pi0_weights = pi0_params.get("PaliGemma", {})
    pi0_embeddings = pi0_weights['llm']['embedder']['input_embedding']

    print("SUCCESS: Both weight sets loaded")

    # Test base embeddings
    print("\n" + "="*70)
    print("PHASE 1: Testing Base PaliGemma Weights")
    print("="*70)

    model_base = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
    if inject_embedding_with_verification(model_base, base_embeddings, "Base PaliGemma"):
        base_results = test_model_on_all_images(model_base, processor, "Base PaliGemma", output_dir)
    else:
        print("ERROR: Failed to inject base embeddings - aborting test")
        return

    # Test Pi0 embeddings
    print("\n" + "="*70)
    print("PHASE 2: Testing Pi0-Trained Weights")
    print("="*70)

    model_pi0 = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
    if inject_embedding_with_verification(model_pi0, pi0_embeddings, "Pi0"):
        pi0_results = test_model_on_all_images(model_pi0, processor, "Pi0", output_dir)
    else:
        print("ERROR: Failed to inject Pi0 embeddings - aborting test")
        return

    # Generate reports
    html_file = create_html_report(base_results, pi0_results, output_dir)

    # Save comprehensive results
    all_results = {
        'test_info': {
            'timestamp': time.ctime(),
            'model_id': model_id,
            'num_images': len(COCO_TEST_IMAGES),
            'num_prompt_types': len(PROMPT_TYPES),
            'test_type': 'multi_prompt_weight_comparison'
        },
        'prompt_types': PROMPT_TYPES,
        'base_results': base_results,
        'pi0_results': pi0_results
    }

    json_file = output_dir / "detailed_results.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*70)
    print("TEST SUITE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"HTML Report: {html_file}")
    print(f"JSON Results: {json_file}")
    print(f"Output Directory: {output_dir}")
    print(f"Total Tests Run: {len(COCO_TEST_IMAGES) * len(PROMPT_TYPES) * 2} individual generations")
    print("="*70)

if __name__ == "__main__":
    main()
