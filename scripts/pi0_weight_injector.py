#!/usr/bin/env python3
"""
Pi0 to PaliGemma Weight Injector

A streamlined module for injecting Pi0-trained weights into HuggingFace PaliGemma models.
This creates a weight-injected model that can be used for inference on multiple datasets.

Usage:
    from pi0_weight_injector import get_pi0_injected_model
    
    model, processor = get_pi0_injected_model()
    # Use model and processor for inference
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
# import logging  # Removed - using print statements instead

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

# JAX imports
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax.numpy as jnp

# Transformers imports
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# OpenPI imports
from src.eval.profiling.openpi.src.openpi.models import model as _model
from src.eval.profiling.openpi.src.openpi.shared import download

# Logging setup removed - using print statements for output


class Pi0WeightInjector:
    """Handles injection of Pi0 weights into HuggingFace PaliGemma models."""
    
    def __init__(self, pi0_weights):
        """
        Initialize the weight injector.
        
        Args:
            pi0_weights: Pi0 checkpoint weights dictionary
        """
        self.pi0_weights = self._flatten_weights(pi0_weights)
        self.pi0_main = self._filter_main_weights(self.pi0_weights)
    
    def _flatten_weights(self, weights, prefix=""):
        """Flatten nested weight dictionary with '/' separated keys."""
        flat = {}
        for k, v in weights.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten_weights(v, key))
            else:
                flat[key] = np.array(v)
        return flat
    
    def _filter_main_weights(self, flat_weights):
        """Filter out action expert weights (those with '_1' suffix)."""
        return {k: v for k, v in flat_weights.items() 
                if not any(part.endswith("_1") for part in k.split("/"))}
    
    def inject_weights(self, hf_model):
        """
        Inject Pi0 weights into HuggingFace model.
        
        Args:
            hf_model: HuggingFace PaliGemma model to inject weights into
            
        Returns:
            bool: True if injection was successful
        """
        print("Starting Pi0 weight injection...")
        hf_state = hf_model.state_dict()
        
        loaded_count = 0
        total_count = 0
        
        # Core parameter mappings
        mappings = [
            # Embeddings and projections
            ("llm/embedder/input_embedding", "language_model.model.embed_tokens.weight"),
            ("llm/final_norm/scale", "language_model.model.norm.weight"),
            ("img/head/kernel", "multi_modal_projector.linear.weight"),
            ("img/head/bias", "multi_modal_projector.linear.bias"),
            
            # Vision transformer
            ("img/embedding/kernel", "vision_tower.vision_model.embeddings.patch_embedding.weight"),
            ("img/embedding/bias", "vision_tower.vision_model.embeddings.patch_embedding.bias"),
            ("img/pos_embedding", "vision_tower.vision_model.embeddings.position_embedding.weight"),
            ("img/Transformer/encoder_norm/scale", "vision_tower.vision_model.post_layernorm.weight"),
            ("img/Transformer/encoder_norm/bias", "vision_tower.vision_model.post_layernorm.bias"),
        ]
        
        # Load direct mappings
        for pi0_key, hf_key in mappings:
            if self._load_single_param(pi0_key, hf_key, hf_state):
                loaded_count += 1
            total_count += 1
        
        # Load layered parameters (LLM and Vision transformer layers)
        llm_loaded = self._load_llm_layers(hf_state)
        vision_loaded = self._load_vision_layers(hf_state)
        loaded_count += llm_loaded + vision_loaded
        
        # Count parameters more accurately by checking actual availability
        total_count += self._count_available_layer_params()
        
        success_rate = loaded_count / max(total_count, 1)
        print(f"Loaded {loaded_count}/{total_count} parameters ({success_rate:.1%})")
        
        if success_rate > 0.5:
            print("✓ Pi0 weight injection successful")
            return True
        else:
            print("✗ Pi0 weight injection failed - insufficient parameters loaded")
            return False
    
    def _load_single_param(self, pi0_key, hf_key, hf_state):
        """Load a single parameter with shape handling."""
        if pi0_key not in self.pi0_main or hf_key not in hf_state:
            return False
        
        pi0_param = self.pi0_main[pi0_key].copy()
        hf_param = hf_state[hf_key]
        
        # Handle shape mismatches
        pi0_param = self._handle_shape_mismatch(pi0_param, hf_param, hf_key)
        
        if pi0_param.shape != hf_param.shape:
            print(f"Warning: Shape mismatch {hf_key}: {pi0_param.shape} vs {hf_param.shape}")
            return False
        
        try:
            with torch.no_grad():
                hf_param.copy_(torch.from_numpy(pi0_param))
            return True
        except Exception as e:
            print(f"Error loading {hf_key}: {e}")
            return False
    
    def _handle_shape_mismatch(self, pi0_param, hf_param, param_name):
        """Handle common shape mismatches between Pi0 and HF parameters."""
        # Handle position embedding shape mismatch (squeeze extra dimension)
        if ("position_embedding.weight" in param_name and 
            pi0_param.ndim == 3 and hf_param.ndim == 2 and
            pi0_param.shape[0] == 1 and pi0_param.shape[1:] == hf_param.shape):
            return np.squeeze(pi0_param, axis=0)
        
        # Handle MLP gate/up projection transpose: (D, H) -> (H, D)

        if (("gate_proj.weight" in param_name or "up_proj.weight" in param_name) and 
            pi0_param.ndim == 2 and hf_param.ndim == 2 and
            pi0_param.T.shape == hf_param.shape):
            return pi0_param.T
        
        # Transpose 2D matrices if needed
        if (pi0_param.ndim == 2 and hf_param.ndim == 2 and 
            pi0_param.shape != hf_param.shape and pi0_param.T.shape == hf_param.shape):
            return pi0_param.T
        
        # Handle convolution weight format (HWIO -> OIHW)
        if (pi0_param.ndim == 4 and "patch_embedding.weight" in param_name and
            pi0_param.shape[-1] == hf_param.shape[0]):
            return np.transpose(pi0_param, (3, 2, 0, 1))
        
        # Handle embedding vocabulary size mismatch
        if ("embed_tokens.weight" in param_name and pi0_param.ndim == 2):
            hf_vocab, embed_dim = hf_param.shape
            pi0_vocab, pi0_dim = pi0_param.shape
            
            if pi0_dim == embed_dim and pi0_vocab != hf_vocab:
                if pi0_vocab < hf_vocab:
                    # Pad with zeros
                    padded = np.zeros((hf_vocab, embed_dim), dtype=pi0_param.dtype)
                    padded[:pi0_vocab] = pi0_param
                    return padded
                else:
                    # Truncate
                    return pi0_param[:hf_vocab]
        
        return pi0_param
    
    def _load_llm_layers(self, hf_state):
        """Load LLM transformer layer weights."""
        loaded = 0
        
        # Find batched LLM parameters
        layer_params = {
            'q': self._find_param("llm/layers/attn/q_einsum/w"),
            'k': self._find_param("llm/layers/attn/k_einsum/w"),
            'v': self._find_param("llm/layers/attn/v_einsum/w"),
            'o': self._find_param("llm/layers/attn/attn_vec_einsum/w"),
            'gating': self._find_param("llm/layers/mlp/gating_einsum"),
            'down': self._find_param("llm/layers/mlp/linear"),
            'attn_norm': self._find_param("llm/layers/attn_norm/scale"),
            'mlp_norm': self._find_param("llm/layers/mlp_norm/scale"),
        }
        
        if not any(param is not None for param in layer_params.values()):
            return 0
        
        # Dynamically determine number of layers from the parameter shape
        num_layers = self._get_num_llm_layers(layer_params)
        if num_layers == 0:
            return 0
            
        for i in range(num_layers):
            loaded += self._load_llm_layer(i, layer_params, hf_state)
        
        print(f"Loaded {loaded} LLM layer parameters across {num_layers} layers")
        return loaded
    
    def _load_vision_layers(self, hf_state):
        """Load vision transformer layer weights."""
        loaded = 0
        
        # Find batched vision parameters
        vision_params = {
            'q': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel"),
            'k': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel"),
            'v': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel"),
            'o': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel"),
            'q_bias': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias"),
            'k_bias': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias"),
            'v_bias': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias"),
            'o_bias': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias"),
            'fc1': self._find_param("img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel"),
            'fc2': self._find_param("img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel"),
            'fc1_bias': self._find_param("img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias"),
            'ln1': self._find_param("img/Transformer/encoderblock/LayerNorm_0/scale"),
            'ln2': self._find_param("img/Transformer/encoderblock/LayerNorm_1/scale"),
            'ln1_bias': self._find_param("img/Transformer/encoderblock/LayerNorm_0/bias"),
        }
        
        if not any(param is not None for param in vision_params.values()):
            return 0
        
        # Dynamically determine number of layers from the parameter shape
        num_layers = self._get_num_vision_layers(vision_params)
        if num_layers == 0:
            return 0
            
        for i in range(num_layers):
            loaded += self._load_vision_layer(i, vision_params, hf_state)
        
        print(f"Loaded {loaded} vision layer parameters across {num_layers} layers")
        return loaded
    
    def _find_param(self, suffix):
        """Find parameter by suffix in Pi0 weights."""
        matches = [k for k in self.pi0_main.keys() if k.endswith(suffix)]
        return self.pi0_main[matches[0]] if matches else None
    
    def _get_num_llm_layers(self, layer_params):
        """Dynamically determine number of LLM layers from parameter shapes."""
        for param_name, param_array in layer_params.items():
            if param_array is not None and param_array.ndim > 0:
                return param_array.shape[0]
        return 0
    
    def _get_num_vision_layers(self, vision_params):
        """Dynamically determine number of vision transformer layers from parameter shapes."""
        for param_name, param_array in vision_params.items():
            if param_array is not None and param_array.ndim > 0:
                return param_array.shape[0]
        return 0
    
    def _load_llm_layer(self, layer_idx, params, hf_state):
        """Load weights for a single LLM transformer layer."""
        loaded = 0
        prefix = f"language_model.model.layers.{layer_idx}"
        
        # Attention projections
        if params['q'] is not None:
            q_weight = params['q'][layer_idx]
            q_reshaped = np.transpose(q_weight, (0, 2, 1)).reshape(-1, q_weight.shape[1])
            if self._copy_param(f"{prefix}.self_attn.q_proj.weight", q_reshaped, hf_state):
                loaded += 1
        
        if params['k'] is not None:
            k_weight = params['k'][layer_idx]
            k_reshaped = np.transpose(k_weight, (0, 2, 1)).reshape(-1, k_weight.shape[1])
            if self._copy_param(f"{prefix}.self_attn.k_proj.weight", k_reshaped, hf_state):
                loaded += 1
        
        if params['v'] is not None:
            v_weight = params['v'][layer_idx]
            v_reshaped = np.transpose(v_weight, (0, 2, 1)).reshape(-1, v_weight.shape[1])
            if self._copy_param(f"{prefix}.self_attn.v_proj.weight", v_reshaped, hf_state):
                loaded += 1
        
        if params['o'] is not None:
            o_weight = params['o'][layer_idx]
            o_reshaped = o_weight.reshape(-1, o_weight.shape[-1])
            if self._copy_param(f"{prefix}.self_attn.o_proj.weight", o_reshaped, hf_state):
                loaded += 1
        
        # MLP projections - handle gating_einsum that contains both gate and up weights
        if params['gating'] is not None:
            gating_weight = params['gating'][layer_idx]  # Shape: (2, D, H)
            gate_weight = gating_weight[0]  # Shape: (D, H)
            up_weight = gating_weight[1]    # Shape: (D, H)
            
            if self._copy_param(f"{prefix}.mlp.gate_proj.weight", gate_weight, hf_state):
                loaded += 1
            if self._copy_param(f"{prefix}.mlp.up_proj.weight", up_weight, hf_state):
                loaded += 1
        
        if params['down'] is not None:
            down_weight = params['down'][layer_idx]
            if self._copy_param(f"{prefix}.mlp.down_proj.weight", down_weight, hf_state):
                loaded += 1
        
        # Layer norms
        if params['attn_norm'] is not None:
            attn_norm_weight = params['attn_norm'][layer_idx]
            if self._copy_param(f"{prefix}.input_layernorm.weight", attn_norm_weight, hf_state):
                loaded += 1
        
        if params['mlp_norm'] is not None:
            mlp_norm_weight = params['mlp_norm'][layer_idx]
            if self._copy_param(f"{prefix}.post_attention_layernorm.weight", mlp_norm_weight, hf_state):
                loaded += 1
        
        return loaded
    
    def _load_vision_layer(self, layer_idx, params, hf_state):
        """Load weights for a single vision transformer layer."""
        loaded = 0
        prefix = f"vision_tower.vision_model.encoder.layers.{layer_idx}"
        
        # Attention projections
        if params['q'] is not None:
            q_weight = params['q'][layer_idx]
            q_reshaped = np.transpose(q_weight, (2, 1, 0)).reshape(-1, q_weight.shape[0])
            if self._copy_param(f"{prefix}.self_attn.q_proj.weight", q_reshaped, hf_state):
                loaded += 1
        
        if params['k'] is not None:
            k_weight = params['k'][layer_idx]
            k_reshaped = np.transpose(k_weight, (2, 1, 0)).reshape(-1, k_weight.shape[0])
            if self._copy_param(f"{prefix}.self_attn.k_proj.weight", k_reshaped, hf_state):
                loaded += 1
        
        if params['v'] is not None:
            v_weight = params['v'][layer_idx]
            v_reshaped = np.transpose(v_weight, (2, 1, 0)).reshape(-1, v_weight.shape[0])
            if self._copy_param(f"{prefix}.self_attn.v_proj.weight", v_reshaped, hf_state):
                loaded += 1
        
        if params['o'] is not None:
            o_weight = params['o'][layer_idx]
            o_reshaped = o_weight.reshape(-1, o_weight.shape[-1])
            if self._copy_param(f"{prefix}.self_attn.out_proj.weight", o_reshaped, hf_state):
                loaded += 1
        
        # Attention biases
        if params['q_bias'] is not None:
            q_bias = params['q_bias'][layer_idx]
            q_bias_reshaped = q_bias.reshape(-1)
            if self._copy_param(f"{prefix}.self_attn.q_proj.bias", q_bias_reshaped, hf_state):
                loaded += 1
        
        if params['k_bias'] is not None:
            k_bias = params['k_bias'][layer_idx]
            k_bias_reshaped = k_bias.reshape(-1)
            if self._copy_param(f"{prefix}.self_attn.k_proj.bias", k_bias_reshaped, hf_state):
                loaded += 1
        
        if params['v_bias'] is not None:
            v_bias = params['v_bias'][layer_idx]
            v_bias_reshaped = v_bias.reshape(-1)
            if self._copy_param(f"{prefix}.self_attn.v_proj.bias", v_bias_reshaped, hf_state):
                loaded += 1
        
        if params['o_bias'] is not None:
            o_bias = params['o_bias'][layer_idx]
            if self._copy_param(f"{prefix}.self_attn.out_proj.bias", o_bias, hf_state):
                loaded += 1
        
        # MLP projections
        if params['fc1'] is not None:
            fc1_weight = params['fc1'][layer_idx]
            if self._copy_param(f"{prefix}.mlp.fc1.weight", fc1_weight, hf_state):
                loaded += 1
        
        if params['fc2'] is not None:
            fc2_weight = params['fc2'][layer_idx]
            if self._copy_param(f"{prefix}.mlp.fc2.weight", fc2_weight, hf_state):
                loaded += 1
        
        # MLP bias
        if params['fc1_bias'] is not None:
            fc1_bias = params['fc1_bias'][layer_idx]
            if self._copy_param(f"{prefix}.mlp.fc1.bias", fc1_bias, hf_state):
                loaded += 1
        
        # Layer norms
        if params['ln1'] is not None:
            ln1_weight = params['ln1'][layer_idx]
            if self._copy_param(f"{prefix}.layer_norm1.weight", ln1_weight, hf_state):
                loaded += 1
        
        if params['ln2'] is not None:
            ln2_weight = params['ln2'][layer_idx]
            if self._copy_param(f"{prefix}.layer_norm2.weight", ln2_weight, hf_state):
                loaded += 1
        
        # Layer norm bias
        if params['ln1_bias'] is not None:
            ln1_bias = params['ln1_bias'][layer_idx]
            if self._copy_param(f"{prefix}.layer_norm1.bias", ln1_bias, hf_state):
                loaded += 1
        
        return loaded
    
    def _copy_param(self, hf_key, pi0_array, hf_state):
        """Copy Pi0 parameter to HF model state."""
        if hf_key not in hf_state:
            return False
        
        hf_param = hf_state[hf_key]
        
        # Handle shape mismatches
        pi0_array = self._handle_shape_mismatch(pi0_array, hf_param, hf_key)
        
        if pi0_array.shape != hf_param.shape:
            return False
        
        try:
            with torch.no_grad():
                hf_param.copy_(torch.from_numpy(pi0_array))
            return True
        except Exception as e:
            print(f"Error copying parameter {hf_key}: {e}")
            return False
    
    def _count_available_layer_params(self):
        """Count available layer parameters in Pi0 weights dynamically."""
        # LLM layer parameter templates
        llm_param_suffixes = [
            "llm/layers/attn/q_einsum/w", "llm/layers/attn/k_einsum/w", 
            "llm/layers/attn/v_einsum/w", "llm/layers/attn/attn_vec_einsum/w",
            "llm/layers/mlp/gating_einsum", "llm/layers/mlp/linear",
            "llm/layers/attn_norm/scale", "llm/layers/mlp_norm/scale"
        ]
        
        # Vision layer parameter templates
        vision_param_suffixes = [
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel",
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel",
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel", 
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel",
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias",
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias",
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias",
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias",
            "img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel",
            "img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel",
            "img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias",
            "img/Transformer/encoderblock/LayerNorm_0/scale",
            "img/Transformer/encoderblock/LayerNorm_1/scale",
            "img/Transformer/encoderblock/LayerNorm_0/bias"
        ]
        
        # Get actual layer counts from the parameter shapes
        llm_layers = 0
        for suffix in llm_param_suffixes:
            param = self._find_param(suffix)
            if param is not None and param.ndim > 0:
                llm_layers = max(llm_layers, param.shape[0])
                break
        
        vision_layers = 0
        for suffix in vision_param_suffixes:
            param = self._find_param(suffix)
            if param is not None and param.ndim > 0:
                vision_layers = max(vision_layers, param.shape[0])
                break
        
        # Count available parameters (accounting for gating_einsum producing 2 parameters)
        llm_available = 0
        for suffix in llm_param_suffixes:
            if self._find_param(suffix) is not None:
                # gating_einsum produces 2 parameters (gate + up), others produce 1
                if suffix == "llm/layers/mlp/gating_einsum":
                    llm_available += 2
                else:
                    llm_available += 1
        
        vision_available = sum(1 for suffix in vision_param_suffixes if self._find_param(suffix) is not None)
        
        llm_count = llm_available * llm_layers
        vision_count = vision_available * vision_layers
        
        return llm_count + vision_count


def load_pi0_weights():
    """Load Pi0 checkpoint weights."""
    print("Loading Pi0 weights...")
    try:
        pi0_params = _model.restore_params(
            download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"),
            restore_type=np.ndarray
        )
        pi0_weights = pi0_params.get("PaliGemma", {})
        
        if not pi0_weights:
            raise ValueError("PaliGemma weights not found in Pi0 checkpoint")
        
        print("✓ Pi0 weights loaded successfully")
        return pi0_weights
    
    except Exception as e:
        print(f"✗ Failed to load Pi0 weights: {e}")
        raise


def get_pi0_injected_model(model_id="google/paligemma-3b-pt-224", device=None):
    """
    Create and return a Pi0 weight-injected PaliGemma model ready for inference.
    
    Args:
        model_id (str): HuggingFace model identifier for the base PaliGemma model
        device (str, optional): Device to load the model on (e.g., 'cuda', 'cpu')
        
    Returns:
        tuple: (model, processor) where model is the weight-injected PaliGemma model
               and processor is the corresponding AutoProcessor
               
    Raises:
        RuntimeError: If weight injection fails
        Exception: If model or processor loading fails
    """
    print(f"Creating Pi0 weight-injected model from {model_id}")
    
    try:
        # Load processor
        processor = AutoProcessor.from_pretrained(model_id)
        print("✓ Loaded processor")
        
        # Load base model
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        print("✓ Loaded base model")
        
        # Move to device if specified
        if device is not None:
            model = model.to(device)
            print(f"✓ Moved model to {device}")
        
        # Load Pi0 weights
        pi0_weights = load_pi0_weights()
        
        # Inject weights
        injector = Pi0WeightInjector(pi0_weights)
        if not injector.inject_weights(model):
            raise RuntimeError("Pi0 weight injection failed")
        
        print("✓ Pi0 weight-injected model ready for inference")
        return model, processor
        
    except Exception as e:
        print(f"Failed to create Pi0 weight-injected model: {e}")
        raise


def main():
    """Example usage of the Pi0 weight injector."""
    import requests
    from PIL import Image
    from io import BytesIO
    
    # Get the weight-injected model
    model, processor = get_pi0_injected_model()
    
    # Example inference
    try:
        # Load a test image
        url = "http://images.cocodataset.org/val2017/000000397133.jpg"
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Prepare input
        prompt = "caption"
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode
        input_len = inputs["input_ids"].shape[-1]
        generated_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        print(f"Generated caption: '{generated_text}'")
        
    except Exception as e:
        print(f"Inference example failed: {e}")


if __name__ == "__main__":
    main()
