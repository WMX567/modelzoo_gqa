"""
Kyle's muP Implementation for GPT2
===================================

This file contains the implementation of Kyle's candidate KV scaling approach for muP (Maximal Update Parameterization).

The implementation follows this specification:

kyle_impl = {
    'name': 'xLLM (muP) Kyle Candidate KV Scaling',
    'embedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: m,
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'kv_layer': {
        'init_std':             lambda m, r: (1 + r**(1/2)) / (2 * m**(1/2)),
        'lr_scale':             lambda m, r: 1 / m,
        'wd_scale':             lambda m, r: m,
        'output_multiplier':    lambda m, r: 2 / (1 + r**(1/2)),
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0,
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0 / m,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1.0 / L,
}

Where:
- m: width multiplier (hidden_size / base_hidden_size)
- r: ratio of num_heads to num_kv_groups (for GQA/MQA)
- d: head dimension
- L: number of layers
"""

import logging

from cerebras.modelzoo.models.nlp.gpt2.gpt2_model import (
    GPT2LMHeadModelConfig,
)
from cerebras.modelzoo.common.utils.model.mup_utils import (
    LRAdjustmentGroup,
)


class MuPConfig:
    """
    Configuration helper for applying Kyle's muP implementation.
    
    This class computes all the scaling factors and provides methods to apply
    them to a GPT2 model configuration.
    """
    
    def __init__(
        self,
        hidden_size: int,
        base_hidden_size: float,
        filter_size: int,
        base_filter_size: float,
        num_hidden_layers: int,
        num_heads: int,
        num_kv_groups: int = 1,
    ):
        """
        Initialize muP configuration.
        
        Args:
            hidden_size: Model hidden dimension
            base_hidden_size: Base model hidden dimension for muP transfer
            filter_size: FFN intermediate dimension
            base_filter_size: Base FFN intermediate dimension
            num_hidden_layers: Number of transformer layers
            num_heads: Number of attention heads
            num_kv_groups: Number of KV groups (1 for MHA, <num_heads for GQA/MQA)
        """
        self.hidden_size = hidden_size
        self.base_hidden_size = base_hidden_size
        self.filter_size = filter_size
        self.base_filter_size = base_filter_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        
        # Compute scaling factors
        # m: width multiplier
        self.m = hidden_size / base_hidden_size
        
        # r: ratio for GQA/MQA (num_heads / num_kv_groups)
        self.r = num_heads / num_kv_groups if num_kv_groups > 0 else 1.0
        
        # d: head dimension
        self.d = hidden_size // num_heads
        
        # L: number of layers
        self.L = num_hidden_layers
        
        logging.info(
            f"  muP configuration initialized:\n"
            f"  Width multiplier (m): {self.m}\n"
            f"  KV ratio (r): {self.r}\n"
            f"  Head dimension (d): {self.d}\n"
            f"  Number of layers (L): {self.L}"
        )
    
    def get_embedding_scales(self):
        """Get scaling factors for embedding layer."""
        return {
            'init_std': 1.0 / self.m,
            'lr_scale': 1.0 / self.m,
            'wd_scale': self.m,
            'output_multiplier': self.m,
        }
    
    def get_hidden_scales(self):
        """Get scaling factors for hidden/dense layers (Q projection, FFN input)."""
        return {
            'init_std': 1.0 / (self.m ** 0.5),
            'lr_scale': 1.0 / self.m,
            'wd_scale': self.m,
            'output_multiplier': 1.0,
        }
    
    def get_kv_scales(self):
        """Get scaling factors for K/V projection layers."""
        return {
            'init_std': (1 + self.r ** 0.5) / (2 * self.m ** 0.5),
            'lr_scale': 1.0 / self.m,
            'wd_scale': self.m,
            'output_multiplier': 2.0 / (1 + self.r ** 0.5),
        }
    
    def get_unembedding_scales(self):
        """Get scaling factors for unembedding/output layer."""
        return {
            'init_std': 1.0 / self.m,
            'lr_scale': 1.0 / self.m,
            'wd_scale': self.m,
            'output_multiplier': 1.0,
        }
    
    def get_normalization_scales(self):
        """Get scaling factors for normalization layers."""
        return {
            'lr_scale': 1.0 / self.m,
        }
    
    def get_attention_scale(self):
        """Get attention scaling factor (replaces 1/sqrt(d) in standard attention)."""
        return 1.0 / self.d
    
    def get_depth_scale(self):
        """Get depth scaling factor for residual connections."""
        return 1.0 / self.L


def apply_mup_to_config(
    config: GPT2LMHeadModelConfig,
    base_hidden_size: float,
    base_filter_size: float,
    base_init_std: float = 0.02,
) -> GPT2LMHeadModelConfig:
    """
    Apply muP implementation to a GPT2 model configuration.
    
    Args:
        config: Original GPT2 configuration
        base_hidden_size: Base hidden size for muP transfer
        base_filter_size: Base filter size for muP transfer
        base_init_std: Base initialization standard deviation
        
    Returns:
        Modified configuration with muP applied
    """
    # Initialize Kyle's muP configuration
    mup_config = MuPConfig(
        hidden_size=config.hidden_size,
        base_hidden_size=base_hidden_size,
        filter_size=config.filter_size,
        base_filter_size=base_filter_size,
        num_hidden_layers=config.num_hidden_layers,
        num_heads=config.num_heads,
        num_kv_groups=config.extra_attention_params.get("num_kv_groups", 1),
    )
    
    # Get scaling factors
    embedding_scales = mup_config.get_embedding_scales()
    hidden_scales = mup_config.get_hidden_scales()
    kv_scales = mup_config.get_kv_scales()
    unembedding_scales = mup_config.get_unembedding_scales()
    norm_scales = mup_config.get_normalization_scales()

    logging.info("Applying muP implementation to GPT2 config")

    # ===== 1. Initialization Scaling =====
    
    # Embedding initialization
    embedding_init_std = base_init_std * embedding_scales['init_std']
    config.embedding_initializer = {
        'name': 'truncated_normal',
        'mean': 0.0,
        'std': embedding_init_std,
        'a': -2.0 * embedding_init_std,
        'b': 2.0 * embedding_init_std,
    }
    
    # Hidden layers (Q projection, FFN input) initialization
    hidden_init_std = base_init_std * hidden_scales['init_std']
    config.initializer = {
        'name': 'truncated_normal',
        'mean': 0.0,
        'std': hidden_init_std,
        'a': -2.0 * hidden_init_std,
        'b': 2.0 * hidden_init_std,
    }
    
    # Q projection uses hidden scales
    config.attention_q_initializer = config.initializer
    
    # K/V projections use special KV scaling
    kv_init_std = base_init_std * kv_scales['init_std']
    kv_initializer = {
        'name': 'truncated_normal',
        'mean': 0.0,
        'std': kv_init_std,
        'a': -2.0 * kv_init_std,
        'b': 2.0 * kv_init_std,
    }
    
    # Note: GPT2 doesn't have separate K/V initializers in the current implementation
    # We'll apply this through lr_adjustment_groups which affects the effective learning
    
    # Output layer initialization (with depth scaling)
    output_init_std = base_init_std * unembedding_scales['init_std'] * mup_config.get_depth_scale()
    config.output_layer_initializer = {
        'name': 'truncated_normal',
        'mean': 0.0,
        'std': output_init_std,
        'a': -2.0 * output_init_std,
        'b': 2.0 * output_init_std,
    }
    
    # FFN output layer initialization (with depth scaling)
    ffn_output_init_std = base_init_std * hidden_scales['init_std'] * mup_config.get_depth_scale()
    config.ffn_output_layer_initializer = {
        'name': 'truncated_normal',
        'mean': 0.0,
        'std': ffn_output_init_std,
        'a': -2.0 * ffn_output_init_std,
        'b': 2.0 * ffn_output_init_std,
    }
    
    # ===== 2. Learning Rate Adjustment Groups =====
    
    lr_adjustment_groups = {
        # Embedding layer
        "embedding": LRAdjustmentGroup(
            "*embedding*weight",
            scale=embedding_scales['lr_scale']
        ),
        
        # Q projection (uses hidden scaling)
        "q_decoder_attention": LRAdjustmentGroup(
            "*decoder*attn*q_dense*weight",
            scale=hidden_scales['lr_scale']
        ),
        
        # K projection (uses KV scaling)
        "k_decoder_attention": LRAdjustmentGroup(
            "*decoder*attn*k_dense*weight",
            scale=kv_scales['lr_scale']
        ),
        
        # V projection (uses KV scaling)
        "v_decoder_attention": LRAdjustmentGroup(
            "*decoder*attn*v_dense*weight",
            scale=kv_scales['lr_scale']
        ),
        
        # Output projection in attention (uses hidden scaling)
        "decoder_kernel": LRAdjustmentGroup(
            [
                "*decoder*dense*weight",
                "*decoder*linear*weight",
            ],
            scale=hidden_scales['lr_scale']
        ),
        
        # FFN input layer (uses hidden scaling)
        "decoder_input_ffn": LRAdjustmentGroup(
            [
                "*decoder*ffn.ffn.[!1]*weight",
            ],
            scale=hidden_scales['lr_scale']
        ),
        
        # FFN output layer (uses hidden scaling)
        "decoder_output_ffn": LRAdjustmentGroup(
            [
                "*decoder*ffn.ffn.[1]*weight",
            ],
            scale=hidden_scales['lr_scale']
        ),
        
        # Normalization layers
        "normalization": LRAdjustmentGroup(
            "*norm*",
            scale=norm_scales['lr_scale']
        ),
    }
    
    config.lr_adjustment_groups = lr_adjustment_groups
    
    # ===== 3. Output Multipliers =====
    
    # Embedding output multiplier
    config.embeddings_scale = embedding_scales['output_multiplier']
    
    # Output logits multiplier
    config.output_logits_alpha = unembedding_scales['output_multiplier']
    
    # ===== 4. Attention Scaling =====
    
    # Use 1/d scaling instead of 1/sqrt(d)
    config.scale_qk_dot_by_d = True
    config.attention_logits_alpha = mup_config.get_attention_scale() / mup_config.d
    # Note: scale_qk_dot_by_d=True means we divide by d, so attention_logits_alpha
    
    # ===== 5. Depth Scaling =====
    
    config.depth_scaling_factor = mup_config.get_depth_scale()
    
    # ===== 6. muP Base Dimensions =====
    
    config.mup_base_hidden_size = base_hidden_size
    config.mup_base_filter_size = base_filter_size
    config.scale_output_logits_by_d = True
    
    return config


def create_mup_gpt2_config(
    vocab_size: int = 50257,
    hidden_size: int = 768,
    base_hidden_size: float = 256,
    num_hidden_layers: int = 12,
    num_heads: int = 12,
    filter_size: int = 3072,
    base_filter_size: float = 1024,
    max_position_embeddings: int = 1024,
    num_kv_groups: int = 1,
    base_init_std: float = 0.02,
    **kwargs
) -> GPT2LMHeadModelConfig:
    """
    Create a GPT2 configuration with Kyle's muP implementation applied.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        base_hidden_size: Base hidden dimension for muP
        num_hidden_layers: Number of transformer layers
        num_heads: Number of attention heads
        filter_size: FFN intermediate dimension
        base_filter_size: Base FFN intermediate dimension for muP
        max_position_embeddings: Maximum sequence length
        num_kv_groups: Number of KV groups (for GQA/MQA, 1 for MHA)
        base_init_std: Base initialization std
        **kwargs: Additional configuration parameters
        
    Returns:
        GPT2LMHeadModelConfig with muP applied
    """
    # Create base config
    config = GPT2LMHeadModelConfig(
        name="GPT2LMHeadModel",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,
        filter_size=filter_size,
        max_position_embeddings=max_position_embeddings,
        extra_attention_params={"num_kv_groups": num_kv_groups},
        **kwargs
    )

    # Apply muP
    config = apply_mup_to_config(
        config,
        base_hidden_size=base_hidden_size,
        base_filter_size=base_filter_size,
        base_init_std=base_init_std,
    )
    
    return config

