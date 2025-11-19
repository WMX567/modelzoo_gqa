"""
muP Validation Training with Cerebras ModelZoo
===============================================

This script uses Cerebras ModelZoo's GPT2 model and training framework to validate
Kyle's muP implementation.

Features:
- Uses real GPT2LMHeadModel from ModelZoo
- Integrates with Cerebras Trainer
- Tests different KV groups (MHA/GQA/MQA) with same learning rate
- Can run on CPU, GPU, or Cerebras hardware
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
import logging

# Add ModelZoo to path
modelzoo_path = Path(__file__).parent / "modelzoo-main" / "src"
sys.path.insert(0, str(modelzoo_path))
sys.path.insert(0, str(Path(__file__).parent))

from cerebras.modelzoo.trainer import Trainer
from cerebras.modelzoo.trainer.callbacks import (
    Checkpoint,
    Logging,
    LossCallback,
)
from cerebras.modelzoo.models.nlp.gpt2.model import Gpt2Model, GPT2ModelConfig
from cerebras.modelzoo.models.nlp.gpt2.gpt2_model import GPT2LMHeadModelConfig
from cerebras.modelzoo.data.nlp.gpt.GptHDF5DataProcessor import (
    GptHDF5DataProcessor,
)
from cerebras.modelzoo.common.utils.run.cli_parser import get_params_from_args

# Import from modelzoo-main directory
sys.path.insert(0, str(Path(__file__).parent / "modelzoo-main"))
from mup_implementation import MuPConfig, apply_mup_to_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mup_gpt2_config(
    hidden_size: int = 768,
    num_kv_groups: int = 1,
    base_hidden_size: int = 256,
    num_layers: int = 2,
    num_heads: int = 12,
) -> Dict:
    """
    Create GPT2 config with muP scaling applied using apply_mup_to_config.
    
    Args:
        hidden_size: Model hidden dimension
        num_kv_groups: Number of KV groups (1=MHA, 12=MQA)
        base_hidden_size: Base model hidden size for muP
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
    
    Returns:
        Configuration dictionary ready for Cerebras Trainer
    """
    
    logger.info("=" * 70)
    logger.info("Creating muP GPT2 Configuration")
    logger.info("=" * 70)
    logger.info(f"Hidden size: {hidden_size}")
    logger.info(f"Base hidden size: {base_hidden_size}")
    logger.info(f"Num layers: {num_layers}")
    logger.info(f"Num heads: {num_heads}")
    logger.info(f"KV groups: {num_kv_groups}")
    logger.info("=" * 70)
    
    # Create base GPT2 config object
    base_config = GPT2LMHeadModelConfig(
        name="GPT2LMHeadModel",
        vocab_size=50257,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_heads=num_heads,
        filter_size=hidden_size * 4,
        max_position_embeddings=1024,
        
        # Attention config
        attention_type="scaled_dot_product",
        attention_dropout_rate=0.1,
        use_projection_bias_in_attention=True,
        use_ffn_bias_in_attention=True,
        
        # GQA/MQA config
        extra_attention_params={"num_kv_groups": num_kv_groups},
        
        # Embeddings
        embedding_dropout_rate=0.1,
        share_embedding_weights=True,
        position_embedding_type="learned",
        
        # FFN
        use_ffn_bias=True,
        nonlinearity="gelu",
        
        # Normalization
        layer_norm_epsilon=1e-5,
        
        # Output
        use_bias_in_output=False,
        
        # Dropout
        dropout_rate=0.1,
        residual_dropout_rate=0.1,
    )
    
    # Apply muP using the apply_mup_to_config function!
    logger.info("Applying muP scaling using apply_mup_to_config...")
    mup_config = apply_mup_to_config(
        config=base_config,
        base_hidden_size=base_hidden_size,
        base_filter_size=base_hidden_size * 4,
        base_init_std=0.02,
    )
    
    # Convert config object to dict for training
    config_dict = {
        "model": mup_config.__dict__,
        
        "optimizer": {
            "optimizer_type": "AdamW",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
            "max_gradient_norm": 1.0,
            "correct_bias": True,
            
            # Base learning rate (same for all experiments!)
            "learning_rate": [
                {
                    "scheduler": "Linear",
                    "initial_learning_rate": 0.0,
                    "end_learning_rate": 6e-4,
                    "total_iters": 100,  # Warmup
                },
                {
                    "scheduler": "CosineDecay",
                    "initial_learning_rate": 6e-4,
                    "end_learning_rate": 6e-5,
                    "total_iters": 4900,
                },
            ],
            
            # muP LR adjustments are already in mup_config.lr_adjustment_groups
            "adjust_learning_rate": mup_config.lr_adjustment_groups if hasattr(mup_config, 'lr_adjustment_groups') else {},
            
            "loss_scaling_factor": "dynamic",
        },
        
        "train_input": {
            "data_processor": "GptHDF5DataProcessor",
            "data_dir": "./dummy_data/train",
            "batch_size": 32,
            "micro_batch_size": "auto",
            "shuffle": True,
            "shuffle_seed": 1337,
            "num_workers": 4,
            "prefetch_factor": 2,
            "persistent_workers": False,
        },
        
        "eval_input": {
            "data_processor": "GptHDF5DataProcessor",
            "data_dir": "./dummy_data/valid",
            "batch_size": 32,
            "micro_batch_size": "auto",
            "shuffle": False,
            "num_workers": 4,
        },
        
        "runconfig": {
            "max_steps": 5000,
            "eval_steps": 500,
            "checkpoint_steps": 1000,
            "log_steps": 50,
            "save_initial_checkpoint": False,
            "seed": 1337,
        },
    }
    
    # Log muP info
    logger.info("\nmuP Configuration Applied:")
    logger.info(f"  mup_base_hidden_size: {mup_config.mup_base_hidden_size}")
    logger.info(f"  mup_base_filter_size: {mup_config.mup_base_filter_size}")
    logger.info(f"  embeddings_scale: {mup_config.embeddings_scale}")
    if hasattr(mup_config, 'attention_logits_alpha'):
        logger.info(f"  attention_logits_alpha: {mup_config.attention_logits_alpha}")
    logger.info("=" * 70)
    
    return config_dict


def create_dummy_data():
    """Create dummy HDF5 data for testing."""
    import h5py
    import numpy as np
    
    os.makedirs("./dummy_data/train", exist_ok=True)
    os.makedirs("./dummy_data/valid", exist_ok=True)
    
    logger.info("Creating dummy training data...")
    
    # Training data
    train_file = "./dummy_data/train/data.h5"
    if not os.path.exists(train_file):
        with h5py.File(train_file, 'w') as f:
            # 2000 samples, sequence length 128
            num_samples = 2000
            seq_len = 128
            
            data = np.random.randint(0, 50257, (num_samples, seq_len), dtype=np.int32)
            f.create_dataset('input_ids', data=data, dtype=np.int32)
            
            # Attention mask
            mask = np.ones((num_samples, seq_len), dtype=np.int32)
            f.create_dataset('attention_mask', data=mask, dtype=np.int32)
            
            # Labels (shifted input_ids)
            labels = np.copy(data)
            f.create_dataset('labels', data=labels, dtype=np.int32)
        
        logger.info(f"Created {train_file}")
    
    # Validation data
    val_file = "./dummy_data/valid/data.h5"
    if not os.path.exists(val_file):
        with h5py.File(val_file, 'w') as f:
            # 200 samples
            num_samples = 200
            seq_len = 128
            
            data = np.random.randint(0, 50257, (num_samples, seq_len), dtype=np.int32)
            f.create_dataset('input_ids', data=data, dtype=np.int32)
            
            mask = np.ones((num_samples, seq_len), dtype=np.int32)
            f.create_dataset('attention_mask', data=mask, dtype=np.int32)
            
            labels = np.copy(data)
            f.create_dataset('labels', data=labels, dtype=np.int32)
        
        logger.info(f"Created {val_file}")


def run_experiment(
    experiment_name: str,
    num_kv_groups: int,
    hidden_size: int = 768,
    base_hidden_size: int = 256,
    num_layers: int = 2,
    num_heads: int = 12,
    backend: str = "CPU",
    model_dir: Optional[str] = None,
):
    """
    Run a single muP validation experiment.
    
    Args:
        experiment_name: Name of experiment
        num_kv_groups: Number of KV groups
        hidden_size: Model hidden size
        base_hidden_size: Base hidden size for muP
        num_layers: Number of layers
        num_heads: Number of attention heads
        backend: Training backend ("CPU", "GPU", or "CSX")
        model_dir: Directory to save model outputs
    """
    
    logger.info("\n" + "=" * 70)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info("=" * 70)
    logger.info(f"Backend: {backend}")
    logger.info(f"KV groups: {num_kv_groups}")
    logger.info(f"Hidden size: {hidden_size}")
    logger.info("=" * 70 + "\n")
    
    # Create dummy data
    create_dummy_data()
    
    # Create config
    config = create_mup_gpt2_config(
        hidden_size=hidden_size,
        num_kv_groups=num_kv_groups,
        base_hidden_size=base_hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    
    # Set model directory
    if model_dir is None:
        model_dir = f"./validation_output/{experiment_name}"
    config["runconfig"]["model_dir"] = model_dir
    
    # Initialize model
    logger.info("Initializing model...")
    model = Gpt2Model(config["model"])
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {num_params:,}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        device=backend.lower(),
        model=lambda: model,
        optimizer=lambda model: config["optimizer"],
        schedulers=lambda optimizer: [],
        precision=None,
        loop=None,
        checkpoint=Checkpoint(),
        logging=Logging(log_steps=config["runconfig"]["log_steps"]),
        callbacks=[LossCallback()],
        loggers=None,
        seed=config["runconfig"]["seed"],
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    from cerebras.modelzoo.data.nlp.gpt.GptHDF5DataProcessor import (
        GptHDF5DataProcessor,
    )
    
    train_dataloader = cstorch.utils.data.DataLoader(
        GptHDF5DataProcessor,
        config["train_input"],
    )
    
    eval_dataloader = cstorch.utils.data.DataLoader(
        GptHDF5DataProcessor,
        config["eval_input"],
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(
        train_dataloader=train_dataloader,
        val_dataloader=eval_dataloader,
        ckpt_path=None,
    )
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Experiment {experiment_name} completed!")
    logger.info(f"Results saved to: {model_dir}")
    logger.info(f"{'=' * 70}\n")


def run_validation_suite(backend: str = "GPU"):
    """Run full validation suite."""
    
    experiments = [
        {"name": "mha_kv1", "num_kv_groups": 1},
        {"name": "gqa_kv6", "num_kv_groups": 6},
        {"name": "gqa_kv4", "num_kv_groups": 4},
        {"name": "gqa_kv3", "num_kv_groups": 3},
        {"name": "gqa_kv2", "num_kv_groups": 2},
        {"name": "mqa_kv12", "num_kv_groups": 12},
    ]
    
    logger.info("\n" + "#" * 70)
    logger.info("# muP VALIDATION SUITE")
    logger.info("#" * 70)
    logger.info(f"Backend: {backend}")
    logger.info(f"Number of experiments: {len(experiments)}")
    logger.info("#" * 70 + "\n")
    
    for exp in experiments:
        run_experiment(
            experiment_name=exp["name"],
            num_kv_groups=exp["num_kv_groups"],
            hidden_size=768,
            base_hidden_size=256,
            backend=backend,
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="muP Validation with Cerebras ModelZoo"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="suite",
        choices=["single", "suite"],
        help="Run mode: single experiment or full suite"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="mup_experiment",
        help="Experiment name"
    )
    
    parser.add_argument(
        "--num_kv_groups",
        type=int,
        default=1,
        help="Number of KV groups"
    )
    
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Model hidden size"
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        default="GPU",
        choices=["CPU", "GPU", "CSX"],
        help="Training backend (default: GPU)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "suite":
        run_validation_suite(backend=args.backend)
    else:
        run_experiment(
            experiment_name=args.experiment_name,
            num_kv_groups=args.num_kv_groups,
            hidden_size=args.hidden_size,
            backend=args.backend,
        )
