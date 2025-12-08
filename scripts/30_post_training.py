#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B and Llama-3.1-8B on GSM8K by SFT, both on English and Chinese datasets.

Usage:
    python scripts/30_post_training.py
    python scripts/30_post_training.py --model qwen3-8b --language en
    python scripts/30_post_training.py --limit 10  # For testing
"""
import argparse
import os
import sys
import yaml
import wandb
import torch
from typing import *
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import ensure_dir, load_gsm8k_sft_data


BASE_CONFIG_QWEN = {
    "model": {
        "name": "Qwen/Qwen3-8B",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "cache_dir": ".cache/models",
        "device": "cuda",
    },
    "lora": {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM",
    },
    "training": {
        "max_steps": 500,
        "learning_rate": 1e-5,
        "gradient_accumulation_steps": 2,
        "logging_steps": 10,
        "bf16": True,
        "fp16": False,
        "max_prompt_length": 512,
        "max_completion_length": 4096,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "run_name": "sft-gsm8k",
    },
    "dataset": {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "cache_dir": ".cache/datasets",
    },
    "reward": {
        "format_reward": 0.5,
        "correctness_reward": 1.5,
    },
    "wandb": {
        "project": "rl-qwen-gsm8k",
        "enabled": True,
    },
}

BASE_CONFIG_LLAMA = {
    "model": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "cache_dir": ".cache/models",
        "device": "cuda",
    },
    "lora": {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM",
    },
    "training": {
        "max_steps": 500,
        "learning_rate": 1e-5,
        "gradient_accumulation_steps": 2,
        "logging_steps": 10,
        "bf16": True,
        "fp16": False,
        "max_prompt_length": 512,
        "max_completion_length": 4096,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 20,
        "run_name": "sft-gsm8k",
    },
    "dataset": {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "cache_dir": ".cache/datasets",
    },
    "reward": {
        "format_reward": 0.5,
        "correctness_reward": 1.5,
    },
    "wandb": {
        "project": "rl-llama-gsm8k",
        "enabled": True,
    },
}


# =============================================================================
# Data Loading & Formatting
# =============================================================================

def format_sft_example_qwen(record: Dict, tokenizer) -> Dict:
    """
    Format a GSM8K record for Qwen3 SFT training.
    Uses chat template with thinking disabled for SFT.
    
    Args:
        record: GSM8K record with question and solution
        tokenizer: Qwen tokenizer
        
    Returns:
        Formatted example with input_ids, attention_mask, labels
    """
    question = record["question"]
    solution = record["solution"]
    
    # Build messages
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": solution}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False  # Disable thinking for SFT
    )
    
    return {"text": text}


def format_sft_example_llama(record: Dict, tokenizer) -> Dict:
    """
    Format a GSM8K record for Llama-3.1 SFT training.
    
    Args:
        record: GSM8K record with question and solution
        tokenizer: Llama tokenizer
        
    Returns:
        Formatted example with text field
    """
    question = record["question"]
    solution = record["solution"]
    
    # Build messages
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": solution}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}


def create_sft_dataset(
    records: List[Dict],
    tokenizer,
    model_type: str,
):
    """
    Create a HuggingFace Dataset for SFT training.
    
    Args:
        records: List of GSM8K records
        tokenizer: Model tokenizer
        model_type: "qwen" or "llama"
        
    Returns:
        HuggingFace Dataset
    """
    # Format examples based on model type
    if model_type == "qwen":
        format_fn = format_sft_example_qwen
    else:
        format_fn = format_sft_example_llama
    
    formatted = [format_fn(r, tokenizer) for r in tqdm(records, desc="Formatting")]
    
    # Create dataset
    dataset = Dataset.from_list(formatted)
    
    return dataset


# =============================================================================
# Model Setup
# =============================================================================

def load_model_and_tokenizer(config: Dict, model_type: str):
    """
    Load model and tokenizer with LoRA configuration.
    
    Args:
        config: Model configuration dictionary
        model_type: "qwen" or "llama"
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    
    print(f"Loading model: {model_cfg['name']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        cache_dir=model_cfg.get("cache_dir"),
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model
    torch_dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        dtype=torch_dtype,
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
        cache_dir=model_cfg.get("cache_dir"),
        device_map="auto",
    )
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"]
    )
    
    model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    
    return model, tokenizer


# =============================================================================
# Training
# =============================================================================

def run_sft_training(
    model,
    tokenizer,
    train_dataset,
    config: Dict,
    output_dir: str,
    run_name: str,
):
    """
    Run SFT training with the given configuration.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        config: Training configuration
        output_dir: Directory to save checkpoints
        run_name: Name for wandb run
    """
    
    train_cfg = config["training"]
    wandb_cfg = config.get("wandb", {})
    
    # Initialize wandb if enabled
    if wandb_cfg.get("enabled", False):
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "sft-gsm8k")
        wandb.login()
    
    def tokenize_fn(ex):
        out = tokenizer(
            ex["text"],
            truncation=True,
            max_length=train_cfg["max_prompt_length"] + train_cfg["max_completion_length"],
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized_dataset = train_dataset.map(
        tokenize_fn,
        remove_columns=train_dataset.column_names,
        batched=True,
    )
    
    # SFT Config
    sft_config = TrainingArguments(
        output_dir=output_dir,
        max_steps=train_cfg.get("max_steps", 500),
        per_device_train_batch_size=train_cfg.get("batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        lr_scheduler_type="cosine",
        logging_steps=train_cfg.get("logging_steps", 10),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        report_to="wandb" if wandb_cfg.get("enabled", False) else "none",
        disable_tqdm=False,
        run_name=run_name,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=sft_config,
        train_dataset=tokenized_dataset,
    )
    
    # Train
    print(f"\nStarting SFT training...")
    print(f"  Output directory: {output_dir}")
    print()
    
    trainer.train()
    
    # Save final model
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    # tokenizer.save_pretrained(final_dir)
    print(f"\nSaved final model to {final_dir}")
    
    return final_dir


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SFT on GSM8K for Qwen3-8B and Llama-3.1-8B")
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen3-8b", "llama-3.1-8b"],
        default="qwen3-8b",
        help="Model to fine-tune"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "zh", "both"],
        default="both",
        help="Training language: en, zh, or both"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of training samples (for testing)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sft",
        help="Base output directory for checkpoints"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    args = parser.parse_args()
    
    # Select config based on model
    if args.model == "qwen3-8b":
        config = BASE_CONFIG_QWEN.copy()
        model_type = "qwen"
    else:
        config = BASE_CONFIG_LLAMA.copy()
        model_type = "llama"
    
    # Override wandb if disabled
    if args.no_wandb:
        config["wandb"]["enabled"] = False
    
    # Determine languages to train on
    languages = ["en", "zh"] if args.language == "both" else [args.language]
    
    print("=" * 60)
    print("GSM8K SFT Training")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Languages: {languages}")
    print(f"  Limit: {args.limit or 'None (full dataset)'}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)
    print()
    
    for lang in languages:
        print(f"\n{'=' * 60}")
        print(f"Training on {lang.upper()} data")
        print(f"{'=' * 60}\n")
        
        # Load data
        records = load_gsm8k_sft_data(lang, limit=args.limit)
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(config, model_type)
        
        # Create dataset
        train_dataset = create_sft_dataset(records, tokenizer, model_type)
        
        # Set up output directory and run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"sft_{args.model}_{lang}_{timestamp}"
        output_dir = os.path.join(args.output_dir, args.model, lang, timestamp)
        ensure_dir(output_dir)
        
        # Save config
        config_path = os.path.join(output_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Run training
        run_sft_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            output_dir=output_dir,
            run_name=run_name
        )
        
        # Clean up GPU memory before next language
        del model, tokenizer, train_dataset
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("SFT Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
