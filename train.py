import argparse
import yaml
import torch
import random
import numpy as np

from src.train import run_training


def set_seed(seed):
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def apply_overrides(config, args):
    """
    Apply CLI overrides to a config dict.
    Only overrides keys where the arg was explicitly provided (not None).
    """
    overrides = {
        "model":            args.model,
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "learning_rate":    args.lr,
        "data_dir":         args.data_dir,
        "model_path":       args.model_path,
        "seed":             args.seed,
    }
    for config_key, value in overrides.items():
        if value is not None:
            config[config_key] = value

    # Boolean flags handled separately (store_true means absence = don't override)
    if args.freeze_backbone:
        config["freeze_backbone"] = True

    return config

def main():
    parser = argparse.ArgumentParser(description="Train Dishwasher Classifier")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )

    # Optional overrides
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--data_dir", type=str, help="Override data directory")
    parser.add_argument("--model_path", type=str, help="Override output model path")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone layers")
    parser.add_argument("--seed", type=int, help="Override random seed")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    apply_overrides(config, args)

    # Set seed
    set_seed(config.get("seed", 42))

    # Run training
    run_training(config)


if __name__ == "__main__":
    main()