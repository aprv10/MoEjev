from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.trainer import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage 1 baseline MoE")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to a YAML experiment config"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional short-run override; saved in the result metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.max_steps is not None:
        config["training"]["requested_max_steps"] = config["training"]["max_steps"]
        config["training"]["max_steps_override"] = args.max_steps
    train(config, PROJECT_ROOT, args.max_steps)


if __name__ == "__main__":
    main()

