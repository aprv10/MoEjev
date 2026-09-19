from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from data.dataset import ByteTokenizer, TokenBlockDataset, prepare_wikitext2
from moe.model import TinyMoELanguageModel


@dataclass
class TrainingResult:
    run_directory: str
    parameter_count: int
    train_loss: float
    validation_loss: float
    validation_perplexity: float
    tokens_per_second: float
    peak_gpu_memory_mb: float
    peak_gpu_memory_reserved_mb: float
    expert_utilization: list[list[float]]
    routing_entropy: float
    steps: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: dict[str, Any]) -> TinyMoELanguageModel:
    model = config["model"]
    router = config["router"]
    return TinyMoELanguageModel(
        vocab_size=ByteTokenizer.vocab_size,
        sequence_length=model["sequence_length"],
        num_layers=model["num_layers"],
        model_dim=model["model_dim"],
        num_heads=model["num_heads"],
        dense_hidden_dim=model["dense_hidden_dim"],
        expert_hidden_dim=model["expert_hidden_dim"],
        num_experts=model["num_experts"],
        top_k=router["top_k"],
        moe_layers=model["moe_layers"],
        dropout=model["dropout"],
        load_balance_weight=router["load_balance_weight"],
        router_z_loss_weight=router["z_loss_weight"],
    )


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalized_utilization(counts: torch.Tensor) -> list[list[float]]:
    if counts.numel() == 0:
        return []
    totals = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return (counts / totals).tolist()


@torch.inference_mode()
def evaluate(
    model: TinyMoELanguageModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    batch_count = 0
    utilization = torch.zeros(
        len([b for b in model.blocks if hasattr(b.feed_forward, "router")]),
        model.num_experts,
        device=device,
    )
    entropy_sum = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            output = model(inputs, targets)
        loss_sum += float(output.language_model_loss)
        utilization += output.expert_utilization
        entropy_sum += float(output.routing_entropy)
        batch_count += 1
        if max_batches and batch_count >= max_batches:
            break
    if batch_count == 0:
        raise RuntimeError("Validation loader produced no batches")
    validation_loss = loss_sum / batch_count
    return {
        "validation_loss": validation_loss,
        "validation_perplexity": math.exp(min(validation_loss, 20.0)),
        "expert_utilization": _normalized_utilization(utilization.cpu()),
        "routing_entropy": entropy_sum / batch_count,
        "validation_batches": batch_count,
    }


def train(
    config: dict[str, Any],
    project_root: Path,
    max_steps_override: int | None = None,
    run_dir_override: Path | None = None,
    checkpoint_filename: str = "last.pt",
    summary_filename: str = "summary.json",
) -> TrainingResult:
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    device = _device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    data_config = config["data"]
    cache_dir = project_root / data_config["cache_dir"]
    paths = prepare_wikitext2(
        cache_dir,
        data_config.get("max_train_tokens"),
        data_config.get("max_validation_tokens"),
    )
    sequence_length = config["model"]["sequence_length"]
    train_dataset = TokenBlockDataset(paths["train"], sequence_length)
    validation_dataset = TokenBlockDataset(paths["validation"], sequence_length)
    training_config = config["training"]
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": training_config["batch_size"],
        "num_workers": training_config["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        betas=tuple(training_config["betas"]),
    )
    use_amp = device.type == "cuda" and training_config["mixed_precision"] == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accumulation_steps = int(training_config["gradient_accumulation_steps"])
    max_steps = (
        max_steps_override
        if max_steps_override is not None
        else int(training_config["max_steps"])
    )
    warmup_steps = min(int(training_config["warmup_steps"]), max_steps // 2)

    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = run_dir_override or (
        project_root / config["experiment"]["output_dir"]
        / f"{config['experiment']['name']}-{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.resolved.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.jsonl"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iterator = iter(train_loader)
    total_tokens = 0
    loss_window = 0.0
    window_steps = 0
    last_train_loss = float("nan")
    start_time = time.perf_counter()
    for step in range(1, max_steps + 1):
        step_loss = 0.0
        for _ in range(accumulation_steps):
            try:
                inputs, targets = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                inputs, targets = next(train_iterator)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                output = model(inputs, targets)
                scaled_loss = output.loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            step_loss += float(output.language_model_loss) / accumulation_steps
            total_tokens += inputs.numel()
        scaler.unscale_(optimizer)
        gradient_norm = clip_grad_norm_(
            model.parameters(), training_config["max_gradient_norm"]
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        last_train_loss = step_loss
        loss_window += step_loss
        window_steps += 1

        if step == 1 or step % training_config["log_every"] == 0:
            elapsed = time.perf_counter() - start_time
            record = {
                "step": step,
                "train_loss": loss_window / window_steps,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": float(gradient_norm),
                "tokens_per_second": total_tokens / max(elapsed, 1e-9),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            loss_window = 0.0
            window_steps = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    validation = evaluate(
        model,
        validation_loader,
        device,
        use_amp,
        training_config.get("validation_batches"),
    )
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    )
    peak_reserved_memory_mb = (
        torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else 0.0
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": max_steps,
        "config": config,
        "validation": validation,
    }
    torch.save(checkpoint, run_dir / checkpoint_filename)
    result = TrainingResult(
        run_directory=str(run_dir),
        parameter_count=model.parameter_count(),
        train_loss=last_train_loss,
        validation_loss=validation["validation_loss"],
        validation_perplexity=validation["validation_perplexity"],
        tokens_per_second=total_tokens / max(elapsed, 1e-9),
        peak_gpu_memory_mb=peak_memory_mb,
        peak_gpu_memory_reserved_mb=peak_reserved_memory_mb,
        expert_utilization=validation["expert_utilization"],
        routing_entropy=validation["routing_entropy"],
        steps=max_steps,
    )
    (run_dir / summary_filename).write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )
    print(json.dumps(asdict(result), indent=2), flush=True)
    return result
