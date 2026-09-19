from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture local CUDA hardware details")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report: dict[str, object] = {
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu_model": properties.name,
                "total_vram_bytes": properties.total_memory,
                "total_vram_mib": properties.total_memory / (1024**2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        driver, total, free = [part.strip() for part in query.split(",")]
        report.update(
            {
                "driver_version": driver,
                "nvidia_smi_total_vram_mib": int(total),
                "available_vram_mib_at_probe": int(free),
            }
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        report["nvidia_smi_query"] = "unavailable"
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

