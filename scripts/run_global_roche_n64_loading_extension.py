"""Resume the N64 physical Roche-loading checkpoint to a bounded target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_adaptive_preflight import run_adaptive_campaign


ROOT = Path(__file__).resolve().parents[1]
RESTART = ROOT / "outputs/checkpoints/global_roche_adaptive_N64.npz"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-loading-fraction", required=True, type=float)
    parser.add_argument("--maximum-accepted-steps", default=40, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.target_loading_fraction <= 0.0:
        raise ValueError("target loading fraction must be positive")
    if arguments.maximum_accepted_steps <= 0:
        raise ValueError("maximum accepted steps must be positive")
    context, evaluation = _canonical_open_evaluation()
    report = run_adaptive_campaign(
        context,
        evaluation,
        n_cells=64,
        target_loading_fraction=arguments.target_loading_fraction,
        initial_dt_loading_fraction=5.0e-8,
        restart_path=RESTART,
        resume=True,
        maximum_accepted_steps=arguments.maximum_accepted_steps,
    )
    output = arguments.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
