"""Fixed cohort evaluation for the separately trained frozen-balance residual."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gym.envs.go2.frozen_balance_residual import FrozenBalanceResidual, ResidualAdapter
from gym.envs.go2.recovery_frontier import replay_batch, sample_from_arrays
from scripts.evaluate_kickup_handoff import Policy, dump
from scripts.evaluate_recovery_frontier import summary
from scripts.recovery_nominal_fixture import evaluate_nominal
from scripts.reproduce_transition_hold import BALANCE
from scripts.optimize_posture_transition import digest


def attach(policy, model):
    adapter = ResidualAdapter(policy, model)
    reset = policy.reset

    def wrapped_reset():
        reset()
        adapter.reset()

    policy.reset = wrapped_reset
    policy.action = lambda zero=False: adapter.apply()
    return adapter


def load_controller(policy, checkpoint=None):
    model = FrozenBalanceResidual(policy.runner.alg.actor)
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["baseline_sha256"] != digest(BALANCE):
            raise ValueError("Baseline checkpoint differs")
        for key, value in model.balance.state_dict().items():
            if not torch.equal(value, saved["controller"]["balance." + key]):
                raise ValueError("Residual checkpoint modified the frozen teacher")
        model.load_state_dict(saved["controller"], strict=True)
    return model.eval()


@torch.inference_mode()
def evaluate(
    checkpoint,
    output,
    dataset=Path("datasets/recovery_frontier_v4"),
    nominal_only=False,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    p = Policy(BALANCE, "go2balance", 16, True)
    try:
        torch.manual_seed(42)
        model = load_controller(p, checkpoint)
        adapter = attach(p, model)
        result = dict(
            retention=evaluate_nominal(p, output), baseline_sha256=digest(BALANCE)
        )
        if not nominal_only:
            for split in ("validation", "heldout"):
                with np.load(Path(dataset) / f"{split}.npz", allow_pickle=False) as d:
                    arrays = {k: d[k] for k in d.files}
                outcomes = []
                for start in range(0, len(arrays["seed"]), 16):
                    samples = [
                        sample_from_arrays(arrays, i)
                        for i in range(start, min(start + 16, len(arrays["seed"])))
                    ]
                    rows, _ = replay_batch(p, samples)
                    outcomes.extend(rows)
                    print(output.name, split, len(outcomes), flush=True)
                result[split] = dict(
                    all_states=summary(outcomes, range(len(outcomes))),
                    strata={
                        str(k): summary(
                            outcomes, np.flatnonzero(arrays["stratum"] == k)
                        )
                        for k in range(3)
                    },
                    outcomes=outcomes,
                    dataset_sha256=digest(Path(dataset) / f"{split}.npz"),
                )
        result["zero_correction_max_target_error_rad"] = adapter.maximum_target_error
        result["additional_correction_rms_rad"] = (
            adapter.correction_square_sum / max(adapter.telemetry_calls, 1)
        ) ** 0.5
        result["additional_correction_activation_frequency"] = (
            adapter.activation_sum / max(adapter.telemetry_calls, 1)
        )
        dump(output / "summary.json", result)
        return result
    finally:
        p.env._backend.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nominal-only", action="store_true")
    a = parser.parse_args()
    torch.set_num_threads(1)
    evaluate(a.checkpoint, a.output_dir, nominal_only=a.nominal_only)
    if a.checkpoint is None and not a.nominal_only:
        result = json.loads((a.output_dir / "summary.json").read_text())
        baseline = json.loads(
            Path(
                "output/recovery_frontier_learning_resumed_v1/summary.json"
            ).read_text()
        )["original"]
        assert result["retention"]["successes"] == 14
        assert result["zero_correction_max_target_error_rad"] == 0
        for split in ("validation", "heldout"):
            assert result[split]["outcomes"] == baseline[split]["outcomes"]
        dump(
            a.output_dir / "parity.json",
            dict(
                exact_complete_outcome_parity=True,
                near_vertical=14,
                validation_plus_heldout=30,
                states=123,
            ),
        )


if __name__ == "__main__":
    main()
