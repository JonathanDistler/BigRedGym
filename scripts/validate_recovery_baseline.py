"""Compare the frozen balance actor on original and recovery nominal resets."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.evaluate_kickup_handoff import Policy, probes, dump
from scripts.reproduce_transition_hold import BALANCE
from scripts.train_kickup_recovery import configuration
from gym.envs.go2.go2kickup_recovery import Go2KickupRecovery
from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.utils.task_registry import task_registry
from gym.utils.helpers import set_seed


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    with torch.inference_mode():
        policy = Policy(BALANCE, "go2balance", 16, True)
        old = a.output_dir / "original"
        old.mkdir()
        baseline = probes(policy, old, [(90, 0.0, 0.0, 0.0)])[0]
        policy.env._backend.close()
        cfg, train = configuration(a.dataset)
        cfg.env.num_envs = 16
        cfg.env.episode_length_s = 12
        cfg.seed = 7
        train.runner.device = "cpu"
        train.logging.enable_local_saving = False
        task_registry.convert_frequencies_to_params(cfg, train)
        task_registry.set_log_dir_name(train)
        set_seed(7)
        policy.env = Go2KickupRecovery(cfg, "cpu", True, MuJocoCPUBackend())
        policy.runner = task_registry.make_alg_runner(policy.env, train)
        saved = torch.load(BALANCE, weights_only=True, map_location="cpu")
        policy.runner.alg.actor.load_state_dict(saved["actor_state_dict"])
        policy.runner.switch_to_eval()
        policy.runner.alg.actor.requires_grad_(False)
        policy.level = 0
        current = a.output_dir / "recovery"
        current.mkdir()
        candidate = probes(policy, current, [(90, 0.0, 0.0, 0.0)])[0]
        dump(
            a.output_dir / "comparison.json",
            dict(original=baseline, recovery=candidate),
        )
        assert [r["success"] for r in baseline["trials"]] == [
            r["success"] for r in candidate["trials"]
        ]
        assert baseline["successes"] >= 14
        policy.env._backend.close()


if __name__ == "__main__":
    main()
