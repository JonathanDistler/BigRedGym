"""Optional CPU benchmark; equal warmup and measured interaction budgets/count."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deployment.molab.common import ROOT, atomic_json, stamp, verify_manifest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--counts", nargs="+", type=int, default=[8, 16, 32, 64])
    p.add_argument("--transitions", type=int, default=512)
    args = p.parse_args()
    verify_manifest()
    import torch
    from experiments.front_catch.data import network, config, Curriculum
    from experiments.front_catch.learning import collect, hp_defaults
    from experiments.front_catch.vector import Vector
    from experiments.morphology.learning.environment import Assets
    from experiments.morphology.learning.ppo import update
    from scripts.morphology_feasibility import save_csv

    torch.set_num_threads(1)
    folder = ROOT / "output" / ("molab_benchmark_" + stamp())
    folder.mkdir(parents=True)
    prep = ROOT / "output/front_catch_preparation_20261112_v3"
    manifest = json.loads((prep / "manifest.json").read_text())
    bank = torch.load(prep / "reset_bank.pt", weights_only=False, map_location="cpu")
    assets = Assets(folder / "assets")
    rows = []
    for n in args.counts:
        if n not in (8, 16, 32, 64) or args.transitions % n:
            raise ValueError("Counts must divide the matched transition budget")
        net, parent = network(ROOT / manifest["initial"])
        cfg = config(parent, "C")
        cfg.comparison_suite = "anticipatory_v1"
        hp = hp_defaults()
        hp.update(epochs=1, bc_weight=0.0)
        opt = torch.optim.Adam(net.parameters(), lr=hp["learning_rate"])
        torch.manual_seed(42)
        started = time.monotonic()
        try:
            with Vector(
                assets, {"C": cfg}, bank, n, seed=42, workers=args.workers
            ) as vector:
                actor, critic = vector.reset("C", Curriculum().mixture)
                startup = time.monotonic() - started
                _, actor, critic, _, _ = collect(
                    net,
                    vector,
                    "C",
                    actor,
                    critic,
                    max(1, 128 // n),
                    Curriculum().mixture,
                    hp,
                )
                started = time.monotonic()
                batch, _, _, _, _ = collect(
                    net,
                    vector,
                    "C",
                    actor,
                    critic,
                    args.transitions // n,
                    Curriculum().mixture,
                    hp,
                )
                seconds = time.monotonic() - started
                memory = vector.memory() / 1024**2
                started = time.monotonic()
                stats = update(net, opt, batch, hp)
                row = dict(
                    num_envs=n,
                    status="ok",
                    transitions=args.transitions,
                    warmup_transitions=128,
                    horizon=args.transitions // n,
                    workers=args.workers,
                    startup_seconds=startup,
                    collection_seconds=seconds,
                    steps_per_second=args.transitions / seconds,
                    aggregate_rss_mib=memory,
                    ppo_update_seconds=time.monotonic() - started,
                    kl=stats["post_update_kl"],
                    seed=42,
                )
        except (
            MemoryError,
            RuntimeError,
            TimeoutError,
            EOFError,
            BrokenPipeError,
        ) as error:
            row = dict(
                num_envs=n,
                status="failed",
                error=str(error),
                transitions=args.transitions,
            )
        rows.append(row)
        print(row, flush=True)
        save_csv(folder / "benchmark.csv", rows)
    eligible = [r for r in rows if r["status"] == "ok"]
    if not eligible:
        raise RuntimeError("No configuration benchmarked successfully")
    fastest = max(r["steps_per_second"] for r in eligible)
    selected = min(
        (r for r in eligible if r["steps_per_second"] >= 0.95 * fastest),
        key=lambda r: r["num_envs"],
    )
    atomic_json(
        folder / "selection.json",
        dict(
            num_envs=selected["num_envs"],
            workers=args.workers,
            transitions_per_update=args.transitions,
            horizon=args.transitions // selected["num_envs"],
            reason="Smallest tested count within 5% of fastest; no training launched",
            memory="Aggregate RSS can double-count shared pages",
        ),
    )
    atomic_json(
        ROOT / ".molab/latest_benchmark.json",
        dict(directory=str(folder.relative_to(ROOT))),
    )
    print("BENCHMARK COMPLETE:", folder)


if __name__ == "__main__":
    main()
