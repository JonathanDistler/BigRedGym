"""One actual worker process, n1/2/4 serial native environments; measured selection."""

import argparse, os, sys, time, resource, gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    WANDB_MODE="disabled",
)
import torch
from scripts.unified_go2 import setup
from experiments.retained_go2.actor import Student
from experiments.retained_go2.learning import Vector
from experiments.unified_go2.core import load, save_json
from experiments.front_catch.learning import collect, hp_defaults


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initial", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    cfg, data, bank, assets = setup(
        Path("output/retained_go2_preparation_v2"), a.output
    )
    net = Student()
    net.load_state_dict(load(a.initial)["network"])
    rows = []
    for n in [1, 2, 4]:
        v = Vector(assets, cfg, bank, n, data["centers"])
        actor, critic = v.reset()
        t = time.monotonic()
        collect(net, v, "", actor, critic, 8, None, hp_defaults())
        elapsed = time.monotonic() - t
        rows.append(
            dict(
                n=n,
                worker_processes=1,
                subprocess_workers=0,
                steps_per_second=8 * n / elapsed,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
        )
        del v
        gc.collect()
    memory = {
        l.split(":")[0]: int(l.split()[1]) * 1024
        for l in Path("/proc/meminfo").read_text().splitlines()
        if l.split(":")[0] in ["MemTotal", "MemAvailable"]
    }
    limit = (
        Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if Path("/sys/fs/cgroup/memory.max").exists()
        else "max"
    )
    total = (
        min(memory["MemTotal"], int(limit)) if limit != "max" else memory["MemTotal"]
    )
    budget = 0.65 * total
    eligible = [r for r in rows if r["peak_rss_kib"] * 1024 < budget]
    if not eligible:
        raise RuntimeError(
            "No measured configuration fits conservative memory budget; inspect resources before training"
        )
    best = max(r["steps_per_second"] for r in eligible)
    selected = min(r["n"] for r in eligible if r["steps_per_second"] >= 0.95 * best)
    result = dict(
        recommended_n=selected,
        measurements=rows,
        memory=memory,
        cgroup_memory_limit=limit,
        torch=str(torch.__version__),
        torch_threads=torch.get_num_threads(),
        scope="Conservative short benchmark; one process only, no16-worker configuration",
    )
    save_json(a.output / "benchmark.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
