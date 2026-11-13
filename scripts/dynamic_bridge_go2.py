"""Five-update bridge pilot / exact continuation, retaining the corrected actor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.retained_go2 as retained
from experiments.dynamic_bridge.sampling import Vector
from experiments.unified_go2.core import digest

_original_contract = retained.contract


def contract():
    result = _original_contract()
    for p in [
        *Path("experiments/dynamic_bridge").glob("*.py"),
        Path(__file__),
        Path("scripts/evaluate_dynamic_bridge.py"),
        Path("scripts/benchmark_dynamic_bridge.py"),
        Path("scripts/continue_dynamic_bridge.py"),
    ]:
        result[str(p.relative_to(Path.cwd())) if p.is_absolute() else str(p)] = digest(
            p
        )
    return result


_endpoint_suite = retained.learning.suite


def suite(net, cfg, bank, assets, centers, *args, **kwargs):
    rows = _endpoint_suite(net, cfg, bank, assets, centers, *args, **kwargs)
    from scripts.evaluate_dynamic_bridge import episode
    import numpy as np

    for index, start in enumerate(bank["evaluation"]["bridge"][2:], start=10):
        row, _, _ = episode(
            net, cfg, assets, centers, ("bridge", "native_heldout", start)
        )
        rows.append(
            dict(
                row,
                index=index,
                success=row["five_second_success"],
                standing_success=False,
                certified_standing_takeoff=False,
                peak_collision_free_pitch=float(np.deg2rad(row["peak_pitch_degrees"])),
                handstand_seconds=row["valid_hold"],
            )
        )
    return rows


retained.learning.suite = suite
retained.runner.Vector = Vector
retained.runner.source_contract = contract
if __name__ == "__main__":
    if "--certificate" in sys.argv:
        i = sys.argv.index("--certificate")
        retained.CERTIFICATE = Path(sys.argv[i + 1])
        del sys.argv[i : i + 2]
    retained.runner.main()
