"""Class-balanced replay of existing competent frames, with explicit unique IDs."""

import argparse, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.unified_go2.core import digest, save_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--certificate-output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    a.certificate_output.mkdir(parents=True, exist_ok=False)
    source = Path("output/retained_go2_preparation_v2")
    for f in source.iterdir():
        if f.is_file() and f.name != "manifest.json":
            shutil.copy2(f, a.output / f.name)
    d = np.load(source / "demonstrations.npz")
    groups = [np.flatnonzero(d["roles"] == i) for i in [0, 1]]
    n = max(map(len, groups))
    ids = np.concatenate([np.resize(g, n) for g in groups])
    values = {k: d[k] for k in d.files}
    values.update(
        observations=d["observations"][ids],
        actions=d["actions"][ids],
        roles=d["roles"][ids],
        source_frame_index=ids,
    )
    np.savez_compressed(a.output / "demonstrations.npz", **values)
    assert len(np.unique(ids)) == len(d["roles"])
    cert_path = Path("output/retained_go2_validation_v3/retention.json")
    cert = json.loads(cert_path.read_text())
    proof = dict(
        source_preparation=str(source),
        source_certificate=str(cert_path),
        configs_identical=digest(source / "config.json")
        == digest(a.output / "config.json"),
        entire_reset_bank_identical=digest(source / "bank.pt")
        == digest(a.output / "bank.pt"),
        capture_centers_identical=bool(np.array_equal(d["centers"], values["centers"])),
        unique_demonstration_frames=len(d["roles"]),
        replay_rows=len(ids),
        role_rows=[n, n],
        scope="Endpoint certificate reuse only: same actor and endpoint physical states/configs, not a new simulated evaluation. Duplicates rebalance teacher sampling; they are not new experience.",
    )
    assert (
        cert["passed"]
        and proof["configs_identical"]
        and proof["entire_reset_bank_identical"]
        and proof["capture_centers_identical"]
    )
    save_json(a.output / "replay_balance.json", proof)
    provenance = json.loads((a.output / "bridge_provenance.json").read_text())
    provenance["competent_demonstrations_identical"] = False
    provenance["unique_demo_values_unchanged"] = True
    provenance["replay_balancing"] = (
        "See replay_balance.json; fresh/direct preparation certificate required"
    )
    save_json(a.output / "bridge_provenance.json", provenance)
    save_json(
        a.output / "manifest.json",
        {
            p.name: digest(p)
            for p in a.output.iterdir()
            if p.is_file() and p.name != "manifest.json"
        },
    )
    cert.update(
        preparation_sha=digest(a.output / "manifest.json"),
        transferred_endpoint_only=True,
        source_certificate=str(cert_path),
        replay_proof_sha=digest(a.output / "replay_balance.json"),
    )
    save_json(a.certificate_output / "retention.json", cert)
    print(proof, flush=True)


if __name__ == "__main__":
    main()
