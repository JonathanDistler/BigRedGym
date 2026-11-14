# Preserved capture reference

The read-only reference is `output/centroidal_capture_reference_20261114/`. It is a separate copy; the original search results are unchanged. `reproducibility/centroidal_capture_reference.json` locks every required large data file by SHA256 and size. It records simulator/control timesteps, gains, seed and successful exact-state prefix replay. The reference includes initial standing/full final environment state, commands/integration states, contact and torque histories, GIF, compiled nominal model and actuator contract. Read-only permissions prevent accidental writes; hashes detect modification. This is not a claim of filesystem-level WORM storage.

Large binary data are deliberately excluded from Git. Keep two backups on separate storage. From the repository root, with an external destination chosen by the operator:

```bash
python3 -c 'import json; print("\n".join(json.load(open("reproducibility/centroidal_capture_reference.json"))["files"]))' > /tmp/capture_backup_files.txt
tar -czf /EXTERNAL_STORAGE/go2_capture_20261114.tar.gz --files-from /tmp/capture_backup_files.txt
sha256sum /EXTERNAL_STORAGE/go2_capture_20261114.tar.gz > /EXTERNAL_STORAGE/go2_capture_20261114.tar.gz.sha256
```

Copy the manifest, source commit hash, and archive checksum alongside each backup. The generated Molab ZIP is another local backup/transport artifact, not the sole backup and not committed to Git. No external upload is implied or performed by these commands.

Supply the manifest's data files at their recorded repository-relative paths in a fresh checkout. Verify every SHA256 before replay:

```bash
env -u PYTHONPATH .venv/bin/python -c 'import hashlib,json,pathlib; m=json.load(open("reproducibility/centroidal_capture_reference.json")); assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==v["sha256"] for p,v in m["files"].items()); print("All preserved data hashes match")'
env -u PYTHONPATH .venv/bin/python scripts/replay_centroidal_capture.py --trajectory output/centroidal_capture_reference_20261114/capture.pt --output output/independent_capture_replay --render
```

Use Python3.11 and the existing locked package workflow. The immutable reference must remain collision-free through1.82s with maximum final integration-state difference0. Continuing the unchanged balance actor reproduces head impact at2.14s; this is a captured prefix, not a solved handstand. Archived optimizer-version3 sources are recorded separately in `reproducibility/capture_v3_source/`; current optimization may use a later objective. No Molab update5 checkpoint is substituted.
