# BigRedGym Molab: one ZIP, automatic Python 3.11

1. Upload `bigredgym_molab_complete.zip` into Molab.
2. Paste and execute the complete contents of `MOLAB_START.py` as a cell. Your
   notebook can remain on Python 3.13. Setup installs its own Python 3.11 under
   `/marimo/bigredgym_molab/.venv311` using uv and validates packages/assets.
3. Paste `MOLAB_TRAIN.py` in a second cell when you want to train. Edit the small
   configuration block there. Defaults run the matched A/B/C anticipatory_v1
   comparison with 16 worlds, seeds 42/43, ten updates, horizon 32, evaluations
   and full checkpoints every two updates, and a 1,200-second runtime limit.
4. Paste `MOLAB_BACKUP.py` to export all new checkpoints, metrics and figures into
   a downloadable ZIP in `/marimo`. Training also creates periodic recovery ZIPs
   and a final full backup automatically.

The setup cell never trains. Repeating setup verifies existing source and keeps
all outputs. Repeating training resumes the latest complete checkpoint; completed
runs do not restart. Set `NEW_RUN=True` for a separate timestamped experiment.

Python and wheels are downloaded during installation. This is a source/data
bundle, not an offline wheelhouse. If uv is missing, install uv or make
`/usr/local/bin/uv` available. Error messages include the failing command.
`BIGREDGYM_BUNDLE` can override the uploaded ZIP path; `BIGREDGYM_PROJECT` can
select another writable project directory. Set these via `os.environ` before
pasting the cells. No shell activation or notebook-kernel change is needed.

The ZIP contains the exact preserved actor checkpoint, prepared reset/BC data,
native Go2 model and unmodified runtime source. Only deployment tooling is new.
See `docs/molab_unified_deployment.md` for explicit commands, budgets, recovery
and limitations. GPU/MJX, W&B logging, external robot hardware and unrelated
historical training outputs are not required.
