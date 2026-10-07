# Installation

Clone the repo:

```git clone https://github.com/LampLighterLab/BigRedGym.git```

Then, enter the BigRedGym folder.

Create the venv using uv:

```uv sync --frozen```

Note: you will need to use the ```--frozen``` flag every time you run a python script in this repo when the optional VSim wheel or Unitree SDK checkout is absent. example: ```uv run --frozen scripts/train.py --task=go2trot  --device=cuda:0 --headless --max_iterations=300```

## VS Code debugger

After `uv sync --frozen`, open the repository folder in VS Code with the Microsoft
Python and Python Debugger extensions installed. Select **Train Go2 (viewer)** in
Run and Debug, then press **F5**.

The shared [.vscode/launch.json](.vscode/launch.json) selects `.venv/bin/python`
on Linux, `.venv/Scripts/python.exe` on Windows, and `.venv/bin/mjpython` on macOS.
Edit the single `args` list to change the task, device, environment count, or
other CLI options. macOS users who encounter the library-loading error should
apply the workaround below first.

## macOS

The [MuJoCo passive viewer](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)
requires `mjpython` on macOS. Start with fewer environments than the Go2 trot
default of 4096 when running on CPU:

```sh
.venv/bin/mjpython scripts/train.py --task=go2trot --device=cpu --num_envs=16
```

If `mjpython` fails with `Library not loaded: @executable_path/../lib/libpython3.11.dylib`,
it is looking for uv's Python library relative to `.venv`. For this error, link
the existing library into `.venv/lib`, then rerun the command above:

```sh
ln -s "$(.venv/bin/python -c 'import sys; print(sys.base_prefix)')/lib/libpython3.11.dylib" .venv/lib/libpython3.11.dylib
```

This workaround is local to the virtual environment; repeat it if you recreate
`.venv` and encounter the error again.

For training without a viewer, use regular Python with `--headless`:

```sh
uv run --frozen scripts/train.py --task=go2trot --device=cpu --num_envs=16 --headless
```

# Training

To train, run ```scripts/train.py```:

```
uv run --frozen scripts/train.py --task=go2trot
```

Mandatory arguments:
```
--task: the name of your environment (go2, go2trot, mini_cheetah, etc)
```

Optional arguments:
```
--device: cpu or cuda:0 (nvidia gpu)
--backend: mujoco or vsim (most likely you will only use mujoco)
--num_envs: how many environments to run in parallel in training
--max_iterations: how many training iterations to run (typically 300-500 is sufficient for the go2)
--headless: disable the viewer window during training
--save_interval: override checkpoint interval in learning iterations
--seed: random seed for rng
--batch_size: override batch size from cfg
--experiment_name: Override experiment_name (log dir is logs/<experiment_name>/...)
--resume: Resume optimizer and model state from an existing run
--load_run: Run directory under logs/<experiment_name>/ (default: latest).
--checkpoint: Checkpoint iteration to resume (default: latest).
--original_cfg: Load environment and runner configs saved with the selected run.
```

# Testing

To test your policy in simulation, run ```scripts/play.py```:

```
uv run --frozen scripts/play.py --task=go2trot
```

Mandatory arguments:
```
--task: the name of your environment (go2, go2trot, mini_cheetah, etc)
```

Optional arguments:
```
--device: cpu or cuda:0 (nvidia gpu)
--backend: mujoco or vsim (most likely you will only use mujoco)
--experiment_name: Override experiment_name (log dir is logs/<experiment_name>/...)
--load_run: Run directory under logs/<experiment_name>/ (default: latest).
--checkpoint: Checkpoint iteration to resume (default: latest).
--original_cfg: Load environment and runner configs saved with the selected run.
```

The automated regression suite runs with `uv run --frozen python -m pytest -q`.
See [limitations and validation scope](genAI_skills/README_MUJOCO.md#limitations-and-validation-scope)
for what the retained tests cover.

## Moving a Molab policy to local playback

Molab trains the GitHub checkout selected by `REPO_URL` and `BRANCH`, not
your local working directory. Commit and push your changes to that branch,
then rerun the notebook's setup cell. Check the printed `Training checkout`
commit against `git rev-parse HEAD` locally before starting training.

`RESUME = False` starts a fresh policy. To continue a saved policy, set
`RESUME = True`, `LOAD_RUN` to its run directory, and `CHECKPOINT` to its
iteration. The training log prints `Loading model from:` when resuming.
`MAX_ITERATIONS` is the number of additional iterations on a resumed run.

For Go2 trot height control, start a fresh experiment with
`EXPERIMENT_NAME = "go2trot_height"` and `MAX_ITERATIONS = 550` as a first
training budget. Evaluate its behavior before deciding to extend training;
iteration count alone does not establish that a policy can stand or walk.
The trot policy now observes gait phase and frequency. This changes its input
size, so old checkpoints cannot resume into the current architecture. Train
fresh with `RESUME = False`; saved old configs can still be used to inspect
old policies with `--original_cfg`.

Download the complete run zip and extract it **inside `logs/`**. The resulting
layout should be `logs/<experiment>/<run>/model_550.pt` with `files/` beside
the checkpoint. Keep those saved configs. Select the run explicitly:

```sh
uv run --frozen scripts/play.py --task go2trot --device cpu --num_envs 1 --experiment_name go2trot_height --load_run YOUR_RUN --checkpoint 550 --original_cfg
uv run --frozen -m scripts.evaluate --task go2trot --device cpu --num_envs 8 --experiment_name go2trot_height --load_run YOUR_RUN --checkpoint 550 --original_cfg --height_sweep
```

Playback prints the checkpoint path and SHA256; compare the hash with the
notebook's download output to verify that the same policy reached your machine.
The evaluation reports five seconds of base height and fall terminations.
Go2 episodes now terminate below 0.20 m even without torso contact, preventing
collapsed policies from continuing to collect locomotion rewards. Height
tracking and the fall penalty have also been strengthened. Train a fresh run
with these settings; existing checkpoint weights do not change when configs
are edited. A 200-iteration run that collapses has not learned reliable support.
Robots dropping below 0.20m indicate collapse or a very low posture even if
the simulator does not flag a base-contact fall. This is a quick diagnostic,
not a complete locomotion validation.

`--height_sweep` tests standing at 0.30, 0.375, and 0.45 m with zero velocity
commands and prints average height error after settling. In interactive
playback, Up/Down changes desired height by 0.01 m within those bounds;
I/J/K/L/N/M retain their locomotion controls. Reset preserves your commands.
These bounds are training targets, not proof that every target is physically
achievable while walking. Evaluate the low and high targets after training.

Bare `play.py --task go2trot` selects the newest local run, which may be a
short local training attempt instead of the imported cloud run. The VS Code
`Train Go2 (viewer)` launch configuration also starts training; use `play.py`
to view an existing policy.

# Note on AI-generated files
Some of the code has been worked on by an AI agent, in the case where code segments have been heavily edited by AI, the AI-generated files in [genAI_skills](genAI_skills/) may be of some value in understanding the code, especially [README_MUJOCO.md](genAI_skills/README_MUJOCO.md).
