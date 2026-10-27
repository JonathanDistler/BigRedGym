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

Go2Trot policies observe sine/cosine gait phase and phase frequency. To resume or
play an older checkpoint trained without these inputs, add `--original_cfg` to
use the configuration saved with that run.

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

`--height_sweep` tests standing at 0.30, 0.35, and 0.40 m with zero velocity
commands and prints average height error after settling. In interactive
playback, Up/Down changes desired height by 0.01 m within those bounds;
I/J/K/L/N/M retain their locomotion controls. Reset preserves your commands.
These bounds are training targets, not proof that every target is physically
achievable while walking. Evaluate the low and high targets after training.

Go2 now starts teleop at 0.35 m. Trot reference motion and the swing-contact
penalty fade to zero at zero locomotion commands; standing rewards support
from all four feet and penalizes loaded feet sliding horizontally. Height-only
commands set a symmetric standing leg reference using the Go2 link lengths.
Residual action scaling is 0.25 radians per policy unit, and linear velocity
observations use a 3 m/s divisor. These changes affect the controller and
policy inputs: existing checkpoints need evaluation and further training with
the current configuration. Loading `--original_cfg` restores the old settings.
The height sweep also reports base drift and contact-foot speed to expose
slipping that fall counts alone miss.

Bare `play.py --task go2trot` selects the newest local run, which may be a
short local training attempt instead of the imported cloud run. The VS Code
`Train Go2 (viewer)` launch configuration also starts training; use `play.py`
to view an existing policy.

# Note on AI-generated files
Some of the code has been worked on by an AI agent, in the case where code segments have been heavily edited by AI, the AI-generated files in [genAI_skills](genAI_skills/) may be of some value in understanding the code, especially [README_MUJOCO.md](genAI_skills/README_MUJOCO.md).

### Go2 handstand primitives

`go2handstand` is a separate experimental task; existing standing and trot rewards
are unchanged. Start a simulation training run with:

```bash
uv run --frozen scripts/train.py --task=go2handstand --device=cpu --num_envs=16 --headless
```

The intended pose is a nose-down vertical torso, supported by both front feet,
with both rear feet raised and rear knees nearly extended. The task exposes its
stage to actor and critic. Standing starts use the complete sequence below;
assisted resets initialize the appropriate kick or balance stage:

1. **Forward lean:** the front-load share `p = front_Fz / total_foot_Fz` earns
   `exp(-((1-p)/0.25)^2)`, multiplied by forward pitch, two-front-foot support,
   and `exp(-((total_foot_Fz/body_weight - 1)/0.30)^2)`. Loads use each
   environment's randomized mass. This rejects flight, single-foot support,
   and large impact spikes. A score of at least 0.5 for 0.10 s unlocks kick-up.
2. **Kick-up:** either rear knee approaches -0.95 rad (about 89% two-link reach, within the URDF knee limit),
   while that foot unloads and lifts 0.12 m above the front feet. A score of at
   least 0.5 for 0.04 s unlocks balance. This rewards a kick-up posture rather
   than measuring a propulsive impulse or prescribing a kick velocity.
3. **Balance:** continuous reward requires front support near one body weight,
   load sharing across both front feet, unloaded rear feet, a vertical torso,
   both rear feet elevated at least 0.35 m, rear extension, and low linear and
   angular velocity. Foot slip, excessive force, body collisions, joint limits,
   torque, and abrupt action changes remain penalized. Torso contact terminates.

Stage gates are remembered until reset; each reward still evaluates the current
physical pose. Parameters and weights are in
`gym/envs/go2/go2handstand_config.py`. This is an initial reward design, not a
trained or validated handstand policy. The upright orientation, fixed-height,
locomotion-tracking, and small-residual objectives are omitted for this task.

The default reset curriculum keeps gravity at **9.81 m/s²**. Assistance comes
from starting closer to the goal; there is no external holding force or gravity
schedule. Each reset samples joint positions, torso orientation, and small
angular velocities. Foot placement uses Go2 leg geometry to set the root height
above the plane, and the initial PD targets match the sampled joint positions.

| Level | Starting torso pitch | Initial training stage |
|---|---|---|
| 0 | 90°, ±1° tilt | Balance |
| 1 | 90°, ±5° tilt and small angular velocity | Balance |
| 2 | 70°, ±5° tilt | Kick-up |
| 3 | 45°, ±5° tilt | Kick-up |
| 4 | 20°, ±5° tilt | Kick-up |
| 5 | Four-foot standing | Lean forward |

Pitch is measured from horizontal. At levels above zero, 70% of resets sample
the current level, 20% replay earlier levels, and 10% start from standing.
A primary episode succeeds after its balance score stays at least 0.5 for
**two uninterrupted seconds** without a termination. Promotion requires at
least 70% success in **two consecutive batches of at least 128 completed primary
episodes each**. Easier replay and standing-mixture episodes do not promote a
level. Failure keeps the current level; iteration count alone never promotes it.
The final level retains 20% easier replay and otherwise uses standing starts.

Training prints `[handstand curriculum]` batch results and promotions. Local
`vitals.jsonl` and W&B (if enabled) include `curriculum/curriculum_level` and
`curriculum/curriculum_success_rate`. A checkpoint saves the level and promotion
counters with the policy and optimizers. Resuming restarts episodes from that
level; transient robot states and exact random-number streams are not restored.
Set `handstand_curriculum.enabled = False` to train from standing only, or change
`initial_level` to choose a starting level for a fresh run.

For **MoLab**, use the updated `notebooks/molab_train.py`:

Open [molab](https://molab.marimo.io), upload the notebook, and attach a GPU
using the notebook's compute settings. Run cells from top to bottom with
Shift+Enter. For a first GPU smoke run, set `NUM_ENVS = 64` and
`MAX_ITERATIONS = 2` with a separate experiment name such as
`go2handstand_smoke`; confirm checkpoints are written before starting the
full run below. Kernel compilation can make the first iteration slow.

1. Commit/push these changes to the branch you train from, or otherwise put the
   updated repository into the runtime. MoLab cannot see local unpushed edits.
2. In Setup, set `REPO_URL` and `BRANCH` to that repository and branch, then run
   setup and the GPU check.
3. In Train, use `TASK = "go2handstand"`, `EXPERIMENT_NAME = "go2handstand"`,
   `DEVICE = "cuda:0"`, `NUM_ENVS = 4096`, and `RESUME = False` for a fresh run.
   The notebook defaults to 1500 iterations as an initial budget, not a
   guaranteed completion point. Run the training cell once; reset distributions
   and training stages change automatically inside that process.
4. To resume, keep the experiment name, set `RESUME = True` and `LOAD_RUN` to
   the saved run directory. The notebook adds `--original_cfg` for handstands
   so the saved curriculum settings accompany the restored progress. Download
   the full run before ending the runtime; on a new runtime restore it under
   `logs/<experiment>/<run>/` before resuming.

Run section 3 and click **Download** to save the complete run archive.
Do not rely on generated checkpoints surviving runtime shutdown: current
[Molab storage rules](https://marimo.io/pages/molab/storage) treat generated
workspace files as scratch. Extract the archive inside your local `logs/`
directory, preserving both checkpoints and `files/`, then shut down the
cloud runtime when finished.

Playback defaults to standing starts, which tests whether entry was learned.
To inspect the checkpoint's assisted reset level instead:

```bash
uv run --frozen scripts/play.py --task go2handstand --device cpu --num_envs 1 \
  --experiment_name go2handstand --load_run YOUR_RUN --original_cfg \
  --handstand_reset curriculum --no-keyboard
```

Next extensions are explicit center-of-mass alignment over the front-foot
support segment and a controlled return to four feet. Reset geometry and
curriculum mechanics have CPU tests; learning performance and GPU training
still require a training run.
