import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess

    REPO = "/marimo/BigRedGym"

    if not os.path.exists(REPO):
        subprocess.run(
            [
                "git",
                "clone",
                "-b",
                "feature/base-height-control",
                "https://github.com/JonathanDistler/BigRedGym.git",
                REPO,
            ],
            check=True,
        )

    subprocess.run(
        ["uv", "python", "install", "3.11"],
        check=True,
    )

    subprocess.run(
        ["uv", "sync", "--python", "3.11", "--frozen"],
        cwd=REPO,
        check=True,
    )

    _result = subprocess.run(
        [
            f"{REPO}/.venv/bin/python",
            "-c",
            "import torch; "
            "print(torch.cuda.is_available()); "
            "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    print(_result.stdout)



    return REPO, subprocess


@app.cell
def _(REPO, subprocess):
    _result = subprocess.run(
        [
            f"{REPO}/.venv/bin/python",
            "scripts/train.py",
            "--task=go2trot",
            "--device=cuda",
            "--num_envs=1024",
            "--max_iterations=200",
            "--headless",
            "--disable_wandb",
        ],
        cwd=REPO,
        text=True,
    )

    print("Training finished with code:", _result.returncode)
    return


if __name__ == "__main__":
    app.run()
