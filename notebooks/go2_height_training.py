import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    os.chdir("/home/jonathandistler/Distler_Gym/BigRedGym")
    print(os.getcwd())

    import subprocess

    _result = subprocess.run(
        [
            "./.venv/bin/python",
            "-c",
            "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')",
        ],
        capture_output=True,
        text=True,
    )

    print(_result.stdout)
    print(_result.stderr)

    return (subprocess,)


@app.cell
def _(subprocess):
    _result = subprocess.run(
        [
            "./.venv/bin/python",
            "scripts/train.py",
            "--task=go2trot",
            "--device=cuda",
            "--num_envs=128",
            "--headless",
            "--disable_wandb",
        ],
        text=True,
    )

    print("Training finished with code:", _result.returncode)
    return


if __name__ == "__main__":
    app.run()
