"""Infrastructure adapter for 8/64 worlds; original PPO source stays unchanged.

The trainer historically accepts only 16/32 via argparse. Extend that CLI choice
at invocation, without editing its source. Eight-world training uses a separate
16-slot evaluation vector because the unchanged evaluator needs nine episodes.
No training rollouts, advantages, minibatches, rewards, contacts or actions change.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deployment.molab.common import verify_manifest


def main():
    verify_manifest()
    import scripts.run_front_catch as runner
    from experiments.front_catch.vector import Vector

    original_add = argparse.ArgumentParser.add_argument
    original_evaluate = runner.evaluate

    def add_argument(parser, *names, **kwargs):
        if "--num-envs" in names:
            kwargs["choices"] = (8, 16, 32, 64)
        return original_add(parser, *names, **kwargs)

    class DeploymentVector(Vector):
        def __init__(self, assets, configs, bank, num_envs, **kwargs):
            self.separate_evaluation = None
            if num_envs == 8:
                eval_config = {"evaluation": configs["evaluation"]}
                train_configs = {k: v for k, v in configs.items() if k != "evaluation"}
                super().__init__(assets, train_configs, bank, num_envs, **kwargs)
                try:
                    self.separate_evaluation = Vector(
                        assets, eval_config, bank, 16, **kwargs
                    )
                except BaseException:
                    self.close()
                    raise
            else:
                super().__init__(assets, configs, bank, num_envs, **kwargs)

        def close(self):
            if self.separate_evaluation is not None:
                self.separate_evaluation.close()
            super().close()

    def evaluate(net, vector, *args, **kwargs):
        evaluation_vector = vector.separate_evaluation or vector
        return original_evaluate(net, evaluation_vector, *args, **kwargs)

    argparse.ArgumentParser.add_argument = add_argument
    runner.Vector = DeploymentVector
    runner.evaluate = evaluate
    try:
        runner.main()
    finally:
        argparse.ArgumentParser.add_argument = original_add


if __name__ == "__main__":
    main()
