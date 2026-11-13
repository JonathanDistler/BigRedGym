"""Batched actor/critic inference with serial or real process-parallel MuJoCo.

Each lane has independent MjData, config, tracker and RNG. Compiled read-only
models may be shared inside a worker; never mutate model physics per lane.
"""

import copy
import multiprocessing as mp
import os
import traceback
from types import SimpleNamespace
import numpy as np
from experiments.front_catch.anticipatory import environment_class


KINDS = (
    "standing",
    "loaded",
    "pre_liftoff",
    "post_liftoff",
    "partial",
    "near_vertical",
)


def compact_assets(assets):
    return SimpleNamespace(
        **{
            key: {1.0: getattr(assets, key)[1.0]}
            for key in ("models", "poses", "nominals", "metadata")
        }
    )


def rss_bytes():
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf(
            "SC_PAGE_SIZE"
        )
    except OSError:
        return 0


class Lane:
    def __init__(self, assets, cfg, bank, seed, index):
        self.env = environment_class(cfg)(assets, copy.deepcopy(cfg))
        self.bank = bank
        self.seed = int(seed)
        self.index = index
        self.episode = 0
        self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, 17]))

    def reset(self, mixture):
        kind = str(self.rng.choice(KINDS, p=mixture))
        seed = int(
            np.random.SeedSequence(
                [self.seed, self.index, self.episode, 91]
            ).generate_state(1)[0]
        )
        self.episode += 1
        if kind == "standing":
            return self.env.reset(1.0, seed)
        states = self.bank["train"][kind]
        if not states:
            raise ValueError("No validated reset states for " + kind)
        state = states[int(self.rng.integers(len(states)))]["state"]
        result = self.env.restore_start(state, kind)
        self.env.seed = seed
        return result

    def step(self, action, mixture):
        (a, c), reward, done, info = self.env.step(action)
        terminal = c.copy()
        row = self.env.trace[-1].copy()
        episode = None
        if done:
            episode = dict(
                **info,
                world=self.index,
                episode=self.episode,
                episode_seed=self.env.seed,
            )
            a, c = self.reset(mixture)
        return a, c, reward, done, terminal, episode, row

    def evaluation_reset(self, spec, condition):
        self.env.cfg.catch_mode = condition
        if spec is None:
            return self.env.reset(1.0, 0)
        if spec["kind"] == "standing":
            return self.env.reset(1.0, spec["seed"])
        state = self.bank["evaluation"][spec["kind"]][spec["index"]]["state"]
        self.env.seed = spec["seed"]
        return self.env.restore_start(state, spec["kind"])

    def evaluation_step(self, action, active):
        if not active:
            a, c = self.env.observations()
            return a, c, 0.0, True, c, None, None
        (a, c), reward, done, info = self.env.step(action)
        return a, c, reward, done, c, info, self.env.trace[-1].copy()

    def state(self):
        return dict(
            environment=self.env.state(),
            episode=self.episode,
            rng=copy.deepcopy(self.rng.bit_generator.state),
        )

    def restore(self, state):
        self.env.restore(state["environment"])
        self.episode = state["episode"]
        self.rng.bit_generator.state = state["rng"]
        return self.env.observations()


def worker(connection, assets, configs, bank, seed, indices):
    try:
        lanes = {
            mode: [
                Lane(
                    assets,
                    cfg,
                    bank,
                    seed.get(mode, 42) if isinstance(seed, dict) else seed,
                    i,
                )
                for i in indices
            ]
            for mode, cfg in configs.items()
        }
        connection.send(("ready", rss_bytes()))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            mode = payload.get("mode")
            if command == "reset":
                result = [lane.reset(payload["mixture"]) for lane in lanes[mode]]
            elif command == "step":
                result = [
                    lane.step(action, payload["mixture"])
                    for lane, action in zip(
                        lanes[mode], payload["actions"], strict=True
                    )
                ]
            elif command == "evaluation_reset":
                result = [
                    lane.evaluation_reset(spec, payload["condition"])
                    for lane, spec in zip(lanes[mode], payload["specs"], strict=True)
                ]
            elif command == "evaluation_step":
                result = [
                    lane.evaluation_step(action, active)
                    for lane, action, active in zip(
                        lanes[mode], payload["actions"], payload["active"], strict=True
                    )
                ]
            elif command == "state":
                result = [lane.state() for lane in lanes[mode]]
            elif command == "restore":
                result = [
                    lane.restore(s)
                    for lane, s in zip(lanes[mode], payload["states"], strict=True)
                ]
            elif command == "memory":
                result = rss_bytes()
            else:
                raise ValueError(command)
            connection.send(("ok", result))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class Vector:
    def __init__(
        self, assets, configs, bank, num_envs, seed=42, workers=0, start_method="spawn"
    ):
        self.n = num_envs
        self.workers = min(workers, num_envs)
        self.configs = configs
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        assets = compact_assets(assets)
        self.groups = []
        self.connections = []
        self.processes = []
        self.closed = False
        if self.workers:
            context = mp.get_context(start_method)
            for i in range(self.workers):
                indices = list(range(i, num_envs, self.workers))
                parent, child = context.Pipe()
                process = context.Process(
                    target=worker,
                    args=(child, assets, configs, bank, seed, indices),
                    daemon=True,
                )
                process.start()
                child.close()
                self.groups.append(indices)
                self.connections.append(parent)
                self.processes.append(process)
            for c in self.connections:
                self._receive(c)
        else:
            self.lanes = {
                mode: [
                    Lane(
                        assets,
                        cfg,
                        bank,
                        seed.get(mode, 42) if isinstance(seed, dict) else seed,
                        i,
                    )
                    for i in range(num_envs)
                ]
                for mode, cfg in configs.items()
            }

    def _receive(self, connection):
        if not connection.poll(180):
            raise TimeoutError("MuJoCo worker did not reply within 180s")
        status, result = connection.recv()
        if status == "error":
            raise RuntimeError(result)
        return result

    def _call(self, command, mode, payload=None):
        payload = dict(payload or {}, mode=mode)
        if not self.workers:
            lanes = self.lanes[mode]
            if command == "reset":
                return [e.reset(payload["mixture"]) for e in lanes]
            if command == "step":
                return [
                    e.step(a, payload["mixture"])
                    for e, a in zip(lanes, payload["actions"], strict=True)
                ]
            if command == "evaluation_reset":
                return [
                    lane.evaluation_reset(spec, payload["condition"])
                    for lane, spec in zip(lanes, payload["specs"], strict=True)
                ]
            if command == "evaluation_step":
                return [
                    lane.evaluation_step(action, active)
                    for lane, action, active in zip(
                        lanes, payload["actions"], payload["active"], strict=True
                    )
                ]
            if command == "state":
                return [e.state() for e in lanes]
            if command == "restore":
                return [
                    e.restore(s) for e, s in zip(lanes, payload["states"], strict=True)
                ]
            raise ValueError(command)
        for c, ids in zip(self.connections, self.groups, strict=True):
            part = payload.copy()
            for key in ("actions", "states", "specs", "active"):
                if key in part:
                    part[key] = [payload[key][i] for i in ids]
            c.send((command, part))
        result = [None] * self.n
        for c, ids in zip(self.connections, self.groups, strict=True):
            values = self._receive(c)
            for index, value in zip(ids, values, strict=True):
                result[index] = value
        return result

    def reset(self, mode, mixture):
        result = self._call("reset", mode, dict(mixture=mixture))
        return np.stack([r[0] for r in result]), np.stack([r[1] for r in result])

    def step(self, mode, actions, mixture):
        return self._call("step", mode, dict(actions=actions, mixture=mixture))

    def state(self, mode):
        return self._call("state", mode)

    def restore(self, mode, states):
        result = self._call("restore", mode, dict(states=states))
        return np.stack([r[0] for r in result]), np.stack([r[1] for r in result])

    def evaluation_reset(self, mode, condition, specs):
        result = self._call(
            "evaluation_reset", mode, dict(condition=condition, specs=specs)
        )
        return np.stack([r[0] for r in result]), np.stack([r[1] for r in result])

    def evaluation_step(self, mode, actions, active):
        return self._call("evaluation_step", mode, dict(actions=actions, active=active))

    def memory(self):
        memory = rss_bytes()
        for c in self.connections:
            c.send(("memory", {}))
        return memory + sum(self._receive(c) for c in self.connections)

    def close(self):
        if self.closed:
            return
        for c in self.connections:
            try:
                c.send(("close", {}))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.processes:
            p.join(10)
            if p.is_alive():
                p.terminate()
                p.join(5)
        for c in self.connections:
            c.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
