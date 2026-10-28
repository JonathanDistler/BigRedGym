"""Keyboard teleop for the MuJoCo passive viewer.

Wires into mujoco.viewer.launch_passive's key_callback, set on the backend
before the first render() call.  MuJoCo delivers discrete key EVENTS, so a
press queues one command increment for the simulation thread (no edge
detection needed — contrast VsimKeyboardInterface, which polls).

Bindings and command logic are shared with the vsim interface — see
teleop_bindings.py for the key layout and why it is IJKL/NM.
"""

from queue import Empty, SimpleQueue

from gym.utils.interfaces.teleop_bindings import KEY_TO_ACTION, TeleopCommands

# GLFW uses ASCII codes for letters and separate codes for special keys.
# Hardcoded to avoid a glfw dependency.
SPECIAL_KEYCODES = {"UP": 265, "DOWN": 264}
KEYCODE_TO_ACTION = {
    ord(key) if len(key) == 1 else SPECIAL_KEYCODES[key]: action
    for key, action in KEY_TO_ACTION.items()
}


class MujocoKeyboardInterface:
    def __init__(self, env):
        self.env = env
        self.commands = TeleopCommands(env)
        self._pending = SimpleQueue()

        env._backend._viewer_key_callback = self._on_key
        env._keyboard_interface = self
        self.commands.print_help("MuJoCo viewer")

    def _on_key(self, keycode: int) -> None:
        action = KEYCODE_TO_ACTION.get(keycode)
        if action is not None:
            # The viewer invokes callbacks on its own thread. Never mutate
            # tensors or reset physics there while the policy is stepping.
            self._pending.put(action)

    def process_pending(self) -> None:
        """Apply queued presses on the simulation thread before inference."""
        while True:
            try:
                action = self._pending.get_nowait()
            except Empty:
                return
            self.commands.apply(action)
