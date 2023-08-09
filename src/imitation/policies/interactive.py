"""Interactive policy classes to query humans for actions and associated utilities."""

import abc
import argparse
import ctypes
import sys
import time

import numpy as np
import pyglet
import retro
import torch as th
from pyglet import gl
from pyglet.window import key as keycodes
from retro import RetroEnv
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv


def query_human() -> np.ndarray:
    """Get action from human.

    0: Move left
    1: Move down
    2: Move right
    3: Move up

    Get arrow keys from human and convert to action.

    Raises:
        ValueError: If invalid action.

    Returns:
        np.array: Action.
    """
    action = None
    while action is None:
        key = input("Enter action: (w/a/s/d) ")
        try:
            action = {
                "w": 3,
                "a": 0,
                "s": 1,
                "d": 2,
            }[key]
        except KeyError:
            raise ValueError("Invalid action.")
    return np.array([action])


class InteractivePolicy(BasePolicy):
    """Interactive policy that queries a human for actions.

    Initialized with a query function that takes an observation and returns an action.
    """

    def __init__(self, venv: VecEnv, render_mode: str = "human"):
        """Builds InteractivePolicy with specified environment."""
        if venv.num_envs != 1:
            raise ValueError("InteractivePolicy only supports a single env.")

        super().__init__(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )
        self.venv = venv
        if not hasattr(venv, "envs"):
            raise ValueError("venv must have an envs attribute")

        self.env = venv.envs[0]
        if not isinstance(self.env, RetroEnv):
            raise ValueError("InteractivePolicy only supports RetroEnv environments.")

        self.render_mode = render_mode  # todo: infer from venv and make configurable
        self.interactive = Interactive(self.env)

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        """Get the action from a human user."""
        self.venv.render(mode=self.render_mode)
        print("action required")
        action = query_human()
        return th.tensor(action)


# everything below adapted from
# https://github.com/openai/retro/blob/094531b16221d9199c2e9b7913259ba448c57e37/retro/examples/interactive.py
"""
Interact with Gym environments using the keyboard

An adapter object is defined for each environment to map keyboard commands to actions and extract observations as pixels.
"""


class Interactive(abc.ABC):
    """
    Base class for making gym environments interactive for human use
    """

    def __init__(self, env: RetroEnv, sync=True, tps=60, aspect_ratio=None):
        if not hasattr(env, "buttons"):
            raise ValueError("env must have a buttons attribute")

        # todo: check env has rgb_array render mode
        # env.metadata["render.modes"] == {"rgb_array", "human"} ...

        # sync=False, tps=60, aspect_ratio=4 / 3
        sync = False
        aspect_ratio = aspect_ratio or 4 / 3

        obs = env.reset()
        self._image = self.get_image(obs, env)
        assert (
            len(self._image.shape) == 3 and self._image.shape[2] == 3
        ), "must be an RGB image"
        image_height, image_width = self._image.shape[:2]

        if aspect_ratio is None:
            aspect_ratio = image_width / image_height

        # guess a screen size that doesn't distort the image too much but also is not tiny or huge
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        max_win_width = screen.width * 0.9
        max_win_height = screen.height * 0.9
        win_width = image_width
        win_height = int(win_width / aspect_ratio)

        while win_width > max_win_width or win_height > max_win_height:
            win_width //= 2
            win_height //= 2
        while win_width < max_win_width / 2 and win_height < max_win_height / 2:
            win_width *= 2
            win_height *= 2

        win = pyglet.window.Window(width=win_width, height=win_height)

        self._key_handler = pyglet.window.key.KeyStateHandler()
        win.push_handlers(self._key_handler)
        win.on_close = self._on_close

        gl.glEnable(gl.GL_TEXTURE_2D)
        self._texture_id = gl.GLuint(0)
        gl.glGenTextures(1, ctypes.byref(self._texture_id))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            image_width,
            image_height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )

        self._env = env
        self._win = win

        # self._render_human = render_human
        self._key_previous_states = {}

        self._steps = 0
        self._episode_steps = 0
        self._episode_returns = 0
        self._prev_episode_returns = 0

        self._tps = tps
        self._sync = sync
        self._current_time = 0
        self._sim_time = 0
        self._max_sim_frames_per_update = 4

        # from RetroInteractive
        self._buttons = env.buttons
        # super().__init__(env=env, sync=False, tps=60, aspect_ratio=4 / 3)

    def _update(self, dt):
        # cap the number of frames rendered so we don't just spend forever trying to catch up on frames
        # if rendering is slow
        max_dt = self._max_sim_frames_per_update / self._tps
        if dt > max_dt:
            dt = max_dt

        # catch up the simulation to the current time
        self._current_time += dt
        while self._sim_time < self._current_time:
            self._sim_time += 1 / self._tps

            keys_clicked = set()
            keys_pressed = set()
            for key_code, pressed in self._key_handler.items():
                if pressed:
                    keys_pressed.add(key_code)

                if not self._key_previous_states.get(key_code, False) and pressed:
                    keys_clicked.add(key_code)
                self._key_previous_states[key_code] = pressed

            if keycodes.ESCAPE in keys_pressed:
                self._on_close()

            # assume that for async environments, we just want to repeat keys for as long as they are held
            inputs = keys_pressed
            if self._sync:
                inputs = keys_clicked

            keys = []
            for keycode in inputs:
                for name in dir(keycodes):
                    if getattr(keycodes, name) == keycode:
                        keys.append(name)

            act = self.keys_to_act(keys)

            if not self._sync or act is not None:
                obs, rew, done, _info = self._env.step(act)
                self._image = self.get_image(obs, self._env)
                self._episode_returns += rew
                self._steps += 1
                self._episode_steps += 1
                np.set_printoptions(precision=2)
                if self._sync:
                    done_int = int(done)  # shorter than printing True/False
                    mess = "steps={self._steps} episode_steps={self._episode_steps} rew={rew} episode_returns={self._episode_returns} done={done_int}".format(
                        **locals()
                    )
                    print(mess)
                elif self._steps % self._tps == 0 or done:
                    episode_returns_delta = (
                        self._episode_returns - self._prev_episode_returns
                    )
                    self._prev_episode_returns = self._episode_returns
                    mess = "steps={self._steps} episode_steps={self._episode_steps} episode_returns_delta={episode_returns_delta} episode_returns={self._episode_returns}".format(
                        **locals()
                    )
                    print(mess)

                if done:
                    self._env.reset()
                    self._episode_steps = 0
                    self._episode_returns = 0
                    self._prev_episode_returns = 0

    def _draw(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        video_buffer = ctypes.cast(
            self._image.tobytes(), ctypes.POINTER(ctypes.c_short)
        )
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self._image.shape[1],
            self._image.shape[0],
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            video_buffer,
        )

        x = 0
        y = 0
        w = self._win.width
        h = self._win.height

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ("v2f", [x, y, x + w, y, x + w, y + h, x, y + h]),
            ("t2f", [0, 1, 1, 1, 1, 0, 0, 0]),
        )

    def _on_close(self):
        self._env.close()
        sys.exit(0)

    def get_image(self, obs, venv):
        """
        Given an observation and the Env object, return an rgb array to display to the user
        """
        return self._env.render(mode="rgb_array")

    # @abc.abstractmethod
    def keys_to_act(self, keys):
        """
        Given a list of keys that the user has input, produce a gym action to pass to the environment

        For sync environments, keys is a list of keys that have been pressed since the last step
        For async environments, keys is a list of keys currently held down
        """
        inputs = {
            None: False,
            "BUTTON": "Z" in keys,
            "A": "Z" in keys,
            "B": "X" in keys,
            "C": "C" in keys,
            "X": "A" in keys,
            "Y": "S" in keys,
            "Z": "D" in keys,
            "L": "Q" in keys,
            "R": "W" in keys,
            "UP": "UP" in keys,
            "DOWN": "DOWN" in keys,
            "LEFT": "LEFT" in keys,
            "RIGHT": "RIGHT" in keys,
            "MODE": "TAB" in keys,
            "SELECT": "TAB" in keys,
            "RESET": "ENTER" in keys,
            "START": "ENTER" in keys,
        }
        return [inputs[b] for b in self._buttons]

    # def run(self):  # todo: eliminate if not needed
    #     """
    #     Run the interactive window until the user quits
    #     """
    #     # pyglet.app.run() has issues like https://bitbucket.org/pyglet/pyglet/issues/199/attempting-to-resize-or-close-pyglet
    #     # and also involves inverting your code to run inside the pyglet framework
    #     # avoid both by using a while loop
    #     prev_frame_time = time.time()
    #     while True:
    #         self._win.switch_to()
    #         self._win.dispatch_events()
    #         now = time.time()
    #         self._update(now - prev_frame_time)
    #         prev_frame_time = now
    #         self._draw()
    #         self._win.flip()


class RetroInteractive(Interactive):
    """
    Interactive setup for retro games
    """

    def __init__(self, game, state, scenario, record):
        env = retro.make(game=game, state=state, scenario=scenario, record=record)
        self._buttons = env.buttons
        super().__init__(env=env, sync=False, tps=60, aspect_ratio=4 / 3)

    def get_image(self, _obs, env):
        return env.render(mode="rgb_array")

    def keys_to_act(self, keys):
        inputs = {
            None: False,
            "BUTTON": "Z" in keys,
            "A": "Z" in keys,
            "B": "X" in keys,
            "C": "C" in keys,
            "X": "A" in keys,
            "Y": "S" in keys,
            "Z": "D" in keys,
            "L": "Q" in keys,
            "R": "W" in keys,
            "UP": "UP" in keys,
            "DOWN": "DOWN" in keys,
            "LEFT": "LEFT" in keys,
            "RIGHT": "RIGHT" in keys,
            "MODE": "TAB" in keys,
            "SELECT": "TAB" in keys,
            "RESET": "ENTER" in keys,
            "START": "ENTER" in keys,
        }
        return [inputs[b] for b in self._buttons]
