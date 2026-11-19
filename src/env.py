"""Gymnasium-compatible ViZDoom environment wrapper for NeuralDoom-Lab.

The wrapper turns raw ViZDoom screen buffers into channels-first ``uint8``
observations ``(C * frame_stack, H, W)`` ready for Stable-Baselines3's
``CnnPolicy`` and reports *honest* episode endings:

* ``terminated=True``  - the agent reached the goal (``"goal"``) or died
  (``"death"``).  These are real terminal states.
* ``truncated=True``   - the scenario's ``episode_timeout`` was hit
  (``"timeout"``).  PPO must bootstrap the value here instead of treating the
  state as terminal, otherwise every long episode biases the value function.

The outcome is read from ViZDoom itself (``is_episode_timeout_reached`` /
``is_player_dead``) rather than guessed from the sign of the reward.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections import deque
from collections.abc import Iterator
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces

from src.actions import build_action_matrix
from src.config import EnvConfig, ScenarioSpec, resolve_scenario

EPISODE_OUTCOMES = ("goal", "death", "timeout")


@contextlib.contextmanager
def suppressed_stdio() -> Iterator[None]:
    """Send file descriptors 1 and 2 to ``os.devnull`` for the duration.

    ViZDoom starts its engine as a child process that inherits our stdout. On
    Windows that child keeps writing its start-up banner (``W_Init: Init
    WADfiles...``) straight to the console, which buries test and training
    output. Redirecting the descriptors before ``game.init()`` means the engine
    inherits a silent handle and stays quiet for its whole lifetime, while
    Python's own output is restored immediately afterwards.

    The redirection is process-wide for the few milliseconds ``init()`` takes,
    so it is only safe because environments are created one at a time.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        # The engine runs in a child process, so the null handle has to be
        # inheritable or the child would inherit an invalid stdout handle.
        with contextlib.suppress(OSError):  # pragma: no cover - platform quirk
            os.set_inheritable(devnull, True)
        saved = (os.dup(1), os.dup(2))
    except OSError:  # pragma: no cover - no usable stdio
        yield
        return

    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        for target, backup in zip((1, 2), saved, strict=True):
            os.dup2(backup, target)
            os.close(backup)
        os.close(devnull)


def has_display() -> bool:
    """Return True if a graphical display looks available.

    Used to avoid handing ViZDoom a visible window on a headless machine, which
    aborts the whole process instead of raising a Python exception.
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class VizDoomEnv(gym.Env[np.ndarray, int]):
    """Gymnasium wrapper around ViZDoom using pixel observations.

    Observations are returned in channels-first ``(C, H, W)`` ``uint8`` format
    (e.g. ``(4, 84, 84)`` for 4 stacked 84x84 grayscale frames).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 35}

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}. "
                f"Choose from {self.metadata['render_modes']}."
            )

        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.scenario: ScenarioSpec = resolve_scenario(self.config.scenario)

        cfg_path = os.path.join(vzd.scenarios_path, self.scenario.cfg_file)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"ViZDoom scenario config not found at: {cfg_path}\nIs vizdoom installed correctly?"
            )

        self.game = vzd.DoomGame()
        try:
            self._configure_game(cfg_path)
        except BaseException:
            # Never leak the C++ game object if configuration/init fails.
            self.close()
            raise

        self._timeout_ticks: int = self._read_timeout_ticks()

        available_buttons = self.game.get_available_buttons()
        self.action_matrix, self.action_names = build_action_matrix(available_buttons)
        self.action_space = spaces.Discrete(len(self.action_matrix))

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.config.observation_shape,
            dtype=np.uint8,
        )
        # Frame skip 4 at 35 tics/s ~= 8.75 agent steps per second.
        self.metadata = {
            **type(self).metadata,
            "render_fps": max(1, round(35 / self.config.frame_skip)),
        }

        self._frames: deque[np.ndarray] = deque(maxlen=self.config.frame_stack)
        self._last_screen: np.ndarray | None = None
        self._episode_reward: float = 0.0
        self._episode_steps: int = 0
        self._episode_ticks: int = 0
        self.episode_outcome: str | None = None
        self._needs_reset: bool = True

    # ------------------------------------------------------------------ setup
    def _configure_game(self, cfg_path: str) -> None:
        self.game.load_config(cfg_path)
        visible = bool(self.config.window_visible or self.render_mode == "human")
        if visible and not has_display():
            raise RuntimeError(
                "A visible ViZDoom window was requested but no DISPLAY/"
                "WAYLAND_DISPLAY is set. Pass window_visible=False (or run "
                "under xvfb-run) for headless machines."
            )
        self.game.set_window_visible(visible)

        if self.config.grayscale:
            self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        else:
            self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_seed(self.config.seed)
        self._init_game()

    def _init_game(self) -> None:
        """Start the engine, optionally with its console banner suppressed."""
        try:
            if self.config.silence_engine:
                with suppressed_stdio():
                    self.game.init()
            else:
                self.game.init()
        except Exception as exc:
            hint = ""
            if self.config.silence_engine:
                hint = (
                    " ViZDoom engine output was suppressed; re-run with "
                    "--engine-log (or silence_engine=False) to see it."
                )
            raise RuntimeError(f"ViZDoom failed to initialise: {exc}.{hint}") from exc

    def _read_timeout_ticks(self) -> int:
        getter = getattr(self.game, "get_episode_timeout", None)
        if getter is not None:
            try:
                return int(getter())
            except Exception:  # pragma: no cover - defensive for old bindings
                pass
        return self.scenario.episode_timeout_ticks

    # ----------------------------------------------------------- preprocessing
    def _preprocess_frame(self, screen_buffer: np.ndarray | None) -> np.ndarray:
        """Resize a ViZDoom screen buffer to ``(C, H, W)`` ``uint8``."""
        size = (self.config.frame_width, self.config.frame_height)
        channels = self.config.channels_per_frame

        if screen_buffer is None:
            self._last_screen = None
            return np.zeros(
                (channels, self.config.frame_height, self.config.frame_width),
                dtype=np.uint8,
            )

        # ViZDoom reuses its internal buffer, so keep a private copy.
        screen = np.array(screen_buffer, copy=True)
        self._last_screen = screen

        if self.config.grayscale:
            gray = screen[:, :, 0] if screen.ndim == 3 else screen
            resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            return resized[np.newaxis, :, :].astype(np.uint8)

        # RGB24 buffers come as (H, W, 3) or (3, H, W) depending on binding.
        if screen.ndim == 3 and screen.shape[0] == 3 and screen.shape[-1] != 3:
            hwc = np.transpose(screen, (1, 2, 0))
        else:
            hwc = screen
        resized = cv2.resize(hwc, size, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(np.transpose(resized, (2, 0, 1))).astype(np.uint8)

    def _get_stacked_obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)

    # --------------------------------------------------------------- gymnasium
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        effective_seed = self.config.seed if seed is None else int(seed)
        self.game.set_seed(effective_seed)
        self.game.new_episode()

        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_ticks = max(0, self._episode_time())
        self.episode_outcome = None
        self._needs_reset = False

        state = self.game.get_state()
        frame = self._preprocess_frame(None if state is None else state.screen_buffer)
        self._frames.clear()
        for _ in range(self.config.frame_stack):
            self._frames.append(frame)

        info: dict[str, Any] = {
            "scenario": self.config.scenario,
            "seed": effective_seed,
        }
        return self._get_stacked_obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError(
                "step() called before reset() or after the episode finished. Call reset() first."
            )
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r} for action space {self.action_space}")

        button_vec = self.action_matrix[int(action)]
        reward = float(self.game.make_action(button_vec, self.config.frame_skip))
        self._episode_reward += reward
        self._episode_steps += 1

        # ViZDoom reports 0 tics once a goal ends the episode, so remember the
        # last valid counter value instead of reporting a misleading zero.
        current_ticks = self._episode_time()
        if current_ticks > 0:
            self._episode_ticks = current_ticks

        done = bool(self.game.is_episode_finished())
        if done:
            self.episode_outcome = self._classify_episode_end()
            frame = self._preprocess_frame(None)
            self._needs_reset = True
        else:
            state = self.game.get_state()
            frame = self._preprocess_frame(None if state is None else state.screen_buffer)

        self._frames.append(frame)
        obs = self._get_stacked_obs()

        terminated = bool(done and self.episode_outcome in ("goal", "death"))
        truncated = bool(done and self.episode_outcome == "timeout")

        info: dict[str, Any] = {
            "scenario": self.config.scenario,
            "episode_reward": self._episode_reward,
            "episode_steps": self._episode_steps,
            "episode_time_ticks": self._episode_ticks,
            "episode_outcome": self.episode_outcome,
            "is_success": self.episode_outcome == "goal",
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------- episode end
    def _episode_time(self) -> int:
        getter = getattr(self.game, "get_episode_time", None)
        if getter is None:  # pragma: no cover - very old bindings
            return -1
        return int(getter())

    def _is_timeout_reached(self) -> bool:
        checker = getattr(self.game, "is_episode_timeout_reached", None)
        if checker is not None:
            try:
                return bool(checker())
            except Exception:  # pragma: no cover - defensive
                pass
        return self._episode_time() >= self._timeout_ticks

    def _is_player_dead(self) -> bool:
        checker = getattr(self.game, "is_player_dead", None)
        if checker is None:  # pragma: no cover - very old bindings
            return False
        return bool(checker())

    def _classify_episode_end(self) -> str:
        """Return ``"timeout"``, ``"death"`` or ``"goal"`` for a finished episode."""
        if self._is_timeout_reached():
            return "timeout"
        if self._is_player_dead():
            return "death"
        return "goal"

    # ------------------------------------------------------------------ extras
    @property
    def timeout_ticks(self) -> int:
        """Scenario ``episode_timeout`` in tics (read from the loaded cfg)."""
        return self._timeout_ticks

    def render(self) -> np.ndarray | None:
        """Return the latest frame as ``uint8`` ``(H, W, 3)`` when asked for it."""
        if self.render_mode != "rgb_array":
            return None

        shape = (self.config.frame_height, self.config.frame_width, 3)
        if self._last_screen is None:
            return np.zeros(shape, dtype=np.uint8)

        screen = self._last_screen
        if self.config.grayscale:
            gray = screen[:, :, 0] if screen.ndim == 3 else screen
            resized = cv2.resize(
                gray,
                (self.config.frame_width, self.config.frame_height),
                interpolation=cv2.INTER_AREA,
            )
            return np.repeat(resized[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

        if screen.ndim == 3 and screen.shape[0] == 3 and screen.shape[-1] != 3:
            hwc = np.transpose(screen, (1, 2, 0))
        else:
            hwc = screen
        resized = cv2.resize(
            hwc,
            (self.config.frame_width, self.config.frame_height),
            interpolation=cv2.INTER_AREA,
        )
        return np.ascontiguousarray(resized, dtype=np.uint8)

    def close(self) -> None:
        """Close the game; safe to call more than once."""
        game = getattr(self, "game", None)
        if game is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - already closed
                game.close()
            self.game = None
        frames = getattr(self, "_frames", None)
        if frames is not None:
            frames.clear()


def make_env(config: EnvConfig | None = None, render_mode: str | None = None) -> VizDoomEnv:
    """Factory function to create a :class:`VizDoomEnv` instance."""
    return VizDoomEnv(config=config, render_mode=render_mode)
