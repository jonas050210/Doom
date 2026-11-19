"""Gymnasium-compatible ViZDoom environment wrapper for NeuralDoom-Lab."""

from collections import deque
import os
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import vizdoom as vzd

from src.actions import build_action_matrix, get_action_names
from src.config import EnvConfig, SCENARIOS


class VizDoomEnv(gym.Env):
    """Gymnasium wrapper around ViZDoom using pixel observations.

    Observations are returned in channels-first ``(C, H, W)`` ``uint8`` format
    (e.g., ``(4, 84, 84)`` for 4 stacked 84x84 grayscale frames), ready for
    PyTorch and Stable-Baselines3 ``CnnPolicy``.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        if self.config.scenario not in SCENARIOS:
            available = ", ".join(sorted(SCENARIOS.keys()))
            raise ValueError(
                f"Unknown scenario '{self.config.scenario}'. Available: {available}"
            )

        cfg_filename = SCENARIOS[self.config.scenario]["cfg"]
        cfg_path = os.path.join(vzd.scenarios_path, cfg_filename)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"ViZDoom scenario config not found at: {cfg_path}"
            )

        self.game = vzd.DoomGame()
        self.game.load_config(cfg_path)

        # Configure display visibility
        visible = self.config.window_visible or (self.render_mode == "human")
        self.game.set_window_visible(visible)

        # Configure pixel format
        if self.config.grayscale:
            self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
            channels_per_frame = 1
        else:
            self.game.set_screen_format(vzd.ScreenFormat.RGB24)
            channels_per_frame = 3

        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_seed(self.config.seed)
        self.game.init()

        # Build discrete action space
        available_buttons = self.game.get_available_buttons()
        self.action_matrix = build_action_matrix(
            available_buttons,
            include_use_attack=self.config.include_use_attack,
        )
        self.action_names = get_action_names(
            include_use_attack=self.config.include_use_attack
        )
        self.action_space = spaces.Discrete(len(self.action_matrix))

        # Observation space: (C * frame_stack, H, W) uint8
        total_channels = channels_per_frame * self.config.frame_stack
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(total_channels, self.config.frame_height, self.config.frame_width),
            dtype=np.uint8,
        )

        self._frames: Deque[np.ndarray] = deque(maxlen=self.config.frame_stack)
        self._last_raw_frame: Optional[np.ndarray] = None
        self._episode_reward: float = 0.0
        self._episode_steps: int = 0

    def _preprocess_frame(self, screen_buffer: Optional[np.ndarray]) -> np.ndarray:
        """Resize a ViZDoom screen buffer to (C, H, W) uint8."""
        if screen_buffer is None:
            c = 1 if self.config.grayscale else 3
            return np.zeros(
                (c, self.config.frame_height, self.config.frame_width),
                dtype=np.uint8,
            )

        self._last_raw_frame = screen_buffer

        if self.config.grayscale:
            # GRAY8 buffer from ViZDoom has shape (H, W)
            resized = cv2.resize(
                screen_buffer,
                (self.config.frame_width, self.config.frame_height),
                interpolation=cv2.INTER_AREA,
            )
            return resized[np.newaxis, :, :].astype(np.uint8)

        # RGB24 buffer from ViZDoom has shape (3, H, W) or (H, W, 3)
        if screen_buffer.ndim == 3 and screen_buffer.shape[0] == 3:
            hwc = np.transpose(screen_buffer, (1, 2, 0))
        else:
            hwc = screen_buffer
        resized = cv2.resize(
            hwc,
            (self.config.frame_width, self.config.frame_height),
            interpolation=cv2.INTER_AREA,
        )
        return np.transpose(resized, (2, 0, 1)).astype(np.uint8)

    def _get_stacked_obs(self) -> np.ndarray:
        """Concatenate stacked frames along the channel axis."""
        return np.concatenate(list(self._frames), axis=0)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)

        self.game.new_episode()
        self._episode_reward = 0.0
        self._episode_steps = 0

        state = self.game.get_state()
        frame = self._preprocess_frame(state.screen_buffer if state else None)
        self._frames.clear()
        for _ in range(self.config.frame_stack):
            self._frames.append(frame)

        return self._get_stacked_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} for {self.action_space}")

        button_vec = self.action_matrix[int(action)]
        reward = float(self.game.make_action(button_vec, self.config.frame_skip))
        self._episode_reward += reward
        self._episode_steps += 1

        done = self.game.is_episode_finished()
        if not done:
            state = self.game.get_state()
            frame = self._preprocess_frame(state.screen_buffer if state else None)
        else:
            frame = self._preprocess_frame(None)

        self._frames.append(frame)
        obs = self._get_stacked_obs()

        # In my_way_home, reaching the vest gives a +1.0 terminal reward.
        # In health_gathering, surviving longer gives positive cumulative reward.
        is_success = bool(done and reward > 0.0)
        info: Dict[str, Any] = {
            "episode_reward": self._episode_reward,
            "episode_steps": self._episode_steps,
            "is_success": is_success,
        }

        terminated = bool(done)
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._last_raw_frame
        return None

    def close(self) -> None:
        if hasattr(self, "game") and self.game is not None:
            self.game.close()


def make_env(config: Optional[EnvConfig] = None, render_mode: Optional[str] = None) -> VizDoomEnv:
    """Factory function to create a VizDoomEnv instance."""
    return VizDoomEnv(config=config, render_mode=render_mode)
