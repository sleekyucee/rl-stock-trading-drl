#frame_stack.py

import numpy as np
from collections import deque
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

class FrameStack(ObservationWrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)

        shp = env.observation_space.shape
        if len(shp) == 2:
            h, w = shp
            c = 1
        elif len(shp) == 3:
            h, w, c = shp
        else:
            raise ValueError(f"Unsupported observation shape: {shp}")

        self._obs_is_2d = (len(shp) == 2)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(h, w, c * num_stack),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._obs_is_2d:
            obs = obs[..., None]
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if self._obs_is_2d:
            obs = obs[..., None]
        self.frames.append(obs)
        return self._get_ob(), reward, done, truncated, info

    def observation(self, observation):
        if self._obs_is_2d:
            observation = observation[..., None]
        self.frames.append(observation)
        return self._get_ob()

    def _get_ob(self):
        while len(self.frames) < self.num_stack:
            h, w, c = self.observation_space.shape[:-1] + (self.observation_space.shape[-1] // self.num_stack,)
            self.frames.append(np.zeros((h, w, c), dtype=self.observation_space.dtype))
        return np.concatenate(list(self.frames), axis=2)
