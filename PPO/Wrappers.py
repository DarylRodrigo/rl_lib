import time
import random

import numpy as np
import cv2
import gym


class DataAgumentationWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)

  def augment_grayscale(self, obs):
      # TODO Shape gets fucked here, becomes (64, 64, 1).
      return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

  def augment_crop(self, obs):
      """Pad 12 pixels as per paper and crop randomly."""
      obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
      # TODO: What to use here ? E.g. could do BORDER_REPLICATE.
      # Paper doesn't specify.
      # Update: Just saw they have code. But for some reason they did everything
      # manually instead of using OpenCV. Maybe I'm missing something.
      obs_padded = cv2.copyMakeBorder(
        obs_bgr, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(0, 0, 0)
      )
      (x1, y1), _ = self._gen_rectangle_pos(24, 24)
      obs_padded = obs_padded[x1:x1 + 64, y1:y1 + 64, :]
      return cv2.cvtColor(obs_padded, cv2.COLOR_BGR2RGB)

  def _gen_rectangle_pos(self, x, y):
      """Generate random positions for a rectangle."""
      x1, y1 = (np.random.randint(0, x), np.random.randint(0, y))
      x2, y2 = (np.random.randint(0, x), np.random.randint(0, y))
      return (x1, y1), (x2, y2)

  def augment_cutout(self, obs, color=(0, 0, 0)):
      """Generate a random black rectangle."""
      p1, p2 = self._gen_rectangle_pos(obs.shape[0], obs.shape[1])
      obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
      cv2.rectangle(obs_bgr, p1, p2, color=color, thickness=-1)
      return cv2.cvtColor(obs_bgr, cv2.COLOR_BGR2RGB)

  def augment_cutout_color(self, obs):
      """Generate a random rectangle with a random color."""
      col = (
        np.random.randint(0, 256),
        np.random.randint(0, 256),
        np.random.randint(0, 256)
      )
      return self.augment_cutout(obs, col)

  def augment_flip(self, obs):
      """Flip around y axis."""
      return cv2.flip(obs, flipCode=1)

  def augment_rotate(self, obs):
      """Randomly rotate in 90 degree steps."""
      rotations = [
        None,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180
      ]
      random_rotation = rotations[np.random.randomint(4)]
      if not random_rotation:
          return obs
      return cv2.rotate(obs, random_rotation)

  def augment_random_convolution(self, obs):
      pass  # TODO

  def augment_color_jitter(self, obs):
      """Add random noise to HSV representation of observation."""
      # TODO This is wrong.
      obs_hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV) / [179, 255, 255]
      noise = np.random.normal(scale=.1, size=obs.shape)
      noisy_hsv_obs = ((obs_hsv + noise) * [179, 255, 255]).astype('uint8')
      return cv2.cvtColor(noisy_hsv_obs, cv2.COLOR_HSV2RGB)

  def observation(self, obs):
    # TODO: Let's assume procgen observation space: (64, 64, 3)
    print(obs)
    # print(self.augment_grayscale(obs).shape)
    cv2.imshow("obs", obs)
    cv2.imshow("augm", self.augment_color_jitter(obs))
    cv2.waitKey(0)
    import pdb; pdb.set_trace()
    return obs


def test():
    env = gym.make('procgen:procgen-coinrun-v0')
    env = DataAgumentationWrapper(env)

    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            break


test()
