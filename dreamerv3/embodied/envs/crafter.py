import embodied
import numpy as np


class Crafter(embodied.Env):

  def __init__(self, task, size=(64, 64), outdir=None, seed=None):
    assert task in ('reward', 'noreward')
    import crafter
    self.inner_env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
    if outdir:
      outdir = embodied.Path(outdir)
      self.inner_env = crafter.Recorder(
          self.inner_env, outdir,
          save_stats=True,
          save_video=False,
          save_episode=False,
      )
    self._achievements = crafter.constants.achievements.copy()
    self._done = True

  @property
  def obs_space(self):
    spaces = {
        'image': embodied.Space(np.uint8, self.inner_env.observation_space.shape),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        'log_reward': embodied.Space(np.float32),
    }
    spaces.update({
        f'log_achievement_{k}': embodied.Space(np.int32)
        for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self.inner_env.action_space.n),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      image = self.inner_env.reset()
      return self._obs(image, 0.0, {}, is_first=True)
    image, reward, self._done, info = self.inner_env.step(action['action'])
    reward = np.float32(reward)
    return self._obs(
        image, reward, info,
        is_last=self._done,
        is_terminal=info['discount'] == 0)

  def _obs(
      self, image, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    log_achievements = {
        f'log_achievement_{k}': info['achievements'][k] if info else 0
        for k in self._achievements}
    return dict(
        image=image,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        log_reward=np.float32(info['reward'] if info else 0.0),
        **log_achievements,
    )

  def render(self, *args, **kwargs):
    return self.inner_env.render(*args, **kwargs)
