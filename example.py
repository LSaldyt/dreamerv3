import numpy as np
import csv

from dreamerv3.embodied.core.basics import convert

from sklearn.metrics import mutual_info_score

def calc_MI(x, y, bins):
    ''' Courtesy of stack overflow '''
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': '~/logdir/run1',
      'run.train_ratio': 64,
      'run.log_every': 30,  # Seconds
      'batch_size': 16,
      'jax.prealloc': False,
      'encoder.mlp_keys': '$^',
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
      # 'jax.platform': 'cpu',
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

  import crafter
  from embodied.envs import from_gym
  crafter_env = crafter.Env()  # Replace this with your Gym env.

  env = from_gym.FromGym(crafter_env, obs_key='image')  # Or obs_key='vector'.
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  # TODO move into a per-step function
  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  data = []

  def save_callback(env, obs, latent, step, episode):
      # try:
      #     obs_large = env.render(size=np.array([128, 128]))
      # except Exception as e:
      #     print(e)
      #     obs_large = None
      if len(data) < episode + 1:
          data.append([])
      else:
          data[episode].append(tuple(map(np.ravel, (obs['image'], latent['deter'], latent['stoch']))))

  try:
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay')
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    embodied.run.train(agent, env, replay, logger, args, save_callback)
  except KeyboardInterrupt:
      pass

  # to_mx = lambda l : np.concatenate([np.expand_dims(el, 0) for el in l], 0)
  avg = lambda l : sum(l) / len(l)

  with open('data.csv', 'w') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=['episode', 'h_z_mi', 'obs_h_mi', 'obs_z_mi'])

      print(data)
      for episode, saved in enumerate(data):
          # first dim is number of steps, variable
          # 12288, 1024, 1024
          l = []
          for obs, h, z in saved:
              print('hz mi')
              print(calc_MI(h, z, 16))
              pad = np.zeros(obs.shape[0] - h.shape[0])
              h_pad = np.concatenate((h, pad), 0)
              z_pad = np.concatenate((z, pad), 0)
              print('obs-h mi')
              print(calc_MI(obs, h_pad, 16))
              print('obs-z mi')
              print(calc_MI(obs, z_pad, 16))
              l.append((calc_MI(h, z, 16), calc_MI(obs, z_pad, 16), calc_MI(obs, h_pad, 16)))
          print(dict(episode=episode, h_z_mi=avg(l[0]), obs_z_mi=avg(l[1]), obs_h_mi=avg(l[2])))
          writer.writerow(dict(episode=episode, h_z_mi=avg(l[0]), obs_z_mi=avg(l[1]), obs_h_mi=avg(l[2])))

  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  main()
