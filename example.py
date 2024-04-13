from rich.pretty import pprint

from boilerplate import run_env 
import jax.numpy as jnp
import jax

def run_sparse_autoencoder(enc, dec, feats):
    print('Called sparse autoencoder')
    feats  = {k : jnp.squeeze(v) for k, v in feats.items()}
    latent = enc(feats)
    result = dec(latent)
    # TODO Use result in the network, e.g. replace h_t with result
    if len(latent.shape) < 2:
        latent = jnp.expand_dims(latent, 0)
    return dict(
        sparse_latent=latent,
        loss=jnp.linalg.norm(latent, ord=1, axis=-1) # L1 norm regularization term
        )

def run_info_regularization(discrete, continuous, feats):
    discrete_predictions   = discrete(feats)
    continuous_predictions = continuous(feats)
    print(discrete_predictions)
    print(continuous_predictions)
    # TODO Add the mutual information proxy function here...
    raise NotImplementedError('No mutual info proxy loss defined')
    loss = jnp.array([0.0])
    return dict(
        discrete=discrete_predictions,
        continuous=continuous_predictions,
        loss=loss
        )

def ablation_callback(data, feats, dists, extra, config):
    results = dict()
    losses  = dict()
    if config.sparse_autoencoder:
        sparse = run_sparse_autoencoder(extra['sparse_enc'], extra['sparse_dec'], feats)
        losses['sparse_autoencoder']  = sparse['loss']
        results.update(**{k : v for k, v in sparse.items() if k != 'loss'})
    if config.info_regularization:
        info   = run_info_regularization(extra['info_discrete'], extra['info_continuous'], feats)
        losses['info_regularization'] = info['loss']
        results.update(**{k : v for k, v in info.items() if k != 'loss'})
    return results, losses

def save_callback(env, obs, latent, step, episode):
    pass

def main():
    run_env(save_callback, ablation_callback)

if __name__ == '__main__':
  main()

# TODO Rewrite the save callback
# from sklearn.metrics import mutual_info_score
# def calc_MI(x, y, bins):
#     ''' Courtesy of stack overflow '''
#     c_xy = np.histogram2d(x, y, bins)[0]
#     mi = mutual_info_score(None, None, contingency=c_xy)
#     return mi
# 
# Don't grow the data list. Dump every episode.
# try:
#     obs_large = env.render(size=np.array([128, 128]))
# except Exception as e:
#     print(e)
#     obs_large = None
# if len(data) < episode + 1:
#     data.append([])
# else:
#     data[episode].append(tuple(map(np.ravel, (obs['image'], latent['deter'], latent['stoch'], env.symbols()))))
# # to_mx = lambda l : np.concatenate([np.expand_dims(el, 0) for el in l], 0)
# avg      = lambda l : sum(l) / len(l)
# make_pad = lambda a, b : np.zeros(np.abs(a.shape[0] - b.shape[0]))
# cat      = lambda c, p : np.concatenate((c, p), 0)
# 
# with open('data.csv', 'w') as csvfile:
#     fieldnames = ['episode', 'h_z_mi', 'obs_h_mi', 'obs_z_mi', 'sym_h_mi', 'sym_z_mi']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
# 
#     for episode, saved in enumerate(data):
#         # first dim is number of steps, variable
#         # 12288, 1024, 1024
#         l = []
#         for obs, h, z, sym in saved:
#             local = []
#             for source in (obs, sym):
#                 pad = make_pad(source, h)
#                 for latent in (h, z):
#                     padded = cat(latent, pad)
#                     local.append(calc_MI(padded, source, 16))
#             l.append((calc_MI(h, z, 16),) + tuple(local))
#         row = dict(episode=episode, **{f : avg(l[i]) for i, f in enumerate(fieldnames[1:])})
#         print(row)
#         writer.writerow(row)

