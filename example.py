from rich.pretty import pprint

from boilerplate import run_env 
import jax.numpy as jnp
import numpy     as np

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

make_pad = lambda a, b : np.zeros(np.abs(a.shape[0] - b.shape[0]))
cat      = lambda c, p : np.concatenate((c, p), 0)

_save_callback_episode = 0
_save_callback_data    = [] # Global
def save_callback(env, obs, latent, step, episode, is_done):
    global _save_callback_episode # Sorry
    global _save_callback_data         # Sorry

    if latent is None:
        return

    h_t = latent['deter']
    s_t = np.expand_dims(np.ravel(env.symbols()), 0)

    _save_callback_data.append((h_t, s_t))

    if is_done:
        h_ts, s_ts = map(lambda a : np.concatenate(a, axis=0),
                         zip(*_save_callback_data))

        vars = np.concatenate((h_ts, s_ts), 1).T
        vars += 1e-8 # epsilon
        R = np.corrcoef(vars)
        print(R)
        path = f'data/{_save_callback_episode}.npz'
        np.savez(path, R)
        print(f'Wrote {path}')
        _save_callback_data = [] # Clear
        _save_callback_episode += 1

def main():
    run_env(save_callback, ablation_callback, size='nano')

if __name__ == '__main__':
    main()

