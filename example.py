from rich.pretty import pprint

from boilerplate import run_env 

def run_sparse_autoencoder(losses):
    pass

def run_info_regularization(losses):
    pass

def ablation_callback(data, feats, dists, losses, extra, config):
    print('ABLATION CALLBACK')
    pprint((data, feats, dists, losses, extra))
    if config.sparse_autoencoder:
        run_sparse_autoencoder(losses)
    if config.info_regularization:
        run_info_regularization(losses)
    exit()

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

