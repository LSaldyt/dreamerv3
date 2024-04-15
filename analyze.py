import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import csv

from pathlib import Path

def main():
    exp_names = ['sparse_nano', 'dense_nano']
    n         = 128
    thresh    = 0.95

    with open('data.csv', 'w') as datafile:
        writer = csv.DictWriter(datafile, fieldnames=['i', 'exp', 'max_corr', 'mean_corr', 'n_corr'])
        writer.writeheader()

        for exp_name in exp_names:
            path      = Path(exp_name)
            data_path = path / 'data'
            plot_path = path / 'plots'

            for i, arr_path in enumerate(data_path.iterdir()):
                data   = np.load(arr_path)
                data   = data['arr_0'][:n, n:]
                n_corr = np.sum(np.max(np.abs(data), axis=0) > thresh)

                max_corr  = np.max(np.abs(data))
                mean_corr = np.mean(np.abs(data))
                print(exp_name)
                print(f'{i:<3} n_corr > {thresh} = {n_corr} max = {max_corr:1.10f}')

                sns.heatmap(data)
                plt.savefig(plot_path / f'corr_{i}.png')
                plt.clf()
                writer.writerow(dict(i=i, exp=exp_name, 
                                     max_corr=max_corr, 
                                     mean_corr=mean_corr, 
                                     n_corr=n_corr))
                datafile.flush()

                # if i == 16:
                #     break

if __name__ == '__main__':
    main()
