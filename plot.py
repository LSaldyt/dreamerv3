import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('data.csv')
    sns.set_style('whitegrid')
    sns.lineplot(df, hue='exp', x='i', y='mean_corr')
    plt.savefig('plot.png')
    plt.clf()

if __name__ == '__main__':
    main()
