from matplotlib import pyplot as plt
import pandas as pd
import sys
from clean import *
from lib import *
import seaborn as sns
import os

def pair_plot(path : str):
    df = pd.read_csv(path)
    df = clean_df(df)
    df.dropna(inplace=True)

    df.drop(columns=['Care of Magical Creatures', 'Arithmancy', 'Defense Against the Dark Arts', 'BestHand', 'Age'], inplace=True)
    sns.pairplot(df, hue='House', height=1, aspect=2)
    plt.tight_layout()
    plt.show()
    if "SAVE" in os.environ:
        plt.savefig('plots/pair_plot.png')

if __name__ == '__main__':
    try:
        args = sys.argv[1]
        pair_plot(args)
    except Exception as e:
        print("Usage: python pair_plot.py <path>")
        sys.exit(1)
