from matplotlib import pyplot as plt
import pandas as pd
import sys
from clean import *
from lib import *
from scipy.stats import pearsonr
import os

def my_pearsonr(x, y):
    if len(x) != len(y): raise ValueError('The lists must have the same length')

    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    dev_x, dev_y = [xi - mean_x for xi in x], [yi - mean_y for yi in y]
    covariance = sum(dxi * dyi for dxi, dyi in zip(dev_x, dev_y))
    std_dev_x, std_dev_y= (sum(dxi ** 2 for dxi in dev_x) / len(x))**0.5, (sum(dyi ** 2 for dyi in dev_y) / len(y))**0.5

    return covariance / (std_dev_x * std_dev_y)

def scatter_plot(path : str):
    df = pd.read_csv(path)
    df = clean_df(df)
    df.dropna(inplace=True)

    stock = ["", "",  0]
    for col in df.columns:
        for j in range(len(df.columns)):
            if col == df.columns[j]: continue
            l = my_pearsonr(normalize(df[col]), normalize(df[df.columns[j]]))
            if (abs(l) > abs(stock[2])): stock = [col, df.columns[j],l]
    plt.scatter(df[stock[0]], df[stock[1]], c=df.House, cmap='viridis'), plt.title(f'{stock[0]} vs {stock[1]}'), plt.show()
    if "SAVE" in os.environ:
        plt.savefig('plots/scatter_plot.png')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <path>")
        sys.exit(1)
    args = sys.argv[1]
    scatter_plot(args)
