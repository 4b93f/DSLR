import sys
import pandas as pd
from lib import *
from clean import clean_df, nan_replace
import numpy as np

def describe(path : str):
    data = pd.read_csv(path)
    data = clean_df(data)
    data = nan_replace(data)
    features = data.columns
    print(data.describe())
    print(' ' * 17, end="")
    print(f'{"Count":>17}|{"Mean":>17}|{"Std":>17}|{"Min":>17}|{"25%":>17}|{"50%":>17}|{"75%":>17}|{"Max":>17}|{"Unique":>17}')
    for i in range(0, len(features)):
        d = data.iloc[:, i]
        if data[features[i]].dtypes == 'object' : continue

        _count = count(d)
        _mean = mean(d)
        _min = min(d)
        _max = max(d)
        _std = std(d)
        _25 = percentile(d, 0.25)
        _50 = percentile(d, 0.50)
        _75 = percentile(d, 0.75)
        _unique = unique(d)

        print(f'{features[i]:16.16}', end="")
        print(f'{_count:>18.6f}|', end="")
        print(f'{_mean:>17.6f}|', end="")
        print(f'{_std:>17.6f}|', end="")
        print(f'{_min:>17.6f}|', end="")
        print(f'{_25:>17.6f}|', end="")
        print(f'{_50:>17.6f}|', end="")
        print(f'{_75:>17.6f}|', end="")
        print(f'{_max:>17.6f}|', end="")
        print(f'{_unique:>17.6f}|')

if __name__ == '__main__':
    try:
        args = sys.argv[1]
        describe(args)
    except Exception as e:
        print("Usage: python describe.py <path>")
        sys.exit(1)
