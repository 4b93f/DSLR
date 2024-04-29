from pandas import DataFrame
import numpy as np
from enum import Enum

drop = ['House', 'BestHand', 'Age', 'Arithmancy', 'Defense Against the Dark Arts', 'Potions', 'Care of Magical Creatures', 'Flying']

class House(Enum):
    RAVENCLAW = 0,
    SLYTHERIN = 1,
    GRYFFINDOR = 2,
    HUFFLEPUFF = 3

def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

def denormalize(data_normalized, data_min, data_max):
    return data_normalized * (data_max - data_min) + data_min

def mylen(array):
    len = 0
    for x in array: len += 1 if not np.isnan(x) else 0
    return len

def count(df : list):
    count = mylen(df)
    return count

def mean(df : list):
    mean = df.sum() / mylen(df)
    return mean

def std(df : DataFrame):
    res = [(x - mean(df))** 2 for x in df]
    return (sum(res) / (mylen(df) -1)) ** 0.5

def min(df : list): #Todo Not sure about that because -1
    x = 10000
    for i in df:
        if i < x or x == -1: x = i
    return x

def max(df):
    x = -1
    for i in df:
        if i > x or x == -1: x = i
    return x

def percentile(df : list, p : float):
    df, n = sorted(df), mylen(df)

    k = (n - 1) * p
    index = int(k)
    n_index = int(k + 1)
    return df[index] + (df[n_index] - df[index]) * (k - index)


def unique(df : list):
    i = 0
    stock = []
    for d in df:
        if d not in stock:
            i += 1
            stock.append(d)
    return i


def data_from_house(df : list):
    R,S,G,H = [],[],[],[]

    for _, d in df.iterrows():
        R.append(d) if d['House'] == House.RAVENCLAW.value else 0
        S.append(d) if d['House'] == House.SLYTHERIN.value else 0
        G.append(d) if d['House'] == House.GRYFFINDOR.value else 0
        H.append(d) if d['House'] == House.HUFFLEPUFF.value else 0
    return [R, S, G, H]

def course_from_house(house_data, course):
    list = []
    for h in house_data:
        tmp = []
        for i in range(len(h)): tmp.append(h[i][course])
        list.append(tmp)
    return list[0], list[1], list[2], list[3]

def len_from_range(min: int, max: int, score : list):
    len = 0
    for s in score:
        if not max:
            len +=1 if s >= min else 0
            continue
        len += 1 if s < max and s >= min else 0
        if s > max: break
    return len

def get_homogeneous(df : list, score : list, course : str):
    list, score = [], sorted(score)
    for i in range(len(df[:-1])): list.append(len_from_range(df[i], df[i + 1], score))
    list.append(len_from_range(df[-1:], None, score))
    return [course, np.std(list), list]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(theta, x):
    return sigmoid(np.dot(theta, x.T))

def gradient(x, y, nrows, theta, h):
    return((h(theta, x) - y).dot(x) / nrows)

def cost(theta, x, y, nrows):
    return -(np.dot(y, np.log(h(theta, x))) + np.dot(1 - y, np.log(1 - h(theta, x)))) / nrows
