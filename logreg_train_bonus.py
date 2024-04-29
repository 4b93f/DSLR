import pandas as pd
import sys
from clean import *
from lib import *
from logreg_train import train, gradient_descent

def stochastic_gradient_descent(x, y, epochs, learningRate, batchSize = 100):
    nrows = len(x)
    nfeatures = len(x[0])

    theta = np.zeros(nfeatures)
    for i in range(epochs):
        xy = np.c_[x.reshape(nrows, nfeatures), y.reshape(nrows)]
        rng = np.random.default_rng()
        rng.shuffle(xy)

        for start in range(0, nrows, batchSize):
            stop = start + batchSize
            x_batch = xy[start:stop, :-1]
            y_batch = xy[start:stop, -1]
            theta -= learningRate * gradient(x_batch, y_batch, nrows, theta, h)

    return theta

if __name__ == '__main__':
    dataPath = sys.argv[1]

    df = pd.read_csv(dataPath)
    df = clean_df(df)
    df.dropna(inplace=True)
    y = np.array(df['House'])
    df.drop(columns=drop, inplace=True)
    x = df.to_numpy()

    train(x, y, stochastic_gradient_descent, True)
