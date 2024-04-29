import pandas as pd
import sys
from clean import *
from lib import *

def gradient_descent(x, y, epochs, learningRate):
    nrows, nfeatures = len(x), len(x[0])

    theta = np.zeros(nfeatures)
    for _ in range(epochs):
        theta -= learningRate * gradient(x, y, nrows, theta, h)

    return theta

def logistic_regression(x, y, optimizationFunction, epochs, learningRate):
    thetas = []
    for i in np.unique(y):
        y_i = np.where(y == i, 1, 0)
        thetas.append(optimizationFunction(x, y_i, epochs, learningRate))

    return thetas

def standardize(x):
    return (x - np.mean(x)) / np.std(x)

def train(x, y, optimizationFunction, save: bool):
    normalization_factors = []
    for i in range(len(x[0])):
        column_mean, column_std = mean(x[:, i]), std(x[:, i])
        x[:, i] = standardize(x[:, i])
        normalization_factors.append(np.array([column_mean, column_std]))
    normalization_factors = np.array(normalization_factors)

    if save:
        with open("outputs/normalization_factors.csv", 'w') as file:
            file.write("mean,std\n")
            for normalization_factor in normalization_factors:
                file.write(f"{normalization_factor[0]},{normalization_factor[1]}\n")

    epochs = 100
    learningRate = 0.01
    weights = logistic_regression(x, y, optimizationFunction, epochs, learningRate)

    df = pd.DataFrame(columns = range(len(weights[0])))
    for i in range(len(weights)):
        df.loc[i] = weights[i]

    weights = df.to_numpy()
    if save:
        with open("outputs/weights.csv", 'w') as file:
            df.to_csv(file, header = False)

    return(normalization_factors, weights)

if __name__ == '__main__':
    try:
        dataPath = sys.argv[1]

        df = pd.read_csv(dataPath)
        df = clean_df(df)
        df.dropna(inplace=True)
        y = np.array(df['House'])
        df.drop(columns=drop, inplace=True)
        x = df.to_numpy()
        train(x, y, gradient_descent, True)
    except Exception as e:
        print("Usage: python logreg_train.py <path>")
        sys.exit(1)
