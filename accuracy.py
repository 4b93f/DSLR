import pandas as pd
import numpy as np
import sys
from clean import *
import random
from logreg_train import train, gradient_descent
from logreg_train_bonus import stochastic_gradient_descent
from logreg_predict import predict

def data_splitter(x, y, ratio):
    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(len(x)):
        if (random.uniform(0, 1) < ratio):
            x_train.append(x[i])
            y_train.append(y[i])
        else:
            x_test.append(x[i])
            y_test.append(y[i])

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def accuracy(x, y, optimizationFunction):
    x_train, y_train, x_test, y_test = data_splitter(x, y, 0.8)
    normalization_factors, weights = train(x_train, y_train, optimizationFunction, False)
    predictions = predict(x_test, normalization_factors, weights, False)
    nconcordances = 0
    for i in range(len(predictions)):
        nconcordances += 1 if predictions[i] == y_test[i] else 0

    return (nconcordances / len(predictions))

if __name__ == '__main__':
    try :
        dataPath = sys.argv[1]

        df = pd.read_csv(dataPath)
        df = clean_df(df)
        y = np.array(df['House'])
        df.drop(columns=["House", "BestHand", "Age", "Arithmancy", "Potions", "Care of Magical Creatures", "Flying"], inplace=True) 
        df = nan_replace(df)
        x = df.to_numpy()

        print(f"Accuracy with gradient descent is {accuracy(x, y, gradient_descent) * 100}%")
        print(f"Accuracy with stochastic gradient descent is {accuracy(x, y, stochastic_gradient_descent) * 100}%")
    except Exception as e:
        print("Usage: python accuracy.py <path>")
        sys.exit(1)
