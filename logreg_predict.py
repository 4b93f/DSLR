import pandas as pd
import sys
from clean import *
import clean
from lib import *

def predict(data, normalization_factors, weights, save: bool):
	for i in range(len(data[0])):
		data[:, i] = (data[:, i] - normalization_factors[i, 0]) / normalization_factors[i, 1]

	probabilities = []
	for x in data:
		row = []
		for i in weights:
			row.append(h(i, x))
		probabilities.append(row)

	predictions = []
	for i in range(len(probabilities)):
		house_index = np.argmax(probabilities[i])
		predictions.append(house_index)

	if save:
		with open("outputs/houses.csv", 'w') as file:
			file.write("Index,Hogwarts House\n")
			for index, prediction in enumerate(predictions):
				house_name = list(clean.house.keys())[list(clean.house.values()).index(prediction)]
				file.write(f"{index},{house_name}\n")

	return(predictions)

if __name__ == '__main__':
	try:
		dataPath = sys.argv[1]
		normalizationFactorsPath = sys.argv[2]
		weightsPath = sys.argv[3]

		df = pd.read_csv(dataPath)
		df = clean_df(df)
		df.drop(columns=drop, inplace=True)
		df = nan_replace(df)
		weights = pd.read_csv(weightsPath, dtype = float, header = None, index_col = 0).to_numpy()
		normalization_factors = pd.read_csv(normalizationFactorsPath).to_numpy()

		x = df.to_numpy()

		predict(x, normalization_factors, weights, True)
	except Exception as e:
		print("Usage: python predict.py <dataPath> <normalizationFactorsPath> <weightsPath>")
		sys.exit(1)
