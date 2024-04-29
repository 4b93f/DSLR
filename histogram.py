from matplotlib import pyplot as plt
import pandas as pd
import sys
from clean import *
from lib import *
import os

def histogram(path : str):
	df = pd.read_csv(path)
	df = clean_df(df)
	df = nan_replace(df)
	house_data = data_from_house(df)
	course_list = df.columns 
	for course in course_list[2:]:
		R, S, G ,H = course_from_house(house_data, course)
		R, S, G, H = np.array(R).astype(float), np.array(S).astype(float), np.array(G).astype(float), np.array(H).astype(float)
		ALL = ["Ravenclaw", course, R], ["Slytherin", course, S], ["Gryffindor", course, G], ["Hufflepuff",course, H]
		plt.figure()
		for a in ALL:
			_, bins, _ = plt.hist(a[2], alpha=0.5, label=a[0], histtype='step')
			plt.title(a[1]), plt.xlabel('Score'), plt.ylabel('Number of students'), plt.legend(loc='upper right'), 
	plt.show()
	if "SAVE" in os.environ:
		plt.savefig('plots/histogram.png')

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: python histogram.py <path>")
		sys.exit(1)
	args = sys.argv[1]
	histogram(args)