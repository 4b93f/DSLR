from pandas import DataFrame
from lib import mean
import numpy as np
from datetime import date


def nan_replace(df : DataFrame):
	for d in df:
		if df[d].isna().sum() > 0:
			for i in range(0, len(df[d])):
				if df[d].isna()[i]: df.loc[i, d] = mean(df[d])
	return df

def clean_df(df : DataFrame):
	global house
	house = {
		float('NaN') : -1,
		'Ravenclaw' : 0,
		'Slytherin' : 1,
		'Gryffindor' : 2,
		'Hufflepuff' : 3
	}
	hand = {
		'Left' : 0,
		'Right' : 1
	}

	df = df.rename(columns={'Hogwarts House' : 'House', 'Best Hand' : 'BestHand', 'Birthday' : 'Age'})
	df.House = df.House.map(house).astype(int)
	df.BestHand = df.BestHand.map(hand)

	for i, age in enumerate(df.Age): df.loc[i, 'Age'] = (date.today() - date.fromisoformat(age)).days // 365
	df['Age'] = df['Age'].astype(int)
	return df.drop(columns=['Index', 'First Name', 'Last Name'])
