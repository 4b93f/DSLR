from lib import *
import numpy as np

if __name__ == '__main__':
	t = [2.3, 6.1, 5.5, 7.6]

	print(f'{np.percentile(t, 25)}')
	print(f'{percentile(t, .25)}')