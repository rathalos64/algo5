#!/usr/bin/env python

import numpy as np

from lib.gmm import ParameterGMMComponent
from lib.es import ParameterVector

def main():
	sigma = 0.05

	gmm_c = ParameterGMMComponent(
		0.5, 
		np.asarray([0, 0]),
		np.asarray([
			[2, 1.0], 
			[1.0, 2]
		])
	)

	gmm_c1 = ParameterGMMComponent(
		0.5, 
		np.asarray([0, 0]),
		np.asarray([
			[2, 1.0], 
			[1.0, 2]
		])
	)

	param_vec = ParameterVector([gmm_c, gmm_c1])
	print(param_vec)

	param_vec.mutate(sigma)
	print(param_vec)

if __name__ == "__main__":
	main()