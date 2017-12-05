#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.datasets.samples_generator import make_blobs

from lib.gmm import ParameterGMMComponent
from lib.gmm import GaussianMixtureModel
from lib.es import ParameterVector
from lib.es import Solution

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
		np.asarray([5, 5]),
		np.asarray([
			[2, 1.0], 
			[1.0, 2]
		])
	)

	samples, _ = make_blobs(
        n_samples=500,
        n_features=2,
        centers=[gmm_c.mean, gmm_c1.mean],
        cluster_std=gmm_c.cov,
    )

	df = pd.DataFrame.from_records(samples, columns=["X", "Y"])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(df["X"], df["Y"], linestyle="None", c="g", marker=".")
	plt.draw()

	param_vec = ParameterVector([gmm_c, gmm_c1])
	print(param_vec)

	candidate = Solution(param_vec, sigma, 1, GaussianMixtureModel.loglikelihood)
	print(candidate.evaluate(samples))

	param_vec.mutate(sigma)
	print(param_vec)
	print(candidate.evaluate(samples))

	plt.show()

if __name__ == "__main__":
	main()