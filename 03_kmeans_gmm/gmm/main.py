#!/usr/bin/env python

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from functools import reduce
from sklearn.datasets.samples_generator import make_blobs

from scipy.stats import norm, chi2

from lib.gmm import ParameterGMMComponent
from lib.gmm import ParameterGMMComponents
from lib.gmm import GaussianMixtureModel
from lib.es import ParameterVector
from lib.es import EvolutionStrategy

def main():
	# TODO:
	# * ES params via command line args
	# * run ES for k in [2, 20]
	# * for every k, generate k components
	# 		-> weight = 1/k
	#		-> means = run KMeans with k
	#		-> covs = np.cov(samples) ?
	#
	# * DBI for every k
	#		-> instead of MQE, the determinate of cov
	#
	# ploooooootttttt!

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

	components = ParameterGMMComponents()
	components.append(gmm_c)
	components.append(gmm_c1)
	param_vec = ParameterVector(components)

	# candidate = Solution(param_vec, sigma, 1, GaussianMixtureModel.loglikelihood)
	# print(candidate.evaluate(samples))

	# param_vec.mutate(sigma)
	# print(param_vec)
	# print(candidate.evaluate(samples))
	mu = 1
	param_vecs = []
	for _ in range(0, mu):
		param_vecs.append(param_vec)

	es = EvolutionStrategy(
		samples,
		1,
		1,
		5,
		EvolutionStrategy.TYPE_PLUS,
		0.05,
		0.5,
		-1,
		0.01,
		param_vecs,
		GaussianMixtureModel.loglikelihood)
	
	ok, validation_result = es.validate_parameter()
	if not ok:
		print(validation_result)
		return

	sigmas = []
	best_fitnesses = []
	for generations_cnt in es.run_iter():
		print(f"~~ {generations_cnt + 1} Generation")
		print("# Best solution candidate")
		print(es.best)
		best_fitnesses.append(es.best.fitness)
		sigmas.append(es.best.sigma)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_ylim([min(best_fitnesses) - 200, 0])
	plt.plot(range(0, len(best_fitnesses)), best_fitnesses, marker="o", markersize=4)

	fig = plt.figure()
	plt.plot(range(0, len(best_fitnesses)), sigmas, marker="o", markersize=4)

	plt.show()

if __name__ == "__main__":
	main()