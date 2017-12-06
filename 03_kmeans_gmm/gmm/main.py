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

	for generations_cnt in es.run_iter():
		print(f"~~ {generations_cnt + 1} Generation")
		print("# Best solution candidate")
		print(es.best)

	# get best candidate
	best = es.best
	ellipses = []
	for component in best.parameter_vector:
		ellipses.append(cov_ellipse(component.cov, nsig=2))

	print(ellipses)

	# draw estimated ellipsed
	for i, component in enumerate(best.parameter_vector):
		print(i)
		ax.scatter(component.mean[0], component.mean[1], linestyle="None", c="b", marker="x")
		
		ellip = Ellipse(xy=component.mean, width=ellipses[i][0], height=ellipses[i][1], angle=ellipses[i][2])
		ellip.set_facecolor('none')
		ellip.set_edgecolor('black')
		ax.add_artist(ellip)

	plt.show()

def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations. 
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov); print(val)
    width, height = 2 * np.sqrt(np.abs(val[:, None] * r2))
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return [width, height, rotation]

if __name__ == "__main__":
	main()