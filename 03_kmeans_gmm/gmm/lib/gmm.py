#!/usr/bin/env python

import math

import numpy as np
from functools import reduce

from lib.utils import Utils

# ParameterGMMComponent stores the parameter for a component
# of a Gaussian Mixture Model. Mostly a Gaussian Mixture Model
# consists for > 1 components (otherwise, it would a simple
# multivariate normal distribution). 
# 
# The parameter for a component consists of θ = {ω, μ, Σ}, with 
#	ω ... the weight of the component (how much data is expected there)
#	μ ... the centroid
# 	Σ ... the covariance matrix, consisting of the variance and the covariance
#
# Each component parameter can be stored within a ParameterVector to resemble
# all component parameters of a Gaussian Mixture Model.
#
# Therefore, the ParameterGMMComponent implements a mutate method, mutating
# each value of each parameter. In order to emulate mutation in nature, the mutation
# step generates a mutation value for each parameter value out of a normal distribution
# with zero mean and a specific variance. This variance is passed to the mutation method
# and refers to the mutation width.
class ParameterGMMComponent():
	def __init__(self, weight, mean, cov):
		self.weight = weight
		self.mean = mean
		self.cov = cov

	# mutate mutates for every parameter of the component {ω, μ, Σ}
	# each individual value with a specific value drawn from a N(0, σ) distribution.
	# σ refers to the mutation-width or the variance of the mutation step.
	def mutate(self, sigma):
		# mutate weights
		self.weight += np.random.normal(0.0, sigma)

		# mutate mean
		# create ndarray of same size, all filled with 
		# normal distributed values and add them up
		normal = np.ndarray(
			self.mean.shape, 
			buffer=np.array([np.random.normal(0.0, sigma) for i in range(0, self.mean.size)]),
			dtype=float
		)
		self.mean = self.mean + normal

		# mutate cov
		# same procedure applyable as for mean
		normal = np.ndarray(
			self.cov.shape, 
			buffer=np.array([np.random.normal(0.0, sigma) for i in range(0, self.cov.size)]),
			dtype=float
		)
		self.cov = self.cov + normal

	def __str__(self):
		return ("GMM Component (\n"
		"\t~~ ω: {}\n" 
		"\t~~ μ: {}\n"
		"\t~~ Σ: {}\n"
		"\t)").format(
			self.weight, 
			self.mean,
			self.cov.tolist()
		)

class ParameterGMMComponents():
	def __init__(self, components=[]):
		self.components = components

	def __iter__(self):
		return iter(self.components)

	def append(self, component):
		self.components.append(component)

	def normalize(self):
		# normalize weights (should sum up to 1.0) by applying softmax
		weights = [component.weight for component in self.components]
		weights = Utils.apply_softmax(weights)

		for i, _ in enumerate(weights):
			self.components[i].weight = weights[i]

		# equal covariances (should be the same)
		covs = [component.cov for component in self.components]
		for i, cov in enumerate(covs):

			# get all indices all covariances in the matrix
			# 
			# np.triu_indices gets the indices upper triangle in the covariance matrix
			# with that we get e.g covxy and covyx, which obviously should be the same
			xs, ys = list(np.triu_indices(cov.shape[0], 1, m=cov.shape[1]))
			xs = xs.reshape(len(xs), 1)
			ys = ys.reshape(len(ys), 1)
			target_indices = np.concatenate((xs, ys), axis=1)

			# compare all covariances (covxy and covyx, covxz and covzx, ...)
			for index in target_indices:
				x = index[0]
				y = index[1]
				covariances = [cov[x][y], cov[y][x]]
				
				# pick randomly (uniformly) the future covariance for e.g covxy and covyx
				target_covariance = np.random.choice(covariances)

				# apply to original covariance matrix of ith component
				self.components[i].cov[x][y] = target_covariance
				self.components[i].cov[y][x] = target_covariance

class MultivariateNormal():
	# pdf calculates the density of a given observation x based
	# on a multivariate normal distribution parametrised by the mean and the covariance matrix
	@staticmethod
	def pdf(x, mean, cov):
		D = len(x)

		# for python: * is element-wise multiplication, not matrix multiplication
		# use dot instead
		mahalanobis = (x - mean).T.dot(np.linalg.inv(cov)).dot((x - mean))
		normalizing = math.pow(2 * math.pi, D / 2) * math.pow(np.linalg.det(cov), 2)

		# calling math.exp with high positive value (> ~720) results in
		# floating point overflow
		# https://stackoverflow.com/questions/36268077/overflow-math-range-error-for-log-or-exp
		adjusted = -0.5 * mahalanobis
		if adjusted > 0:
			adjusted *= -1
			
		return math.exp(adjusted) / normalizing

class GaussianMixtureModel():
	@staticmethod
	def _likelihood(x, components):
		likelihood = 0
		for component in components:
			likelihood += (component.weight * MultivariateNormal.pdf(x, component.mean, component.cov))
		return likelihood

	@staticmethod
	def loglikelihood(X, components):
		return reduce(lambda acc, x: acc + math.log(GaussianMixtureModel._likelihood(x, components)), X, 0)