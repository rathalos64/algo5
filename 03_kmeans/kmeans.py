#!/usr/bin/env python

import random
import math
import numpy as np
import cluster as cl

class Kmeans:
	def __init__(self, data, K, D, dist):
		self.data = data
		self.K = K
		self.D = D
		self.dist = dist

		self.centers = []
		self.clusters = None

	def set_parameters(self, **kwargs):
		self.data = kwargs["data"] if kwargs.get("data") != None else self.data
		self.K = 	kwargs["K"] if kwargs.get("K") != None else self.K
		self.D = 	kwargs["D"] if kwargs.get("D") != None else self.D
		self.dist = kwargs["dist"] if kwargs.get("dist") != None else self.dist

	def clear(self):
		for cluster in self.clusters:
			cluster.clear_observations()

	def seed(self):
		# seed cluster centers as proposed by MacQueen's second method (1967)
		self.centers = []
		for i in range(0, self.K):
			self.centers.append(list(random.choice(self.data)))

		# define clusters
		self.clusters = cl.Clusters()
		for i in range(0, self.K):
			self.clusters.append_cluster(cl.Cluster(
				cluster_id 		= f"c{i}",
				observations 	= [],
				mean 			= self.centers[i]
			))

	def run(self):
		bestmqe = -1
		i = 0

		while True:
			self.clear()

			# assign observations to cluster
			for observation in self.data:
				self.clusters.append_to_nearest_cluster(observation, self.dist, self.D)

			# define new means
			for cluster in self.clusters:
				cluster.center_mean()

			# compute mqe (mean quantisation error) in relation to |data|.
			# The average scattering across ALL(!) clusters.
			error = 0
			for cluster in self.clusters:
				error += cluster.approximation_error(self.dist, self.D)
			mqe = error / len(self.data)

			if mqe >= bestmqe and bestmqe != -1:
				break

			if mqe < bestmqe or bestmqe == -1:
				bestmqe = mqe

			i = i + 1

	# ================================================================================================
	# Quality measures
	# ================================================================================================

	# get_empty_clusters gets the empty clusters
	def get_empty_clusters(self):
		return list(filter(lambda x: len(x.observations) == 0, self.clusters))

	# get_dbi calculates the Davies - Bouldin Index (DB Index)
	# It assesses the used K's over how many K's actually exist in the data.
	def get_dbi(self):
		similarity = 0
		for i, ci in enumerate(self.clusters):
			similarity += max([self.__goodness_of_clustering(ci, cj) for j, cj in enumerate(self.clusters)
				if i != j])

		return similarity / self.K

	# goodness_of_clustering calculates ratio of the within cluster scatter
	# to the between cluster separation for two given clusters.
	# = averaged similarity of two clusters.
	def __goodness_of_clustering(self, c1, c2):
		if len(c1.observations) == 0 or len(c2.observations) == 0:
			return 0

		within_c1 = c1.approximation_error(self.dist, self.D) / len(c1.observations)
		within_c2 = c2.approximation_error(self.dist, self.D) / len(c2.observations)
		between = self.dist(c1.mean, c2.mean, self.D)

		return (within_c1 + within_c2) / between

	# get_sse calculates the Sum of Squared Error (SSE) which
	# measures of the discrepancy between the data and the found K clusters
	def get_sse(self):
		error = 0
		for ci in self.clusters:
			error += ci.approximation_error_squared(self.dist, self.D)

		return error

	# get_bic calculates the Baysian Information Criterion (BIC) or the Schwarz Criterion.
	# The calculation method for BIC consisting of the loglikelihood of the data and
	# the free parameters for k-means regarding K and D were taken from
	# http://www.aladdin.cs.cmu.edu/papers/pdfs/y2000/xmeans.pdf.
	#
	# Important to note are some apparent mistakes by the authors, regarding the derivation
	# of the likelihood function. This was unvealed and corrected by
	# https://github.com/bobhancock/goxmeans with his implementation.
	#
	# Especially with the document at
	# https://github.com/bobhancock/goxmeans/blob/master/doc/BIC_notes.pdf, he mentions
	# the flaws of the original authors formulation for BIC.
	#
	# As a sidenote, the original stackoverflow post can be found at http://bit.ly/2zXoEf2.
	def get_bic(self):
		ll = self.__loglikelihood()
		params = self.__free_params()

		bic = ((params / 2) * math.log(len(self.data))) - ll

		return bic

	# get_aic calculates the Akaike Information Criterion (AIC).
	# The likelihood function and number of free parameters are the same as for BIC.
	def get_aic(self):
		ll = self.__loglikelihood()
		params = self.__free_params()

		aic = (params - ll)
		return aic

	def __free_params(self):
		return (self.K - 1) * (self.K * self.D) + 1

	def __loglikelihood(self):
		N = len(self.data)

		# variance
		variance = 0
		for ci in self.clusters:
			variance += ci.approximation_error_squared(self.dist, self.D)
		variance /= (N - self.K)  

		# log likelihood of data
		ll = 0
		for ci in self.clusters:
			Ni = len(ci.observations)
			if Ni == 0:
				continue

			p1 = (Ni * math.log(Ni))
			p2 = (Ni * math.log(N))
			p3 = ((Ni * self.D) / 2) * math.log(2 * math.pi * variance)
			p4 = (Ni - 1) / 2
			
			ll += (p1 - p2 - p3 - p4)
		
		return ll
