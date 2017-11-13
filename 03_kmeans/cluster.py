#!/usr/bin/env python

from functools import reduce

class Clusters:
	def __init__(self):
		self.clusters = {}

	def __iter__(self):
		return iter(self.clusters.values())

	def append_cluster(self, cluster):
		self.clusters[cluster.id] = cluster

	def append_to_nearest_cluster(self, observation, dist, D):
		# find the minimized distance between an observation
		# and all cluster centers
		cluster = min([(cluster.id, dist(observation, cluster.mean, D)) 
						for cluster in self.clusters.values()], key = lambda x: x[1])

		self.clusters[cluster[0]].append_observation(observation)

class Cluster:

	# Observations is the terminology used in Wikipedia:
	# https://en.wikipedia.org/wiki/K-means_clustering
	def __init__(self, cluster_id, observations, mean):
		self.id = cluster_id
		self.observations = observations
		self.mean = mean

	def append_observation(self, observation):
		self.observations.append(observation)

	def clear_observations(self):
		self.observations = []

	# center_mean centers the mean of a cluster by
	# averaging over all observations within the clster
	def center_mean(self):
		if len(self.observations) != 0:
			self.mean = reduce(lambda x, y: x + y, self.observations) / len(self.observations)
			self.mean = self.mean[:-1]

	# approximation_error calculates the summed error of all observations
	# if they are approximated by the current cluster a given distance function.
	# It is the scattering of all points within the current cluster.
	def approximation_error(self, dist, D):
		error = 0
		for observations in self.observations:
			error += dist(observations, self.mean, D)

		return error
