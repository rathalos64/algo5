#!/usr/bin/env python
#
# Example call:
# $ python main.py -plot -N 100 5
#
# TODO: 
# * uniform vs gaussian mixture model
#	given as paramters
#
# * parameter per json file for uniform and gaussian mixture model
#
# * 3D-dimensional plotting

import os
import math
import shutil
import argparse

import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

from lib.kmeans import Kmeans

class Application:
	def __init__(self, N, D, K, min_size, max_size):
		self.N = N
		self.D = D
		self.K = K

		self.min_size = min_size
		self.max_size = max_size

def main():
	parser = argparse.ArgumentParser(description='K-means clustering')

	# define command line arguments
	parser.add_argument("-D", type=int, metavar="<D>", action="store", default=2, dest="D",
		help="dimensionality of data (default 2)")
	parser.add_argument("-min", type=float, metavar="<MIN>", action="store", default=0.0,
		dest="min_size", help="the minimum threshold for the generated data")
	parser.add_argument("-max", type=float, metavar="<MAX>", action="store", default=10000.0,
		dest="max_size", help="the maximum threshold for the generated data")
	parser.add_argument("-max_iter", type=int, metavar="<MAX_ITER>", action="store", default=300,
		dest="max_iter", help="the maximum number of iterations for kmeans")
	parser.add_argument("-plot", action="store_true", default=False, 
		dest="plot", help="create / save plots on every iteration (if D == 2)")
	parser.add_argument("-plot_path", metavar="<PATH>", action="store", default="iterations", 
		dest="plot_path",
		help="path for plots")
	parser.add_argument("-N", type=int, metavar="<N>", action="store", required=True, dest="N",
		help="number of samples taken randomly from a uniform distribution")
	parser.add_argument("K", type=int, help="the number of clusters")
	args = parser.parse_args()

	# define application
	app = Application(
		D 		= args.D,
		N		= args.N,
		K 		= args.K,

		min_size = args.min_size,
		max_size = args.max_size,
	)

	# create plot directory
	if app.D == 2 and args.plot:
		path = args.plot_path
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path, exist_ok=False)

	data, _ = make_blobs(
		n_samples=app.N, 
		n_features=app.D, 
		cluster_std=5.0, 
		center_box=(-10.0, 10.0), 
		centers=10
	)
	#data = np.random.uniform(low=app.min_size, high=app.max_size, size=(app.N,app.D))
	# #data = np.random.normal(size=(app.N,app.D))
	# data1 = np.random.normal(loc=-16.0, scale=1.0, size=(app.N,app.D))
	# data2 = np.random.normal(loc=100.0, scale=20.0, size=(app.N,app.D))
	# data = np.append(data1, data2, axis=0)


	kmeans = Kmeans(data, app.K, app.D, "euclidian", args.max_iter)

	print("============================================================")
	print("[i] Starting k-means algorithm")
	print(f"# Number of samples from uniform distribution: {app.N}")
	print(f"# Min size: {app.min_size}")
	print(f"# Max size: {app.max_size}")
	print(f"# Dimensionality of data: {app.D}")
	print(f"# Number of clusters K: {app.K}")
	print("============================================================")
	print("[i] Seeding centers")
	kmeans.seed()

	# show means
	means = np.array(list(map(lambda x: x.mean, kmeans.clusters)))
	print(means)

	# show initial state
	if app.D == 2 and args.plot:
		print("============================================================")
		print(f"[i] Save plots at '{path}'")

		fig1 = plt.figure()
		ax1 = fig1.add_subplot(111)
		ax1.scatter(data.T[0], data.T[1], s=25.0, marker=".", c="orange")
		ax1.scatter(means.T[0], means.T[1], s=12.5, marker="x", c="black")

		plt.title("Dataset")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title(f"Seeding")
		plt.savefig(f"{path}/-1.png", bbox_inches="tight")

	print("============================================================")
	for i, res in enumerate(kmeans.run_iter()):
		print(f"# {i} Iteration")
		clusters = res[0]
		mqe = res[1]

		print(f"> Average distance (MQE) {mqe}")
		if app.D == 2 and args.plot:
			save_2d_graph(i, clusters, path, [[[0.02, 0.02, f"MQE = {str(mqe)}"]]])

	print("============================================================")
	print(f"[i] Quality measures for k = {app.K}")
	print(f"# Number of iterations: {kmeans.get_n_iter()}")
	print(f"# Empty clusters: {len(kmeans.get_empty_clusters())}")
	print(f"# [MQE] Mean Quantisation Error: {kmeans.get_mqe()}")
	if app.K > 1:
		print(f"# [DBI] Davies Bouldin Index: {kmeans.get_dbi()}")
	print(f"# [SSE] Sum of Squared Errors: {kmeans.get_sse()}")
	print(f"# [AIC] Akaike Information Criterion: {kmeans.get_aic()}")
	print(f"# [BIC] Baysian Information Criterion: {kmeans.get_bic()}")

def save_2d_graph(i, clusters, path, texts=[]):
	# show graph
	fig1 = plt.figure()
	fig1.subplots_adjust()
	ax1 = fig1.add_subplot(111)

	for text in texts:
		plt.gcf().text(*text[0])

	for cluster in clusters:
		if len(cluster.observations) == 0:
			continue

		observations = np.array(cluster.observations)
		ax1.scatter(observations.T[0], observations.T[1], s=25.0, marker=".")
		ax1.scatter(cluster.mean[0], cluster.mean[1], s=12.5, marker="x", c="black")

		plt.title("Dataset")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title(f"{i} iteration")
		plt.savefig(f"{path}/{i}.png", bbox_inches="tight")

if __name__ == "__main__":
	main()