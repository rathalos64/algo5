#!/usr/bin/env python
import numpy as np
import math
from matplotlib import pyplot as pl
from functools import reduce
from mpl_toolkits.mplot3d import axes3d, Axes3D

def main():
	input_file = "input-2d.csv"
	
	# number of clusters
	k = 3

	# dimension (# of features)
	d = 2

	# define cluster centers
	centers = []
	if d == 2:
		centers = np.array([
			[9, 0],
			[8, 1],
			[4, 0]
		])

	if d == 3:
		centers = np.array([
			[9, 0, 1],
			[8, 1, 1]
		])

	# define clusters
	clusters = []
	mappings = {}
	for i in range(0, k):
		clusters.append({
			"id": f"c{i}",
			"points": [],
			"center": centers[i]
		})
		mappings[f"c{i}"] = i
	
	data = np.genfromtxt(input_file, delimiter=",", skip_header=1)

	# start k-means
	bestmqe = -1

	i = 0
	while True:
		# clear points out of cluster
		for cluster in clusters:
			cluster["points"] = []

		# assign data points to cluster
		for point in data:
			cluster_id = find_nearest_cluster(point, clusters, d)

			# append to best cluster
			clusters[mappings[cluster_id]]["points"].append(point)

		# define new means
		for cluster in clusters:
			cluster["center"] = reduce(lambda x, y: x + y, cluster["points"]) / len(cluster["points"])

		# compute mqe
		sumdist = 0
		for cluster in clusters:
			for point in cluster["points"]:
				sumdist += euclidian_distance(point, cluster["center"], d)

		mqe = sumdist / len(data)
		print(f"[{i}] {mqe}")

		if mqe == bestmqe:
			break

		if mqe < bestmqe or bestmqe == -1:
			bestmqe = mqe

		i = i + 1
	# stop k-means

	# visualize 2d data
	# define axis
	factor = 2
	min_x = data.T[0].min() - factor
	max_x = data.T[0].max() + factor

	min_y = data.T[1].min() - factor
	max_y = data.T[1].max() + factor

	pl.axis([min_x, max_x, min_y, max_y])

	if d == 2:
		markers = ["x", "o", ">"]

		for i, cluster in enumerate(clusters):
			points = np.array(cluster["points"])
			pl.scatter(points.T[0], points.T[1], marker=markers[i])
			pl.scatter(cluster["center"][0], cluster["center"][1], marker="*", c="black")

		pl.show()

	if d == 3:
		markers = ["x", "o"]
		fig = pl.gcf()
		ax = Axes3D(fig)

		for i, cluster in enumerate(clusters):
			points = np.array(cluster["points"])
			ax.scatter(points.T[0], points.T[1], points.T[2], marker=markers[i])
			ax.scatter(cluster["center"][0], cluster["center"][1], cluster["center"][2], marker="*", c="black")

		pl.show()

	# print(data)
	# print(data.T)

	# print("c1 = " + str(centers[0]))
	# print("c2 = " + str(centers[1]))
	# print("X5 = " + str(data[7]))
	# print("d(X5, c1) = " + str(euclidian_distance(data[5], centers[0], 2)))
	# print("d(X5, c2) = " + str(euclidian_distance(data[5], centers[1], 2)))

	# # show initial data - for plotting x's and y's are needed, therefore transpose
	# pl.scatter(data.T[0], data.T[1])
	# pl.scatter(centers.T[0], centers.T[1], marker="x")
	# pl.plot(centers.T[0], centers.T[1])
	# pl.show()

def find_nearest_cluster(p, clusters, d):
	mindist = -1
	target_cluster = None

	for cluster in clusters:
		dist = euclidian_distance(p, cluster["center"], d)
		if dist < mindist or mindist == -1:
			mindist = dist
			target_cluster = cluster["id"]

	return target_cluster

def euclidian_distance(x, y, d):
	dist = 0
	for i in range(0, d):
		dist += math.pow(y[i] - x[i], 2)

	return math.sqrt(dist)

if __name__ == "__main__":
	main()