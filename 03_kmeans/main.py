#!/usr/bin/env python
# native libs
import argparse
import math
import os
import json
import sys
import random
import shutil

# third-party libs
import numpy as np
from matplotlib import pyplot as pl
from matplotlib import patches as patc

# custom libs
import cluster as cl

class Application:
	def __init__(self, K, D, dist, input_file, is_graph, is_label_graph):
		self.K = K
		self.D = D
		self.dist = dist

		self.input_file = input_file
		self.is_graph = is_graph
		self.is_label_graph = is_label_graph

def main():
	parser = argparse.ArgumentParser(description='K-means clustering')

	# define command line arguments
	parser.add_argument("-f", metavar="<file>", action="store", required=True,
		dest="input_file",
		help="input CSV file")
	parser.add_argument("-graph", action="store_true", default=False, dest="is_graph",
		help="whether to create a graph (only for 2D)")
	parser.add_argument("-label_graph", action="store_true", default=False, dest="is_label_graph",
		help="whether to label the data points in graph")
	parser.add_argument("K", type=int, help="the number of clusters")
	args = parser.parse_args()

	# validate arguments
	validation_result = validate(args)
	if validation_result:
		print(json.dumps(validation_result, indent=4))
		sys.exit(1)

	# read data from CSV and append data point labeling (for graph)
	#data = np.genfromtxt(args.input_file, delimiter=",", skip_header=1)
	data = np.random.randint(100, size=(100,2))
	data = np.append(data, [[i+1] for i in range(0, len(data))], axis=1)

	# define application
	app = Application(
		K 		= args.K,
		D 		= len(data[0]) - 1,
		dist 	= euclidian_distance,

		input_file 		= args.input_file,
		is_graph 		= args.is_graph,
		is_label_graph 	= args.is_label_graph
	)

	if app.is_graph:
		shutil.rmtree("steps")
		os.makedirs("steps")

	# define cluster centers as proposed by MacQueen's second method. 
	# Choose k centers randomly from the data points.
	centers = []
	for i in range(0, app.K):
		centers.append(list(random.choice(data)))

	# define clusters
	clusters = cl.Clusters()
	for i in range(0, app.K):
		clusters.append_cluster(cl.Cluster(
			cluster_id 		= f"c{i}", 
			observations 	= [], 
			mean 			= centers[i]
		))

	# start k-means algorithm
	print("============================================================")
	print("[i] Starting k-means algortithm")
	print(f"# Dimensionality of data: {app.D}")
	print(f"# Number of given clusters: {app.K}")

	print("# Centers seed:")
	
	# remove label column
	mapped = list(map(lambda x: x[:-1], centers))
	for i, center in enumerate(mapped):
		print(f"# {i}: {center}")

	print("============================================================")

	bestmqe = -1
	i = 0

	while True:
		print(f"[i] {i} Iteration")

		for cluster in clusters:
			cluster.clear_observations()

		# assign observations to cluster
		for observation in data:
			clusters.append_to_nearest_cluster(observation, euclidian_distance, app.D)

		# visualize and save clustering step
		if app.is_graph and i == 0:
			save_figure_2d(app, data, clusters, i, "steps")

		# define new means
		for cluster in clusters:
			cluster.center_mean()

		print("# Clustering")
		for j, cluster in enumerate(clusters):

			# show only label
			mapped = list(map(lambda x: x[-1], cluster.observations))
			print(f"# {j}: {mapped}")

		# compute mqe (mean quantisation error) in relation to |data|.
		# The average scattering across ALL(!) clusters.
		error = 0
		for cluster in clusters:
			error += cluster.approximation_error(euclidian_distance, app.D)

		mqe = round(error / len(data), 8)
		print(f"> Average error: {mqe}")

		if mqe >= bestmqe and bestmqe != -1:
			print("> No improvment found. End.")
			break

		if mqe < bestmqe or bestmqe == -1:
			bestmqe = mqe

		# visualize and save clustering step
		if app.is_graph:
			save_figure_2d(app, data, clusters, i + 1, "steps")
			
		i = i + 1
		print("------------------------------------------------------------")

	# show quality measures only for non-single cluster number
	if app.K > 1:

		print("============================================================")
		print(f"[i] Quality measures for k = {app.K}")

		# number of empty clusters
		c_empty = len(list(filter(lambda x: len(x.observations) == 0, clusters)))
		print(f"# Empty clusters: {c_empty}")

		# calculate Davies - Bouldin Index
		# assess the used k's over how many k's actually exist in the data.
		# Omit empty clusters as well as comparing the goodness of clustering with ci = cj.
		similarity = 0
		for i, ci in enumerate(clusters):
			similarity += max([goodness_of_clustering(ci, cj, app) for j, cj in enumerate(clusters) 
				if i != j])

		DBI = similarity / app.K
		print(f"# Davies - Bouldin Index (DBI): {DBI}")

		print("============================================================")

def save_figure_2d(app, data, clusters, i, path):
	_, ax = pl.subplots()

	for cluster in clusters:
		if len(cluster.observations) == 0:
			continue

		observations = np.array(cluster.observations)
		
		pl.scatter(observations.T[0], observations.T[1], s=50.0, marker="o", label=cluster.id)
		pl.scatter(cluster.mean[0], cluster.mean[1], s=25.0, marker="x", c="black")

		if app.is_label_graph:
			# update font size and label the points
			pl.rcParams.update({"font.size": 8})
			for j, label in enumerate(observations.T[2]):
				pl.annotate(str(int(label)), (observations.T[0][j], observations.T[1][j]),
					textcoords="data")


		# draw ellipses showing clustering borders
		angle = calculate_angle_deg(observations)
		width = max(observations.T[0]) - min(observations.T[0])
		height = max(observations.T[1]) - min(observations.T[1])

		# calculate standard deviations and use it as scaling factor for the ellipses
		sx = math.sqrt(sum(math.pow(observation[0] - cluster.mean[0], 2) for observation in observations)
			/ len(observations))
		sx = round(sx, 1)
		if sx == 0:
			sx = 2

		sy = math.sqrt(sum(math.pow(observation[1] - cluster.mean[1], 2) for observation in observations)
			/ len(observations))
		sy = round(sy, 1)
		if sy == 0:
			sy = 2

		width += 2.5 * sx
		height += 2.5 * sy

		ellipse = patc.Ellipse(xy=cluster.mean, width=width, height=height, angle=angle,
				fill=False, color="black", ls="dotted")
		ax.add_patch(ellipse)

	ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	ax.autoscale()

	pl.xlabel("x")
	pl.ylabel("y")

	factor = 2
	min_x = data.T[0].min() * factor
	max_x = data.T[0].max() * factor

	min_y = data.T[1].min() * factor
	max_y = data.T[1].max() * factor
	
	pl.xlim([min_x, max_x])
	pl.ylim([min_y, max_y])

	suffices = ["st", "nd", "rd"]
	suffix = "th"
	if (i - 1) in range(0, 3):
		suffix = suffices[(i - 1)]

	pl.title(f"{i}{suffix} iteration")
	pl.savefig(f"{path}/{i}.png", bbox_inches="tight")

def validate(args):
	validation_result = {}

	if args.K < 1:
		validation_result["K"] = "Invalid number of clusters"

	if not os.path.exists(args.input_file):
		validation_result["input_file"] = f"File {args.input_file} does not exist"
	else:	
		data = None
		try: 
			data = np.genfromtxt(args.input_file, delimiter=",", skip_header=1, loose=False, invalid_raise=True)
		except ValueError as e:
			validation_result["input_file"] = str(e)
			return validation_result

		nobservations = len(data)

		if nobservations < args.K:
			validation_result["K"] = "The number of clusters cannot exceed the number of data points"
		
		if nobservations == 0:
			validation_result["input_file"] = "Empty input file given; no data points found"

		if args.is_graph:
			if len(data[0]) != 2:
				validation_result["graph"] = "Graph cannot be generated for non 2D data"
	
	return validation_result

def euclidian_distance(x, y, d):
	dist = 0
	for i in range(0, d):
		dist += math.pow(y[i] - x[i], 2)

	return math.sqrt(dist)

# goodness_of_clustering calculates the ratio of the averaged
# variability within two clusters to the variability between two clusters.
# = similarity of two clusters
def goodness_of_clustering(c1, c2, app):
	if len(c1.observations) == 0 or len(c2.observations) == 0:
		return 0

	within_c1 = c1.approximation_error(app.dist, app.D) / len(c1.observations)
	within_c2 = c2.approximation_error(app.dist, app.D) / len(c2.observations)

	between = euclidian_distance(c1.mean, c2.mean, app.D)	
	return (within_c1 + within_c2) / between

# calculate_angle computes the angle for the ellipse
# which should fit the data using least squares linear regression
def calculate_angle_deg(observations):
	# calculate averages
	avg_x = sum(observation[0] for observation in observations) / len(observations)
	avg_y = sum(observation[1] for observation in observations) / len(observations)

	# differences to averages
	x_diff = [observation[0] - avg_x for observation in observations]
	y_diff = [observation[1] - avg_y for observation in observations]

	# squared to prevent negativ values
	x_diff_squared = [math.pow(element, 2) for element in x_diff]
	x_diff_squared_sum = sum(x_diff_squared)
	if x_diff_squared_sum == 0:
		x_diff_squared_sum = 1

	# get the slope of the trend line
	slope = sum(x * y for x,y in zip(x_diff, y_diff)) / x_diff_squared_sum

	# the slope is equal to the tangent of the angle
	angle = math.atan(slope)

	# return in degrees
	return math.degrees(angle)

if __name__ == "__main__":
	main()