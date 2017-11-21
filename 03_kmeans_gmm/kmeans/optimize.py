#!/usr/bin/env

import argparse
import os
import shutil

import numpy as np
from matplotlib import pyplot as plt

from lib.kmeans import Kmeans

class Application:
	def __init__(self, N, D, K_range, min_size, max_size):
		self.N = N
		self.D = D
		self.K_range = K_range

		self.min_size = min_size
		self.max_size = max_size

def main():
	parser = argparse.ArgumentParser(description='K-means k optimization')

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
		dest="plot", help="create / save optimization plots")
	parser.add_argument("-plot_path", metavar="<PATH>", action="store", default="optimization", 
		dest="plot_path",
		help="path for optimization plots")
	parser.add_argument("-N", type=int, metavar="<N>", action="store", required=True, dest="N",
		help="number of samples taken randomly from a uniform distribution")
	parser.add_argument("-K", type=str, metavar="<K_FROM-K_TO>", default="2-20", dest="K_range", 
		help="the range of clusters to be tested")
	args = parser.parse_args()

	k_range = args.K_range.split("-")

	# define application
	app = Application(
		D 		= args.D,
		N		= args.N,
		K_range = range(int(k_range[0]), int(k_range[1])),

		min_size = args.min_size,
		max_size = args.max_size,
	)

	# create plot directory
	if args.plot:
		path = args.plot_path
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path, exist_ok=False)

	data = np.random.uniform(low=app.min_size, high=app.max_size, size=(app.N,app.D))
	kmeans = Kmeans(data, 0, app.D, "euclidian", args.max_iter)

	measures = {
		"MQE": {"values": [], "xlabel": "K", "ylabel": "MQE", "title": "Mean Quantisation Error"},
		"DBI": {"values": [], "xlabel": "K", "ylabel": "DBI", "title": "Davies-Bouldin Index"},
		"SSE": {"values": [], "xlabel": "K", "ylabel": "SSE", "title": "Sum of Squared Error"},
		"EC": {"values": [], "xlabel": "K", "ylabel": "Empty Clusters", "title": "Number of Empty Clusters"},
		"IT": {"values": [], "xlabel": "K", "ylabel": "Iterations", "title": "Number of Iterations"},
		"AICBIC": {"values": {"BIC": [], "AIC": []}, "xlabel": "K", "ylabel": "Criteria Value", "title": "BIC & AIC"}
	}

	k_min = int(k_range[0])

	print("============================================================")
	print("[i] Optimizing k for k-means algorithm")
	print(f"# Number of samples from uniform distribution: {app.N}")
	print(f"# Min size: {app.min_size}")
	print(f"# Max size: {app.max_size}")
	print(f"# Dimensionality of data: {app.D}")
	print(f"# Clustering range: {args.K_range}")
	print("============================================================")

	for i, k in enumerate(app.K_range):	
		print(f"## K = {k}")

		kmeans.set_parameters(K=k)
		kmeans.seed()
		kmeans.run()

		mqe = kmeans.get_mqe()
		print(f"> Mean Quantisation Error (MQE): {mqe}")

		dbi = kmeans.get_dbi()
		print(f"> Davies-Bouldin Index (DBI): {dbi}")

		sse = kmeans.get_sse()
		print(f"> Sum of Squared Error (SSE): {sse}")

		aic = kmeans.get_aic()
		print(f"> Akaike Information Criteria (AIC): {aic}")

		bic = kmeans.get_bic()
		print(f"> Baysian Information Criteria (BIC): {bic}")

		ec = len(kmeans.get_empty_clusters())
		print(f"> Number of Empty Clusters: {ec}")

		it = kmeans.get_n_iter()
		print(f"> Number of Iterations: {it}")

		if args.plot:
			measures["MQE"]["values"].append([k_min+i, mqe])
			measures["DBI"]["values"].append([k_min+i, dbi])
			measures["SSE"]["values"].append([k_min+i, sse])
			measures["EC"]["values"].append([k_min+i, ec])
			measures["IT"]["values"].append([k_min+i, it])
			measures["AICBIC"]["values"]["BIC"].append([k_min+i, bic])
			measures["AICBIC"]["values"]["AIC"].append([k_min+i, aic])

		print("============================================================")

	if args.plot:
		for key, measure in measures.items():
			if key == "AICBIC":
				aic_values = np.array(measure["values"]["AIC"])
				bic_values = np.array(measure["values"]["BIC"])
				fig = plt.figure()
				ax = fig.add_subplot(111)
				ax.plot(aic_values.T[0], aic_values.T[1], label="BIC", marker='o', markersize=4)
				ax.plot(bic_values.T[0], bic_values.T[1], label="AIC", marker='o', markersize=4)
				ax.legend()

				# disable scientific notation (e.g 1xe^4)
				ax.get_xaxis().get_major_formatter().set_scientific(False)
				ax.get_yaxis().get_major_formatter().set_scientific(False)

				plt.title(measure["title"])
				plt.xlabel(measure["xlabel"])
				plt.ylabel(measure["ylabel"])

				plt.savefig(f"{path}/{key.lower()}.png", bbox_inches="tight")
				continue

			values = np.array(measure["values"])
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.plot(values.T[0], values.T[1], marker='o', markersize=4)

			# disable scientific notation (e.g 1xe^4)
			ax.get_xaxis().get_major_formatter().set_scientific(False)
			ax.get_yaxis().get_major_formatter().set_scientific(False)

			plt.title(measure["title"])
			plt.xlabel(measure["xlabel"])
			plt.ylabel(measure["ylabel"])

			plt.savefig(f"{path}/{key.lower()}.png", bbox_inches="tight")

	# fig1 = plt.figure()
	# ax1 = fig1.add_subplot(111)
	# for cluster in kmeans.clusters:
	# 	if len(cluster.observations) == 0:
	# 		continue

	# 	observations = np.array(cluster.observations)
		
	# 	ax1.scatter(observations.T[0], observations.T[1], s=50.0, marker="o", label=cluster.id)
	# 	ax1.scatter(cluster.mean[0], cluster.mean[1], s=25.0, marker="x", c="black")

if __name__ == "__main__":
	main()
