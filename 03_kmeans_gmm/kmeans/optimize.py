#!/usr/bin/env

import numpy as np
import math
from matplotlib import pyplot as pl

from kmeans.kmeans import Kmeans

def main():
	data = np.random.uniform(low=0.0, high=1000.0, size=(1000,2))

	kmeans = Kmeans(data, 1, len(data[0]), euclidian_distance)

	kmin = 2
	kmax = 20
	k_range = range(kmin, kmax)

	dbis = []
	sses = []
	c_empty = []
	bics = []
	aics = []
	for i, k in enumerate(k_range):	
		kmeans.set_parameters(K=k)
		kmeans.seed()
		kmeans.run()
		dbis.append([kmin+i, kmeans.get_dbi()])
		sses.append([kmin+i, kmeans.get_sse()])
		c_empty.append([kmin+i, len(kmeans.get_empty_clusters())])
		bics.append([kmin+i, kmeans.get_bic()])
		aics.append([kmin+i, kmeans.get_aic()])

	fig1 = pl.figure()
	ax1 = fig1.add_subplot(111)
	for cluster in kmeans.clusters:
		if len(cluster.observations) == 0:
			continue

		observations = np.array(cluster.observations)
		
		ax1.scatter(observations.T[0], observations.T[1], s=50.0, marker="o", label=cluster.id)
		ax1.scatter(cluster.mean[0], cluster.mean[1], s=25.0, marker="x", c="black")

	pl.title("Data")
	pl.xlabel("x")
	pl.ylabel("y")

	# DBI
	dbis = np.array(dbis)
	fig2 = pl.figure()
	ax2 = fig2.add_subplot(111)
	ax2.plot(dbis.T[0], dbis.T[1])

	pl.title("Davies Bouldin Index")
	pl.xlabel("K")
	pl.ylabel("DBI")

	# SSE
	sses = np.array(sses)
	fig3 = pl.figure()
	ax3 = fig3.add_subplot(111)
	ax3.plot(sses.T[0], sses.T[1])

	pl.title("Sum of Squared Error")
	pl.xlabel("K")
	pl.ylabel("SSE")

	ax3.get_xaxis().get_major_formatter().set_scientific(False)
	ax3.get_yaxis().get_major_formatter().set_scientific(False)

	# Empty cluster
	c_empty = np.array(c_empty)
	fig4 = pl.figure()
	ax4 = fig4.add_subplot(111)
	ax4.plot(c_empty.T[0], c_empty.T[1])

	pl.title("Number of empty clusters")
	pl.xlabel("K")
	pl.ylabel("|Empty Cs|")

	# BIC & AIC
	bics = np.array(bics)
	aics = np.array(aics)
	fig5 = pl.figure()
	ax5 = fig5.add_subplot(111)
	ax5.plot(bics.T[0], bics.T[1], label="BIC", marker='o', markersize=4)
	ax5.plot(aics.T[0], aics.T[1], label="AIC", marker='o', markersize=4)
	ax5.legend()

	pl.title("BIC & AIC")
	pl.xlabel("K")
	pl.ylabel("Criteria value")

	pl.show()

def euclidian_distance(x, y, d):
	dist = 0
	for i in range(0, d):
		dist += math.pow(y[i] - x[i], 2)

	return math.sqrt(dist)

if __name__ == "__main__":
	main()
