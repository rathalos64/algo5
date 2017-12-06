#!/usr/bin/env python

import math
import numpy as np

class Utils():

	# softmax or also called normalized exponential function "squashes" a K-dimensional vector Z
	# or arbitrary real values into a vector S of real values in the range of [0, 1] 
	# that adds up to 1
	# https://en.wikipedia.org/wiki/Softmax_function
	@staticmethod
	def softmax(Z, j):
		return math.exp(Z[j]) / sum(np.exp(Z))

	# apply_softmax generates a new scaled vector of values by applying softmax to each value
	@staticmethod
	def apply_softmax(X):
		return [Utils.softmax(X, i) for i in range(0, len(X))]