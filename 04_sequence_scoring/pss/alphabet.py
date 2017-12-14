#!/usr/bin/env python

import collections
from matplotlib.text import TextPath

# Alphabet provides information about sequence alphabets
# like nucleotides, proteins
class Alphabet:
	# nucleotide returns the alphabet for nucleotides (C, G, T, A)
	@staticmethod
	def nucleotide():
		return collections.OrderedDict(sorted(dict({
			"T": {
				"text": TextPath((-0.305, 0), "T", size=1),
				"color": "darkgreen",
				"weight": 0.25
			},
			"G": {
				"text": TextPath((-0.384, 0), "G", size=1),
				"color": "orange",
				"weight": 0.25
			},
			"A": {
				"text": TextPath((-0.35, 0), "A", size=1),
				"color": "red",
				"weight": 0.25
			},
			"C": {
				"text": TextPath((-0.366, 0), "C", size=1),
				"color": "blue",
				"weight": 0.25
			}
		}).items()))

		# nucleotide returns the alphabet for proteins (amino acids)
		@staticmethod
		def protein():
			# to be implemented
			return {}