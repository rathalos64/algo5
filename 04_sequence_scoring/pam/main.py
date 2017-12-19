#!/usr/bin/env

import sys
import math
import numpy as np
import pandas as pd

def main():
	a_priori_f = "testing/a_priori"
	l_matrix_f = "testing/l_matrix"

	target_f = "testing/target2"

	# the evoluationary distance
	pam_step = 1

	# floating point tolerance for sum of a priori probabilites
	precision = 2

	# pseudocount for preventing zero probabilities
	pseudocount = 0.00000000001

	# read a priori probabilities
	a_priori_df = pd.DataFrame.from_csv(a_priori_f, sep=";", index_col=None)

	# validate; must sum up to 1.<precision> (precision = # of significant digits)
	# meaning up to <precision> floating digits, it must be equal
	# abs_tol refers to the minimum absolute tolerance
	summed = a_priori_df.sum(axis=1)[0]
	if not math.isclose(summed, 1, abs_tol=10**-precision):
		print(f"[x] invalid prior probabilities; must sum up to 1 with tolerance {precision}: {summed}")
		sys.exit(1)

	# extract the alphabet
	alphabet_index = a_priori_df.columns 
	alphabet = list(alphabet_index)

	# read in the L matrix
	l_matrix_df = pd.DataFrame.from_csv(l_matrix_f, sep=";")

	# validate matrix; must correspond to alphabet
	columns_valid = (l_matrix_df.columns == alphabet).all()
	index_valid = (l_matrix_df.index == alphabet).all()
	if not columns_valid & index_valid:
		print("[x] invalid L matrix; columns and rows indices must correspond expected alphabet")
		sys.exit(1)

	# get number of replacements
	n = l_matrix_df.sum(axis=1).sum(axis=0)

	# extend upper diagnoal triangle
	L = l_matrix_df + l_matrix_df.T

	# compute pab
	pab = pd.DataFrame.from_records(data=np.zeros(L.shape), index=alphabet_index)
	pab.columns = alphabet_index

	# non-diagonal
	for i in alphabet:
		for j in alphabet:
			if i == j:
				continue

			pab[i][j] = L[i][j] / (100 * n * a_priori_df[i][0])
			if pab[i][j] < pseudocount:
				pab[i][j] = pseudocount

	# diagonal
	for i in alphabet:
		pab[i][i] = 1 - pab.loc[i].sum()

	pabn = pab**pam_step

	# prevent zero probabilities
	for i in alphabet:
		for j in alphabet:
			if pabn[i][j] < pseudocount:
				pabn[i][j] = pseudocount

	# compute pam
	pamn = pd.DataFrame.from_records(data=np.zeros(L.shape), index=alphabet_index)
	pamn.columns = alphabet_index

	for i in alphabet:
		for j in alphabet:
			pamn[i][j] = math.log10(pabn[i][j] / a_priori_df[i][0]) * 10

	print(pamn)

	# scoring time
	sequences = []
	with open(target_f) as fh:
		sequences = fh.readlines()

	sequences = [str.strip(sequence) for sequence in sequences]


	print("======================================================")
	i = 0
	while i < len(sequences) - 1:
		seq1 = sequences[i]
		seq2 = sequences[i+1]
		print(f"# Score: {seq1} vs {seq2}")

		score = 0
		for j, _ in enumerate(seq1):
			score += pamn[seq1[j]][seq2[j]]

		print(f"# x = {score}")
		if i + 1 != len(sequences) - 1:
			print()

		i = i + 2
	print("======================================================")

if __name__ == "__main__":
	main()