#!/usr/bin/env python

import numpy as np
import pandas as pd

# PSS describes methods for calculating the position specific scoring
class PSS():
    def __init__(self, sources, alphabet, weights, avg_sequence_length, pseudocount):
        self.sources = sources
        self.alphabet = alphabet
        self.weights = weights
        self.avg_sequence_length = avg_sequence_length
        self.pseudocount = pseudocount

    # build_frequency_matrix computes a position frequence matrix (PFM)
    # out of the given sources. The sequences are based on the given 
    # alphabet.
    def build_frequency_matrix(self):
        frequency_matrix = {}
        for onegram in self.alphabet:
            frequency_matrix[onegram] = np.zeros(self.avg_sequence_length)

        for source in self.sources:
            for i in range(0, self.avg_sequence_length):
                onegram = source[i]
                # exclude invalid characters
                if onegram not in frequency_matrix.keys():
                    continue

                frequency_matrix[onegram][i] += 1

        return pd.DataFrame(list(frequency_matrix.values()), dtype=int, 
                index=self.alphabet)

    # build_probability_matrix computes a position probability matrix (PPM)
    # based on the given frequency matrix. Zero values are removed
    # by adding a given pseudocount prevent zero-frequency problems.
    def build_probability_matrix(self, pfm):
        probability_matrix = round(pfm / len(self.sources), 2)
        return probability_matrix.clip(self.pseudocount)

    # build_weight_matrix constructs the position weight matrix (PWM)
    def build_weight_matrix(self, ppm):
        return np.log(ppm)

    # score computes a score for a given target sequence, given
    # the position weight matrix (PWM)
    def score(self, target, pwm):
        score = 0
        for i, onegram in enumerate(target):
            score += pwm[i][onegram]
        return score
        