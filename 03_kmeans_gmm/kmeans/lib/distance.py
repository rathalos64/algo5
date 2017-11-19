#!/usr/bin/env python

import math

class Distance:
    @staticmethod
    def get_distance_method(key):
        mappings = {
            "euclidian": Distance.euclidian_distance
        }

        if key not in mappings.keys():
            return Distance.euclidian_distance

        return mappings[key]

    @staticmethod
    def euclidian_distance(x, y, d):
        dist = 0
        for i in range(0, d):
            dist += math.pow(y[i] - x[i], 2)

        return math.sqrt(dist)