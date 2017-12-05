#!/usr/bin/env python

# ParameterVector stores a number of parameters lists
# for a specific problem. Mathematically it can be imagined
# as a vector of thetas ([θ1, θ2, θ3, ...]), where θ 
# consists of the actual parameter. For example
#
# Multivariate Normal Distribution
#	ParameterVector = [θ], with θ = {μ, Σ}
#
# Gaussian Mixture Model
#	ParameterVector = [θ1, θ2, ..] with
#		θ1 = {ω1, μ1, Σ1},
#		θ2 = {ω2, μ2, Σ2}
#
# Each ParameterVector can be mutated, mutating each parameter lists
# individually. The actual mutation will be handled by the parameter lists
# itself. (must implemented "mutate(sigma)" method).
class ParameterVector():
	def __init__(self, parameters_lists):
		self.parameters_lists = parameters_lists

	# mutate mutates every parameter list individually,
	# with a given σ (refers to mutation-width)
	def mutate(self, sigma):
		for parameter_list in self.parameters_lists:
			parameter_list.mutate(sigma)

	def __str__(self):
		s = "ParameterVector (\n"
		for parameter_list in self.parameters_lists:
			s += "\t" + str(parameter_list) + "\n"
		s += ")"
		return s

# Solution resembles a solution candidate as defined within a Evolution Strategy
class Solution():
	def __init__(self, parameter_vector, sigma, age, fitness_f):
		self.parameter_vector = parameter_vector
		self.sigma = sigma
		self.age = age
		self.fitness_f = fitness_f

	def mutate(self):
		self.parameter_vector.mutate(self.sigma)

	def evaluate(self):
		return self.fitness_f(self.parameter_vector)

# class EvolutionStrategy():

if __name__ == "__main__":
	main()