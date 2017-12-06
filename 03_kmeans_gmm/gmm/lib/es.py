#!/usr/bin/env python

import copy
import json
import numpy as np

# ParameterVector stores a number of parameters lists
# for a specific problem. Mathematically it can be imagined
# as a vector of thetas ([θ1, θ2, θ3, ...]), where θ is parameter list
# consists of the actual parameter. For example
#
# Multivariate Normal Distribution
#	ParameterVector = [θ1], with θ1 = {μ, Σ}
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
	def __init__(self, parameter_lists):
		self.parameter_lists = parameter_lists

	def __iter__(self):
		return iter(self.parameter_lists)

	def __getitem__(self, key):
		return self.parameter_lists[key]

	# mutate mutates every parameter list individually,
	# with a given σ (refers to mutation-width). After mutation,
	# it let's the parameter lists also normalize in order to 
	# adjust e.g wrong values (ω's in GMMs should add up to 1)
	def mutate(self, sigma):
		for parameter_list in self.parameter_lists:
			parameter_list.mutate(sigma)
		
		# call parameter lists defined normalization function
		self.parameter_lists.normalize()

	def __str__(self):
		s = "ParameterVector (\n"
		for parameter_list in self.parameter_lists:
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
		self.fitness = 0

	def mutate(self):
		self.parameter_vector.mutate(self.sigma)

	def evaluate(self, X):
		self.fitness = self.fitness_f(X, self.parameter_vector)
		return self.fitness

	def __str__(self):
		s = "Solution (\n"
		s += "\tσ = " + str(self.sigma) + "\n"
		s += "\tκ = " + str(self.age) + "\n"
		s += "\tfitness = " + str(self.fitness) + "\n"
		s += "\t" + str(self.parameter_vector) + "\n"
		s += ")"
		return s

class EvolutionStrategy():
	TYPE_COMMA = ","
	TYPE_PLUS = "+"

	def __init__(self,
		X,
		mu,
		rho,
		lamd,
		es_type,
		sigma,
		tau,
		max_iter,
		min_sigma,
		parameter_vectors,
		fitness_f):

		############### [ES Base Parameter] ###############

		# [μ] number of parents
		self.mu = mu

		# [ρ] how many parents should be selected
		self.rho = rho
		if self.rho > self.mu or self.rho == 0:
			self.rho = self.mu

		# [λ] how many children should be "born"
		self.lamd = lamd
		if self.lamd < self.mu:
			self.lamd = self.mu

		# the type of ES (comma, plus)
		self.es_type = es_type
		if self.es_type not in [EvolutionStrategy.TYPE_COMMA, EvolutionStrategy.TYPE_PLUS]:
			self.es_type = EvolutionStrategy.TYPE_PLUS

		# [κ] the age of solution candidates
		self.k = np.inf if self.es_type == EvolutionStrategy.TYPE_PLUS else 1

		# the number of solution candidates (parents, initialized below)
		self.solutions = []

		# the best solution candidate
		self.best = None

		############### [Strategic Parameter] ###############

		# [σ] the mutation width (mutation step size)
		self.sigma = sigma

		# [τ] the adjustment parameter for sigma (1/5 success)
		self.tau = tau

		############### [Abortion Criteria] ###############

		# how many generations should be produced
		self.max_iter = max_iter

		# how far can sigma drop after every adjustment?
		# to low sigma leads to not significant mutations (stuck)
		self.min_sigma = min_sigma

		############### [User Input] ###############

		# samples
		self.X = X

		# fitness function
		self.fitness_f = fitness_f

		# initial list of parameter_vector (start parents)
		self.parameter_vectors = parameter_vectors

		# init solutions
		self._init_solutions()

		# all generation iterations
		self.generations_cnt = 0

	def _init_solutions(self):
		for parameter_vector in self.parameter_vectors:
			self.solutions.append(
				Solution(parameter_vector, self.sigma, self.k, self.fitness_f))

	def validate_parameter(self):
		validation_result = {}

		if self.mu <= 0:
			validation_result["mu"] = "number of parents must be positive"

		if self.rho <= 0:
			validation_result["rho"] = "number of selected parents must be positive"

		if self.lamd <= 0:
			validation_result["lamd"] = "number of children must be positive"

		if self.sigma <= self.min_sigma:
			validation_result["sigma"] = "mutation step-width must be higher then defined minimum"

		if self.tau < 0 or self.tau > 1:
			validation_result["tau"] = "mutation adjustment parameter must be within [0, 1]"

		if self.max_iter <= 0 and self.max_iter != -1:
			validation_result["max_iter"] = "generation max iteration must be positive"

		if self.X == list([]):
			validation_result["X"] = "no data given"

		if self.parameter_vectors == list([]):
			validation_result["parameter_vectors"] = "no initial parameter given"

		if self.fitness_f is None:
			validation_result["fitness_f"] = "no fitness function given"

		if validation_result != dict({}):
			return False, json.dumps(validation_result, indent=4)

		return True, validation_result

	def run(self):
		selection_pressure = self.lamd / self.mu
		self.generations_cnt = 0
		while self.generations_cnt != self.max_iter:
			next_best = max(self.solutions, key=lambda solution: solution.evaluate(self.X))

			# in order to compare the generation's best solution candidates
			# as an abortion criteria, at least one generation must be created
			if self.generations_cnt > 0:
				if next_best.evaluate(self.X) < self.best.evaluate(self.X):
					# print("[x] No improvement found in comparison to last generation")
					break

			# set best solution candidate
			self.best = next_best

			# select uniformly ρ parents for the next generation
			parents = []
			for _ in range(0, self.rho):
				parents.append(np.random.choice(self.solutions))

			# generate next generation
			children = []
			
			for _ in range(0, self.lamd):
				# select uniformly the child out of the parents
				# to prevent the same object (with same adress) from being chosen,
				# create deep copy
				child = copy.deepcopy(np.random.choice(parents))

				# mutation time
				child.mutate()
				children.append(child)

			# print([child.evaluate(self.X) for child in children])
			
			# for ES(ρ/μ,λ), pool is built only with children
			# for ES(ρ/μ+λ), pool also contains parents
			pool = copy.deepcopy(children)
			if self.es_type == EvolutionStrategy.TYPE_PLUS:
				pool += parents

			# sort children based on fitness function
			pool.sort(key=lambda child: child.evaluate(self.X), reverse=True)

			# select μ best for the next generation
			self.solutions = pool[:self.mu]

			# 1/5 success rate
			self._adjust_sigma(children)
		
			if self.sigma <= self.min_sigma:
				# print("[x] sigma below min_sigma threshold")
				break

			self.generations_cnt += 1

	def run_iter(self):
		selection_pressure = self.lamd / self.mu
		self.generations_cnt = 0
		while self.generations_cnt != self.max_iter:
			next_best = max(self.solutions, key=lambda solution: solution.evaluate(self.X))

			# in order to compare the generation's best solution candidates
			# as an abortion criteria, at least one generation must be created
			if self.generations_cnt > 0:
				if next_best.evaluate(self.X) < self.best.evaluate(self.X):
					# print("[x] No improvement found in comparison to last generation")
					break

			# set best solution candidate
			self.best = next_best

			# yield to user
			yield(self.generations_cnt)

			# select uniformly ρ parents for the next generation
			parents = []
			for _ in range(0, self.rho):
				parents.append(np.random.choice(self.solutions))

			# generate next generation
			children = []
			
			for _ in range(0, self.lamd):
				# select uniformly the child out of the parents
				# to prevent the same object (with same adress) from being chosen,
				# create deep copy
				child = copy.deepcopy(np.random.choice(parents))

				# mutation time
				child.mutate()
				children.append(child)

			# print([child.evaluate(self.X) for child in children])
			
			# for ES(ρ/μ,λ), pool is built only with children
			# for ES(ρ/μ+λ), pool also contains parents
			pool = copy.deepcopy(children)
			if self.es_type == EvolutionStrategy.TYPE_PLUS:
				pool += parents

			# sort children based on fitness function
			pool.sort(key=lambda child: child.evaluate(self.X), reverse=True)

			# select μ best for the next generation
			self.solutions = pool[:self.mu]

			# 1/5 success rate
			self._adjust_sigma(children)
		
			if self.sigma <= self.min_sigma:
				# print("[x] sigma below min_sigma threshold")
				break

			self.generations_cnt += 1

	# _adjust_sigma adapts σ based on the 1/5 success rule for mutants (children).
	# For this, the adjustment parameter τ is used.
	def _adjust_sigma(self, mutants):
		fifth = round(self.lamd / 5)

		# count number of successful mutants
		s = 0
		for mutant in mutants:
			if mutant.evaluate(self.X) > self.best.evaluate(self.X):
				s += 1

		# print(f"{fifth} out of {self.lamd} should be successful; {s} are")

		if s < fifth:
			self.sigma *= self.tau
		if s > fifth:
			self.sigma /= self.tau

		# update on every solution candidate
		for i in range(0, self.mu):
			self.solutions[i].sigma = self.sigma

