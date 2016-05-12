#! /usr/bin/env python


#    Experimental prototype for solving Sudokus with Genetic Algorithms.
#
#    Copyright (C) 2013 Efstathios Chatzikyriakidis <contact@efxa.org>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


try:
	import random, math, operator, time
except (ImportError) as error:
	import sys, os
	sys.exit('{0}: {1}.'.format(os.path.basename(__file__), error))


class SizeExtension(object):
	def __init__ (self, size):
		self.size = size

	def __len__(self):
		return self.size


class Allele(object):
	@staticmethod
	def symbols (size):
		return range(1, size + 1)


class Gene(SizeExtension):
	mutationMethods = dict(Reset = 'resetMutation',
                               Swap  = 'swapMutation')

	def __init__ (self, candidate):
		super(Gene, self).__init__(len(candidate))

		self.candidate = candidate

		self.symbols = Allele.symbols(self.size)

		self.missing = list(set(self.symbols) - set(self.candidate))

		self.alleles = []

	def randomize (self):
		missing = list(self.missing)

		random.shuffle(missing)

		self.alleles = list(self.candidate)

		indexes = [i for i, x in enumerate(self.alleles) if x == 0]

		for index in indexes:
			self.alleles[index] = missing.pop()

	def resetMutation (self):
		self.randomize()

	def swapMutation (self):
		minimumNumberOfPoints = 2

		allPoints = [i for i, x in enumerate(self.candidate) if x == 0]

		numberOfAllPoints = len(allPoints)

		if (numberOfAllPoints < minimumNumberOfPoints): return

		randomNumberOfPoints = random.randrange(minimumNumberOfPoints, numberOfAllPoints + 1)

		points = random.sample(allPoints, randomNumberOfPoints)

		points.sort (reverse = (True if (random.random() < 0.5) else False))

		for i in points:
			xorValue = 0
			for j in points:
				xorValue ^= self.alleles[j]

			self.alleles[i] = xorValue

		xorValue = 0
		for point in points:
			xorValue ^= self.alleles[point]

		self.alleles[points[0]] = xorValue

	def mutate (self):
		key = random.choice(Gene.mutationMethods.keys())

		getattr (self, Gene.mutationMethods[key])()


class Genotype(SizeExtension):
	crossoverMethods = dict(OnePoint      = 'onePointCrossover',
                                UniformAllele = 'uniformAlleleCrossover',
                                UniformGene   = 'uniformGeneCrossover')

	def __init__ (self, candidate):
		super(Genotype, self).__init__(len(candidate))

		self.candidate = candidate

		self.symbols = Allele.symbols(self.size)

		self.genes = []

	def randomize (self):
		self.genes = []

		for gene in range(self.size):
			g = Gene(self.candidate[gene])

			g.randomize()

			self.genes.append(g)

	def set (self, genotype):
		self.genes = genotype.genes

	def uniformAlleleCrossover(self, other):
		childs = []

		childs.append(Genotype(self.candidate))
		childs.append(Genotype(self.candidate))

		for gene in range(self.size):
			childs[0].genes.append(Gene(self.candidate[gene]))
			childs[1].genes.append(Gene(self.candidate[gene]))

			for allele in range(len(self.symbols)):
				if (random.random() < 0.5):
					childs[0].genes[gene].alleles.append(self.genes[gene].alleles[allele])
					childs[1].genes[gene].alleles.append(other.genes[gene].alleles[allele])
				else:
					childs[0].genes[gene].alleles.append(other.genes[gene].alleles[allele])
					childs[1].genes[gene].alleles.append(self.genes[gene].alleles[allele])

		return childs

	def uniformGeneCrossover(self, other):
		childs = []

		childs.append(Genotype(self.candidate))
		childs.append(Genotype(self.candidate))

		for gene in range(self.size):
			if (random.random() < 0.5):
				childs[0].genes.append(self.genes[gene])
				childs[1].genes.append(other.genes[gene])
			else:
				childs[0].genes.append(other.genes[gene])
				childs[1].genes.append(self.genes[gene])

		return childs

	def onePointCrossover(self, other):
		point = random.randrange(1, self.size)

		childs = []

		childs.append(Genotype(self.candidate))
		childs.append(Genotype(self.candidate))

		childs[0].genes = self.genes[:point] + other.genes[point:]
		childs[1].genes = other.genes[:point] + self.genes[point:]

		return childs

	def mutate (self):
		gene = random.randrange(0, self.size)

		self.genes[gene].mutate()

	def crossover (self, other):
		key = random.choice(Genotype.crossoverMethods.keys())

		return getattr (self, Genotype.crossoverMethods[key])(other)


class Phenotype(SizeExtension):
	ranges = {
		'4': [ range(0, 2), range(2, 4) ],
	 	'9': [ range(0, 3), range(3, 6), range(6, 9) ]
	}

	slices = {
	 	'4': [ slice(0, 2), slice(2, 4) ],
	 	'9': [ slice(0, 3), slice(3, 6), slice(6, 9) ]
	}

	def __init__ (self, genotype):
		super(Phenotype, self).__init__(len(genotype))

		self.genotype = genotype

		self.table = []

	def getRanges(self):
		try:
			return Phenotype.ranges[str(self.size)]
		except KeyError:
			pass

	def getSlices(self):
		try:
			return Phenotype.slices[str(self.size)]
		except KeyError:
			pass

	def __str__(self):
		return str(self.table)

	def convert (self):
		self.table = []

		ranges = self.getRanges()
		slices = self.getSlices()

		for r, rng in enumerate(ranges):
			for s, slc in enumerate(slices):
				row = []

				for i in rng:
					row.extend(self.genotype.genes[i].alleles[slc])

				self.table.append(row)


class Individual(object):
	def __init__ (self, candidate):
		self.genotype = Genotype(candidate)

	def randomize (self):
		self.genotype.randomize()

	def phenotype(self):
		phenotype = Phenotype(self.genotype)
		phenotype.convert()
		return phenotype

	def crossover (self, other):
		offsprings = self.genotype.crossover(other.genotype)

		childs = []

		for offspring in offsprings:
			child = Individual(self.genotype.candidate)
			child.genotype.set(offspring)
			childs.append(child)

		return childs

	def mutate (self):
		self.genotype.mutate()

	def estimate (self):
		phenotype = self.phenotype()

		size = len(phenotype)

		symbols = self.genotype.symbols

		rowsSumDistance = [0] * size
		colsSumDistance = [0] * size

		rowsPrdDistance = [1] * size
		colsPrdDistance = [1] * size

		rowsMissed = [0] * size
		colsMissed = [0] * size

		symbolsSum = sum(symbols)
		symbolsFac = math.factorial(len(symbols))
		symbolsSet = set(symbols)

		for i in range(size):
			for j in range(size):
				rowsSumDistance[i] += phenotype.table[i][j]
				rowsPrdDistance[i] *= phenotype.table[i][j]

				colsSumDistance[i] += phenotype.table[j][i]
				colsPrdDistance[i] *= phenotype.table[j][i]

			rowsSumDistance[i] = abs(symbolsSum - rowsSumDistance[i])
			rowsPrdDistance[i] = abs(symbolsFac - rowsPrdDistance[i])

			colsSumDistance[i] = abs(symbolsSum - colsSumDistance[i])
			colsPrdDistance[i] = abs(symbolsFac - colsPrdDistance[i])

			rowsMissed[i] = len(list(symbolsSet - set(phenotype.table[i])))
			colsMissed[i] = len(list(symbolsSet - set([row[i] for row in phenotype.table])))

		rootsColsPrd = [math.sqrt(i) for i in colsPrdDistance]
		rootsRowsPrd = [math.sqrt(i) for i in rowsPrdDistance]

		self.fitness = (10 * (sum(rowsSumDistance) + sum(colsSumDistance)) + 50 * (sum(rowsMissed) + sum(colsMissed)) + sum(rootsColsPrd) + sum(rootsRowsPrd))


class Population(SizeExtension):
	selectionMethods = dict(Tournament = 'tournamentSelection',
                                Rank       = 'rankSelection')

	def __init__ (self, populationSize, tournamentSize, candidateGenotype):
		super(Population, self).__init__(populationSize)

		self.candidateGenotype = candidateGenotype

		self.tournamentSize = tournamentSize

		self.strongest = None

		self.parents = []
		self.childs = []

	def initialize(self):
		self.parents = []

		for i in range(self.size):
			individual = Individual(self.candidateGenotype)
			individual.randomize()
			self.parents.append(individual)

	def estimate(self):
		[parent.estimate() for parent in self.parents]

	def resolved(self):
		self.strongest = min(self.parents, key = lambda o: o.fitness)

		return (self.strongest.fitness == 0)

	def generation(self):
		self.childs = []

	def elitism(self):
		self.childs.append(self.strongest)

	def fitnessOfStrongest(self):
		return self.strongest.fitness

	def reproduced(self):
		return (len(self.childs) >= len(self.parents))

	def select(self):
		key = random.choice(Population.selectionMethods.keys())

		return getattr (self, Population.selectionMethods[key])()

	def addChilds(self, childs):
		for child in childs:
			if self.reproduced():
				break
			else:
				self.childs.append(child)

	def replacement(self):
		self.parents = self.childs

	def rankSelection(self):
		parents = sorted (self.parents, key = lambda o: o.fitness)

		ranks = range(1, self.size + 1)

		sumOfRanks = sum(ranks)

		for i, parent in enumerate(parents):
			parent.probability = (((self.size - ranks[i] + 1) / float(sumOfRanks)) * 100)

		pick = random.uniform(0, sum([p.probability for p in parents]))

		current = 0

		for parent in parents:
			current += parent.probability

			if current > pick:
				return parent

	def tournamentSelection(self):
		indexes = [random.randrange(0, self.size) for i in range(self.tournamentSize)]

		return min([self.parents[i] for i in indexes], key = lambda o: o.fitness)


class SudokuGA(object):
	def __init__ (self, parameters):
		self.mutationProbability = parameters['mutationProbability']

		self.populationSize = parameters['populationSize']

		self.tournamentSize = parameters['tournamentSize']

		self.candidateGenotype = parameters['candidateGenotype']

		self.population = Population (self.populationSize, self.tournamentSize, self.candidateGenotype)

		random.seed (time.time ())

	def evolution(self):
		self.population.initialize()
		self.population.estimate()

		while (not self.population.resolved()):
			self.population.generation()
			self.population.elitism()

			print self.population.fitnessOfStrongest()

			while (not self.population.reproduced()):
				mother = self.population.select()
				father = self.population.select()

				childs = mother.crossover(father)

				for child in childs:
					if (random.random() < self.mutationProbability):
						child.mutate()

					child.estimate()

				self.population.addChilds(childs)

			self.population.replacement()

		return self.population.strongest.phenotype()
