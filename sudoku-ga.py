from random import randint
from collections import Counter

N = 9
POP_LEN = 100
CROMO_LEN = N*N
GENERATIONS_LEN = 1000
MUTAT_RATE = 0.05
CROSS_RATE = 0.7
CROMO_NAME = 'cromossome'
CROMO_FIT = 'fitness'
CROMO_FIT_INV = -1
population = []

#-----------------------------
# Utility functions
#-----------------------------
# count similar itens in a list
def count_similars(item):
    h = len([dup for (dup, i) in Counter(item).items() if i > 1])
    return h

# count similar itens in same line
def count_line(individual):
    h = sum([count_similars(individual[CROMO_NAME][i*N:i*N+N]) for i in range(0,N)])
    return h

# count similar itens in same column
def count_column(individual):
    h = sum([count_similars(individual[CROMO_NAME][i*N:i*N+N]) for i in range(0,N)])
    return h

# count similar itens in same box
def count_box(individual):
    h = sum([count_similars(individual[CROMO_NAME][i*N:i*N+N]) for i in range(0,N)])
    return h

#-----------------------------
# Individuals functions
#-----------------------------
# calculate fitness from individual
def fitness_cromossome(individual):
    sum_lines = count_line(individual)
    sum_columns = count_column(individual)
    sum_boxes = count_box(individual)
    return sum_lines + sum_columns + sum_boxes

# crossover two individuals
def crossover_cromossome():
    return 0

# mutate individual
def mutation_cromossome():
    return 0

#-----------------------------
# population functions
#-----------------------------
# calcule fitness of all individuals from population
def fitness_population(population):
    [p.update({CROMO_FIT:fitness_cromossome(p)}) for p in population if p[CROMO_FIT]==CROMO_FIT_INV]

# init random population
def init_population(population):
    [population.append({CROMO_NAME:[randint(1,9) for x in range(CROMO_LEN)], CROMO_FIT:CROMO_FIT_INV}) for i in range(POP_LEN)]

# select population individuals
def selection_population(population):
    return 0

# crossover individuals randomly
def crossover_population(population):
    return 0

# mutation individuals randomly
def mutation_population(population):
    return 0

init_population(population)
fitness_population(population)
for i in range(GENERATIONS_LEN):
    selection_population(population)
    crossover_population()
    mutation_population()
    fitness_population(population)
    # check result


