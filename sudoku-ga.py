from random import randint, random
from collections import Counter
from operator import itemgetter, attrgetter

SLOT = 3
N = SLOT*SLOT
POP_LEN = 100
CROMO_LEN = N*N
GENERATIONS_LEN = 1000
MUTAT_RATE = 0.1
CROSS_RATE = 0.9
CROMO_NAME = 'cromossome'
CROMO_FIT = 'fitness'
CROMO_FIT_INV = -1
CROMO_FIT_SOL = 0
BEST_K_PERCENTAGE = 0.1
WORST_K_PERCENTAGE = 0.2
TOURNAMENT_PERCENTAGE = 0.7
BEST_K_SELECTION = int(POP_LEN * BEST_K_PERCENTAGE)
WORST_K_SELECTION = int(POP_LEN * WORST_K_PERCENTAGE)
TOURNAMENT_SELECTION = int(POP_LEN * TOURNAMENT_PERCENTAGE)
TOURNAMENT_SIZE = 3

population = []

line_indexes = [[N1*N+N2 for N2 in range(0,N)] for N1 in range(0,N)]
column_indexes = [[N1+N*N2 for N2 in range(0,N)] for N1 in range(0,N)]
box_indexes = [[0,1,2,9,10,11,18,19,20], # TODO programmatically
               [3,4,5,12,13,14,21,22,23],
               [6,7,8,15,16,17,24,25,26],
               [27,28,29,36,37,38,45,46,47],
               [30,31,32,39,40,41,48,49,50],
               [33,34,35,42,43,44,51,52,53],
               [54,55,56,63,64,65,72,73,74],
               [57,58,59,66,67,68,75,76,77],
               [60,61,62,69,70,71,78,79,80]]

sample = { 'cromossome':[   1, 2, 3,  4, 5, 6,  7, 8, 9,
                           11,12,13, 14,15,16, 17,18,19,
                           21,22,23, 24,25,26, 27,28,29,

                           31,32,33, 34,35,36, 37,38,39,
                           41,42,43, 44,45,46, 47,48,49,
                           51,52,53, 54,55,56, 57,58,59,

                           61,62,63, 64,65,66, 67,68,69,
                           71,72,73, 74,75,76, 77,78,79,
                           81,82,83, 84,85,86, 87,88,89]}
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
    h=sum([count_similars([individual[CROMO_NAME][l] for l in x]) for x in column_indexes])
    return h

# count similar itens in same box
def count_box(individual):
    h=sum([count_similars([individual[CROMO_NAME][l] for l in x]) for x in box_indexes])
    return h

#-----------------------------
# Individuals functions
#-----------------------------
# calculate fitness from individual
def fitness_cromossome(individual):
    sum_lines = count_line(individual)
    sum_columns = count_column(individual)
    sum_boxes = count_box(individual)
    #sum_similar = count_similar(individual)
    return sum_lines + sum_columns + sum_boxes

def crossover_one_step(individual1, individual2):
    p = randint(0,CROMO_LEN-1)
    newindividual1 = individual1.copy()
    newindividual2 = individual2.copy()
    newindividual1[CROMO_FIT] = CROMO_FIT_INV
    newindividual2[CROMO_FIT] = CROMO_FIT_INV
    for i in range(p, CROMO_LEN-1):
        newindividual1[CROMO_NAME][i] = individual2[CROMO_NAME][i]
        newindividual2[CROMO_NAME][i] = individual1[CROMO_NAME][i]
    population.append(newindividual1)
    population.append(newindividual2)

def crossover_one_line(individual1, individual2):
    p = randint(0,len(line_indexes)-1)
    newindividual1 = individual1.copy()
    newindividual2 = individual2.copy()
    newindividual1[CROMO_FIT] = CROMO_FIT_INV
    newindividual2[CROMO_FIT] = CROMO_FIT_INV
    for i in line_indexes[p]:
        newindividual1[CROMO_NAME][i]=individual2[CROMO_NAME][i]
        newindividual2[CROMO_NAME][i]=individual1[CROMO_NAME][i]
    population.append(newindividual1)
    population.append(newindividual2)

def crossover_one_column(individual1, individual2):
    p = randint(0,len(column_indexes)-1)
    newindividual1 = individual1.copy()
    newindividual2 = individual2.copy()
    newindividual1[CROMO_FIT] = CROMO_FIT_INV
    newindividual2[CROMO_FIT] = CROMO_FIT_INV
    for i in column_indexes[p]:
        newindividual1[CROMO_NAME][i]=individual2[CROMO_NAME][i]
        newindividual2[CROMO_NAME][i]=individual1[CROMO_NAME][i]
    population.append(newindividual1)
    population.append(newindividual2)

def crossover_one_box(individual1, individual2):
    p = randint(0,len(box_indexes)-1)
    newindividual1 = individual1.copy()
    newindividual2 = individual2.copy()
    newindividual1[CROMO_FIT] = CROMO_FIT_INV
    newindividual2[CROMO_FIT] = CROMO_FIT_INV
    for i in box_indexes[p]:
        newindividual1[CROMO_NAME][i]=individual2[CROMO_NAME][i]
        newindividual2[CROMO_NAME][i]=individual1[CROMO_NAME][i]
    population.append(newindividual1)
    population.append(newindividual2)

# crossover two individuals
def crossover_cromossome(individual1, individual2):
    crossover_one_step(individual1, individual2)
    crossover_one_line(individual1, individual2)
    crossover_one_column(individual1, individual2)

def mutation_swap(individual):
    newindividual = individual.copy()
    p1 = randint(0,CROMO_LEN-1)
    p2 = randint(0,CROMO_LEN-1)
    v1 = newindividual[CROMO_NAME][p1]
    v2 = newindividual[CROMO_NAME][p2]
    newindividual[CROMO_NAME][p1] = v2
    newindividual[CROMO_NAME][p2] = v1
    newindividual[CROMO_FIT] = CROMO_FIT_INV
    population.append(newindividual)

def mutation_new_value(individual):
    newindividual = individual.copy()
    newindividual[CROMO_NAME][randint(0,CROMO_LEN-1)] = randint(1,N)
    newindividual[CROMO_FIT] = CROMO_FIT_INV
    population.append(newindividual)

# mutate individual
def mutation_cromossome(individual):
    mutation_new_value(individual)
    mutation_swap(individual)


#-----------------------------
# population functions
#-----------------------------
# calcule fitness of all individuals from population
def fitness_population(population):
    [p.update({CROMO_FIT:fitness_cromossome(p)}) for p in population if p[CROMO_FIT]==CROMO_FIT_INV]

# init random population
def init_population(population):
    [population.append({CROMO_NAME:[randint(1,9) for x in range(CROMO_LEN)], CROMO_FIT:CROMO_FIT_INV}) for i in range(POP_LEN)]

def selection_tournament(population):
    newpopulation = []
    for i in range(0, TOURNAMENT_SELECTION):
        individual = population[randint(0,POP_LEN-1)].copy()
        for r in range(1,TOURNAMENT_SIZE):
            individual1 = population[randint(0,POP_LEN-1)]
            if (individual[CROMO_FIT] > individual1[CROMO_FIT]):
                individual = individual.copy()
        newpopulation.append(individual)
    return newpopulation

# select population individuals
def selection_population(population):
    newpopulation = []
    spopulation = sorted(population, key=itemgetter(CROMO_FIT))
    newpopulation.extend(spopulation[:BEST_K_SELECTION])
    newpopulation.extend(spopulation[-WORST_K_SELECTION:])
    newpopulation.extend(selection_tournament(population))

    return newpopulation, [spopulation[0][CROMO_FIT], spopulation[-1][CROMO_FIT], sum(d[CROMO_FIT] for d in population) / len(population)]

# crossover individuals randomly
def crossover_population(population):
    for i in range(0,len(population)):
        r = random()
        if r <= CROSS_RATE: # do crossover
            crossover_cromossome(population[randint(0,POP_LEN-1)], population[randint(0,POP_LEN-1)])

# mutation individuals randomly
def mutation_population(population):
    for i in range(0,len(population)):
        r = random()
        if r <= MUTAT_RATE: # do mutation
            mutation_cromossome(population[randint(0,POP_LEN-1)])

def found_solution(population):
    return [p for p in population if p[CROMO_FIT]==CROMO_FIT_SOL]

def print_sudoku(individual):
    print("Found Solution")
    print("Sudoku")
    for i in range(N):
        print(individual[CROMO_NAME][i*N:i*N+N])

init_population(population)
fitness_population(population)
for i in range(GENERATIONS_LEN):
    crossover_population(population)
    mutation_population(population)
    fitness_population(population)
    population, statistics = selection_population(population)
    print('Iteration ' + str(i) + ', Best: ' + str(statistics[0]) + ', Worst: ' + str(statistics[1]) + ', Average: ' + str(statistics[2]))

    # check result
    s = found_solution(population)
    if len(s) > 0:
        print_sudoku(s[0])
        break


