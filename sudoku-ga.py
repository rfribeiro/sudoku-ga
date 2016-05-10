import copy
from random import randint, random, sample
from collections import Counter
from operator import itemgetter, attrgetter

SLOT = 3
N = SLOT*SLOT
POP_LEN = 100
CROMO_LEN = N*N
GENERATIONS_LEN = 10000
MUTAT_RATE = 0.2
CROSS_RATE = 0.90
CROMO_NAME = 'cromossome'
CROMO_FIT = 'fitness'
CROMO_FIT_INV = -1
CROMO_FIT_SOL = 0
BEST_K_PERCENTAGE = 0.01
WORST_K_PERCENTAGE = 0.0
TOURNAMENT_PERCENTAGE = 0.99
BEST_K_SELECTION = int(POP_LEN * BEST_K_PERCENTAGE)
WORST_K_SELECTION = int(POP_LEN * WORST_K_PERCENTAGE)
TOURNAMENT_SELECTION = int(POP_LEN * TOURNAMENT_PERCENTAGE)
TOURNAMENT_SIZE = 5

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

#box_indexes = [[0,1,4,5], # TODO programmatically
#               [2,3,6,7],
#               [8,9,12,13],
#               [10,11,14,15]]

sample_test = { CROMO_NAME:[   1, 2, 3,  4, 5, 6,  7, 8, 9,
                           11,12,13, 14,15,16, 17,18,19,
                           21,22,23, 24,25,26, 27,28,29,

                           31,32,33, 34,35,36, 37,38,39,
                           41,42,43, 44,45,46, 47,48,49,
                           51,52,53, 54,55,56, 57,58,59,

                           61,62,63, 64,65,66, 67,68,69,
                           71,72,73, 74,75,76, 77,78,79,
                           81,82,83, 84,85,86, 87,88,89]}
solved = { CROMO_NAME : [ 8,7,1,4,5,3,9,2,6,
                            9,5,3,7,2,6,4,8,1,
                            2,4,6,8,9,1,5,3,7,
                            7,6,5,3,8,9,1,4,2,
                            1,3,2,5,4,7,8,6,9,
                            4,9,8,1,6,2,7,5,3,
                            6,8,7,9,3,5,2,1,4,
                            5,2,9,6,1,4,3,7,8,
                            3,1,4,2,7,8,6,9,5],
           CROMO_FIT:-1}

list_indexes = range(0, N)
list_values = range(1,N+1)

CROMO_FIT_SOL = (3*N*N*(N-1))/2

#-----------------------------
# Utility functions
#-----------------------------
# count similar itens in a list
def count_values(item):
    g = [1 for i in list_indexes for j in list_indexes if item[i]!=item[j]]
    h = sum(g)
    return h

# count similar itens in a list
def count_differents(item):
    g = [i for (dup, i) in Counter(item).items() if i == 1]
    h = sum(g)
    return h

# count similar itens in a list
def count_similars(item):
    g = [i for (dup, i) in Counter(item).items() if i > 1]
    h = sum(g)
    return h

# count similar itens in same line
def count_line_similars(individual):
    h=sum([count_similars([individual[CROMO_NAME][l] for l in x]) for x in line_indexes])
    return h

# count similar itens in same column
def count_column_similars(individual):
    h=sum([count_similars([individual[CROMO_NAME][l] for l in x]) for x in column_indexes])
    return h

# count similar itens in same box
def count_box_similars(individual):
    h=sum([count_similars([individual[CROMO_NAME][l] for l in x]) for x in box_indexes])
    return h

def count_line_values(individual):
    h=sum([count_values([individual[CROMO_NAME][l] for l in x]) for x in line_indexes])
    return h

# count similar itens in same column
def count_column_values(individual):
    h=sum([count_values([individual[CROMO_NAME][l] for l in x]) for x in column_indexes])
    return h

# count similar itens in same box
def count_box_values(individual):
    h=sum([count_values([individual[CROMO_NAME][l] for l in x]) for x in box_indexes])
    return h

#-----------------------------
# Individuals functions
#-----------------------------
# calculate fitness from individual
def fitness_cromossome_sum(individual):
    sum_lines = count_line_similars(individual)
    sum_columns = count_column_similars(individual)
    sum_boxes = count_box_similars(individual)
    #sum_similar = count_similar(individual)
    return sum_lines + sum_columns + sum_boxes

def fitness_cromossome_sum_of_sum(individual):
    sum_lines = count_line_values(individual)
    sum_columns = count_column_values(individual)
    sum_boxes = count_box_values(individual)
    #sum_similar = count_similar(individual)
    return (sum_lines + sum_columns + sum_boxes)/2

def fitness_cromossome(individual):
    return fitness_cromossome_sum_of_sum(individual)

def crossover_one_point(individual1, individual2):
    p = randint(0,CROMO_LEN-1)
    newindividual1 = copy.deepcopy(individual1)
    newindividual2 = copy.deepcopy(individual2)
    newindividual1[CROMO_FIT] = CROMO_FIT_INV
    newindividual2[CROMO_FIT] = CROMO_FIT_INV
    for i in range(p, CROMO_LEN-1):
        newindividual1[CROMO_NAME][i] = individual2[CROMO_NAME][i]
        newindividual2[CROMO_NAME][i] = individual1[CROMO_NAME][i]
    population.append(newindividual1)
    population.append(newindividual2)

def crossover_one_line(individual1, individual2):
    p = randint(0,len(line_indexes)-1)
    newindividual1 = copy.deepcopy(individual1)
    newindividual2 = copy.deepcopy(individual2)
    newindividual1[CROMO_FIT] = CROMO_FIT_INV
    newindividual2[CROMO_FIT] = CROMO_FIT_INV
    for i in line_indexes[p]:
        newindividual1[CROMO_NAME][i]=individual2[CROMO_NAME][i]
        newindividual2[CROMO_NAME][i]=individual1[CROMO_NAME][i]
    population.append(newindividual1)
    population.append(newindividual2)

def crossover_one_column(individual1, individual2):
    p = randint(0,len(column_indexes)-1)
    newindividual1 = copy.deepcopy(individual1)
    newindividual2 = copy.deepcopy(individual2)
    newindividual1[CROMO_FIT] = CROMO_FIT_INV
    newindividual2[CROMO_FIT] = CROMO_FIT_INV
    for i in column_indexes[p]:
        newindividual1[CROMO_NAME][i]=individual2[CROMO_NAME][i]
        newindividual2[CROMO_NAME][i]=individual1[CROMO_NAME][i]
    population.append(newindividual1)
    population.append(newindividual2)

def crossover_one_box(individual1, individual2):
    p = randint(0,len(box_indexes)-1)
    newindividual1 = copy.deepcopy(individual1)
    newindividual2 = copy.deepcopy(individual2)
    newindividual1[CROMO_FIT] = CROMO_FIT_INV
    newindividual2[CROMO_FIT] = CROMO_FIT_INV
    for i in box_indexes[p]:
        newindividual1[CROMO_NAME][i]=individual2[CROMO_NAME][i]
        newindividual2[CROMO_NAME][i]=individual1[CROMO_NAME][i]
    population.append(newindividual1)
    population.append(newindividual2)

# crossover two individuals
def crossover_cromossome(individual1, individual2):
    #crossover_one_point(individual1, individual2)
    crossover_one_line(individual1, individual2)
    crossover_one_column(individual1, individual2)

def mutation_change(individual):
    newindividual = copy.deepcopy(individual)
    p1 = randint(0,CROMO_LEN-1)
    p2 = randint(0,CROMO_LEN-1)
    v1 = newindividual[CROMO_NAME][p1]
    v2 = newindividual[CROMO_NAME][p2]
    newindividual[CROMO_NAME][p1] = v2
    newindividual[CROMO_NAME][p2] = v1
    newindividual[CROMO_FIT] = CROMO_FIT_INV
    population.append(newindividual)

def mutation_swap_2(individual):
    newindividual = copy.deepcopy(individual)
    i = randint(0, len(line_indexes)-1)
    idx1 = randint(0,N-1)
    idx2 = randint(0,N-1)
    if idx1 != idx2:
        newindividual[CROMO_NAME][line_indexes[i][idx1]] = individual[CROMO_NAME][line_indexes[i][idx2]]
        newindividual[CROMO_NAME][line_indexes[i][idx2]] = individual[CROMO_NAME][line_indexes[i][idx1]]
        newindividual[CROMO_FIT] = CROMO_FIT_INV
        population.append(newindividual)

def mutation_swap_3(individual):
    newindividual = copy.deepcopy(individual)
    i = randint(0, len(line_indexes)-1)
    idx1 = randint(0,N-1)
    idx2 = randint(0,N-1)
    idx3 = randint(0,N-1)
    if idx1 != idx2:
        newindividual[CROMO_NAME][line_indexes[i][idx1]] = individual[CROMO_NAME][line_indexes[i][idx3]]
        newindividual[CROMO_NAME][line_indexes[i][idx2]] = individual[CROMO_NAME][line_indexes[i][idx1]]
        newindividual[CROMO_NAME][line_indexes[i][idx3]] = individual[CROMO_NAME][line_indexes[i][idx2]]
        newindividual[CROMO_FIT] = CROMO_FIT_INV
        population.append(newindividual)

def mutation_swap_inline(individual):
    newindividual = copy.deepcopy(individual)
    l = randint(0, len(line_indexes)-1)
    idx1 = randint(1, SLOT-1)
    idx2 = randint(0, SLOT-1)
    if (idx1 != idx2):
        for i in xrange(SLOT):
            newindividual[CROMO_NAME][line_indexes[l][i*idx1]] = individual[CROMO_NAME][line_indexes[l][i*idx2]]
            newindividual[CROMO_NAME][line_indexes[l][i]] = individual[CROMO_NAME][line_indexes[l][i]]
        newindividual[CROMO_FIT] = CROMO_FIT_INV
        population.append(newindividual)

def mutation_swap_line(individual):
    newindividual = copy.deepcopy(individual)
    idx1 = randint(0, len(line_indexes)-1)
    idx2 = randint(0, len(line_indexes)-1)
    if (idx1 != idx2):
        for i in xrange(len(line_indexes)):
            newindividual[CROMO_NAME][line_indexes[idx1][i]] = individual[CROMO_NAME][line_indexes[idx2][i]]
            newindividual[CROMO_NAME][line_indexes[idx2][i]] = individual[CROMO_NAME][line_indexes[idx1][i]]
        newindividual[CROMO_FIT] = CROMO_FIT_INV
        population.append(newindividual)

def mutation_swap_column(individual):
    newindividual = copy.deepcopy(individual)
    idx1 = randint(0, len(column_indexes)-1)
    idx2 = randint(0, len(column_indexes)-1)
    if (idx1 != idx2):
        for i in xrange(len(column_indexes)):
            newindividual[CROMO_NAME][column_indexes[idx1][i]] = individual[CROMO_NAME][column_indexes[idx2][i]]
            newindividual[CROMO_NAME][column_indexes[idx2][i]] = individual[CROMO_NAME][column_indexes[idx1][i]]
        newindividual[CROMO_FIT] = CROMO_FIT_INV
        population.append(newindividual)

def mutation_new_value(individual):
    newindividual = copy.deepcopy(individual)
    newindividual[CROMO_NAME][randint(0,CROMO_LEN-1)] = randint(1,N)
    newindividual[CROMO_FIT] = CROMO_FIT_INV
    population.append(newindividual)

def mutation_multi_values(individual):
    newindividual = copy.deepcopy(individual)
    for i in range(0,randint(0,CROMO_LEN-1)):
        newindividual[CROMO_NAME][randint(0,CROMO_LEN-1)] = randint(1,N)
    newindividual[CROMO_FIT] = CROMO_FIT_INV
    population.append(newindividual)

# mutate individual
def mutation_cromossome(individual):
    #mutation_new_value(individual)
    #mutation_change(individual)
    #mutation_multi_values(individual)
    mutation_swap_2(individual)
    mutation_swap_3(individual)
    mutation_swap_line(individual)
    mutation_swap_column(individual)


#-----------------------------
# population functions
#-----------------------------
# calcule fitness of all individuals from population
def fitness_population(population):
    [p.update({CROMO_FIT:fitness_cromossome(p)}) for p in population if p[CROMO_FIT]==CROMO_FIT_INV]

# init random population
def init_population_with_constraint(population):
    values = range(1,N+1)
    for z in range(POP_LEN):
        i = []
        [i.extend(sample(values, len(values))) for x in range(N)]
        population.append({CROMO_NAME:i, CROMO_FIT:CROMO_FIT_INV})


# init random population
def init_population_random(population):
    [population.append({CROMO_NAME:[randint(1,N) for x in range(CROMO_LEN)], CROMO_FIT:CROMO_FIT_INV}) for i in range(POP_LEN)]

def init_population(population):
    #init_population_random(population)
    init_population_with_constraint(population)

def selection_tournament(population):
    newpopulation = []
    for i in range(0, TOURNAMENT_SELECTION):
        individual = population[randint(0,POP_LEN-1)].copy()
        for r in range(1,TOURNAMENT_SIZE):
            individual1 = population[randint(0,POP_LEN-1)]
            if (individual[CROMO_FIT] < individual1[CROMO_FIT]):
                individual = individual1.copy()
        newpopulation.append(individual)
    return newpopulation

# select population individuals
def selection_population(population):
    newpopulation = []
    spopulation = sorted(population, key=itemgetter(CROMO_FIT), reverse=True)
    #print(str(spopulation[0][CROMO_NAME]))
    newpopulation.extend(spopulation[:BEST_K_SELECTION])
    #newpopulation.extend(spopulation[-WORST_K_SELECTION:])
    newpopulation.extend(selection_tournament(population))
    return newpopulation, [spopulation[0][CROMO_FIT], spopulation[-1][CROMO_FIT], sum(d[CROMO_FIT] for d in population) / len(population)]

# crossover individuals randomly
def crossover_population(population):
    for i in range(0,len(population)):
        r = random()
        if r <= CROSS_RATE: # do crossover
            r1 = randint(0,POP_LEN-1)
            r2 = randint(0,POP_LEN-1)
            #print(r1,r2, len(population))
            crossover_cromossome(population[r1], population[r2])

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

bestValue = 0
restartCount = 0
init_population(population)
fitness_population(population)
for i in range(GENERATIONS_LEN):
    #print('Population 1 : ' + str(len(population)))
    crossover_population(population)
    #print('Population 2 : ' + str(len(population)))
    mutation_population(population)
    #print('Population 3 : ' + str(len(population)))
    fitness_population(population)
    population, statistics = selection_population(population)
    print('Iteration ' + str(i) + ', Best: ' + str(statistics[0]) + ', Worst: ' + str(statistics[1]) + ', Average: ' + str(statistics[2]))

    # check result
    s = found_solution(population)
    if len(s) > 0:
        print_sudoku(s[0])
        break
    else:
        if bestValue == population[0][CROMO_FIT]:
            restartCount = restartCount + 1
        else:
            restartCount = 0
            bestValue = population[0][CROMO_FIT]

    if restartCount == 100:
        restartCount = 0
        population = []
        init_population(population)




