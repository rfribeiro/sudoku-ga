import copy, math
from random import randint, random, sample, shuffle, randrange, choice
from collections import Counter
from operator import itemgetter, attrgetter

CROMO_NAME = 'cromossome'
CROMO_FIT = 'fitness'

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

POPULATION = 'population'
GENERATIONS = 'generation'
MUTATION_PROB = 'mutation_prop'
CROSSOVER_PROB = 'crossover_prop'
TOURNAMENT = 'tournament'
FITNESS_F = 'fitness'
SUM_OF_SUM = 'sum_of_sum'
CALCULATED = 'calculated'
REMOVE_DUP = 'remove_dup'
INIT_POP = 'init_pop'
CONSTRAINT = 'constraint'
RANDOM = 'random'

SLOT = 3

if SLOT == 3:
    parameters = [{POPULATION:4000, GENERATIONS:1000, MUTATION_PROB:0.01, CROSSOVER_PROB:0.95, TOURNAMENT:5, FITNESS_F:CALCULATED, REMOVE_DUP:False, INIT_POP:CONSTRAINT},
              {POPULATION:4000, GENERATIONS:1000, MUTATION_PROB:0.01, CROSSOVER_PROB:0.95, TOURNAMENT:5, FITNESS_F:CALCULATED, REMOVE_DUP:True, INIT_POP:CONSTRAINT},
              {POPULATION:4000, GENERATIONS:1000, MUTATION_PROB:0.3, CROSSOVER_PROB:0.3, TOURNAMENT:5, FITNESS_F:CALCULATED, REMOVE_DUP:False, INIT_POP:CONSTRAINT},
              {POPULATION:4000, GENERATIONS:1000, MUTATION_PROB:0.3, CROSSOVER_PROB:0.3, TOURNAMENT:5, FITNESS_F:CALCULATED, REMOVE_DUP:True, INIT_POP:CONSTRAINT},]

else:
    parameters = [{POPULATION:250, GENERATIONS:500, MUTATION_PROB:0.01, CROSSOVER_PROB:0.95, TOURNAMENT:3, FITNESS_F:CALCULATED, REMOVE_DUP:False, INIT_POP:RANDOM},
              {POPULATION:50, GENERATIONS:500, MUTATION_PROB:0.01, CROSSOVER_PROB:0.95, TOURNAMENT:3, FITNESS_F:CALCULATED, REMOVE_DUP:True, INIT_POP:RANDOM},
              {POPULATION:250, GENERATIONS:500, MUTATION_PROB:0.3, CROSSOVER_PROB:0.5, TOURNAMENT:3, FITNESS_F:CALCULATED, REMOVE_DUP:False, INIT_POP:RANDOM},
              {POPULATION:50, GENERATIONS:500, MUTATION_PROB:0.3, CROSSOVER_PROB:0.5, TOURNAMENT:3, FITNESS_F:CALCULATED, REMOVE_DUP:True, INIT_POP:RANDOM},]


N = SLOT*SLOT
POP_LEN = 10
CROMO_LEN = N*N
GENERATIONS_LEN = 1000
MUTAT_RATE = 0.3
CROSS_RATE = 0.6
CROMO_FIT_INV = -1
CROMO_FIT_SOL = 0
BEST_K_PERCENTAGE = 0.01
WORST_K_PERCENTAGE = 0.00
TOURNAMENT_PERCENTAGE = 1
TOURNAMENT_SIZE = 5
RESTART_VALUE = 100

BEST_K_SELECTION = int(POP_LEN * BEST_K_PERCENTAGE)
WORST_K_SELECTION = int(POP_LEN * WORST_K_PERCENTAGE)
TOURNAMENT_SELECTION = int(POP_LEN * TOURNAMENT_PERCENTAGE)

population = []

line_indexes = [[N1*N+N2 for N2 in range(0,N)] for N1 in range(0,N)]
column_indexes = [[N1+N*N2 for N2 in range(0,N)] for N1 in range(0,N)]

fitness_calls = 0

if SLOT == 2:
    box_indexes = [[0,1,4,5], # TODO programmatically
                   [2,3,6,7],
                   [8,9,12,13],
                  [10,11,14,15]]
else:
    box_indexes = [[0,1,2,9,10,11,18,19,20], # TODO programmatically
                   [3,4,5,12,13,14,21,22,23],
                   [6,7,8,15,16,17,24,25,26],
                   [27,28,29,36,37,38,45,46,47],
                   [30,31,32,39,40,41,48,49,50],
                   [33,34,35,42,43,44,51,52,53],
                   [54,55,56,63,64,65,72,73,74],
                   [57,58,59,66,67,68,75,76,77],
                   [60,61,62,69,70,71,78,79,80]]

list_indexes = range(0, N)
list_values = range(1,N+1)

CROMO_FIT_SOL_DEFINED = (3*N*N*(N-1))/2

if SLOT == 2:
    given_cromossome = [1,0,0,0,
                        2,0,0,3,
                        0,0,4,1,
                        4,0,0,0]
    given_cromossome = [0,0,3,0,
                        0,4,0,0,
                        0,0,0,2,
                       4,0,1,0]
    given_cromossome = [0,0,0,0,
                        0,0,0,0,
                        0,0,0,0,
                       0,0,0,0]
else:
    given_cromossome = [  2, 0, 0, 8, 0, 4, 0, 7, 1,
                          5, 8, 9, 0, 0, 1, 0, 0, 3,
                          0, 1, 7, 3, 6, 0, 0, 2, 0,
                          0, 6, 0, 0, 2, 3, 1, 9, 0,
                          0, 2, 5, 9, 0, 0, 7, 0, 8,
                          7, 0, 3, 0, 8, 1, 6, 0, 0,
                          6, 0, 0, 5, 4, 0, 0, 1, 2,
                          3, 9, 0, 0, 1, 0, 4, 5, 0,
                          0, 5, 4, 0, 3, 6, 0, 0, 8]

    #given_cromossome = [  2, 3, 6, 0, 5, 0, 9, 7, 0,
    #                      5, 0, 9, 2, 0, 1, 0, 0, 3,
    #                      0, 1, 0, 0, 6, 9, 8, 2, 0,
    #                      4, 0, 8, 7, 0, 3, 0, 0, 5,
    #                      0, 2, 0, 9, 6, 0, 7, 0, 8,
    #                      7, 0, 3, 0, 8, 0, 0, 4, 2,
    #                      0, 0, 7, 5, 4, 9, 3, 1, 2,
    #                      3, 9, 0, 1, 0, 7, 0, 0, 6,
    #                      1, 5, 4, 0, 3, 6, 0, 0, 8]

    given_cromossome = [  0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0]

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

def get_cromossome_box(individual):
    return [individual[CROMO_NAME][l] for x in box_indexes for l in x]

#-----------------------------
# Individuals functions
#-----------------------------
def fitness_calculated(individual):
    phenotype = get_cromossome_box(individual)

    size = len(phenotype)

    symbols = list_values

    rowsSumDistance = [0] * N
    colsSumDistance = [0] * N

    rowsPrdDistance = [1] * N
    colsPrdDistance = [1] * N

    rowsMissed = [0] * N
    colsMissed = [0] * N

    symbolsSum = sum(symbols)
    symbolsFac = math.factorial(len(symbols))
    symbolsSet = set(symbols)

    for i in range(N):
        for j in range(len(line_indexes)):
            rowsSumDistance[i] += phenotype[line_indexes[i][j]]
            rowsPrdDistance[i] *= phenotype[line_indexes[i][j]]

            colsSumDistance[i] += phenotype[column_indexes[i][j]]
            colsPrdDistance[i] *= phenotype[column_indexes[i][j]]

        rowsSumDistance[i] = abs(symbolsSum - rowsSumDistance[i])
        rowsPrdDistance[i] = abs(symbolsFac - rowsPrdDistance[i])

        colsSumDistance[i] = abs(symbolsSum - colsSumDistance[i])
        colsPrdDistance[i] = abs(symbolsFac - colsPrdDistance[i])

        rowsMissed[i] = len(list(symbolsSet - set([phenotype[l] for l in line_indexes[i]])))
        colsMissed[i] = len(list(symbolsSet - set([phenotype[l] for l in column_indexes[i]])))

    rootsColsPrd = [math.sqrt(i) for i in colsPrdDistance]
    rootsRowsPrd = [math.sqrt(i) for i in rowsPrdDistance]

    return (10 * (sum(rowsSumDistance) + sum(colsSumDistance)) \
                             + 50 * (sum(rowsMissed) + sum(colsMissed)) \
                             + sum(rootsColsPrd) + sum(rootsRowsPrd))

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
    return CROMO_FIT_SOL_DEFINED -((sum_lines + sum_columns + sum_boxes)/2)

def fitness_cromossome(individual):
    global fitness_calls
    fitnessFunction = { 'sum':fitness_cromossome_sum,
                        SUM_OF_SUM:fitness_cromossome_sum_of_sum,
                        CALCULATED:fitness_calculated,
                        }
    fitness_calls += 1
    return fitnessFunction[FITNESS](individual)

def crossover_uniform(population, individual1, individual2):
    newindividual1 = {CROMO_NAME:[0]*CROMO_LEN, CROMO_FIT:CROMO_FIT_INV}
    newindividual2 = {CROMO_NAME:[0]*CROMO_LEN, CROMO_FIT:CROMO_FIT_INV}
    for i in range(CROMO_LEN):
        if random() < 0.5:
            newindividual1[CROMO_NAME][i] = individual1[CROMO_NAME][i]
            newindividual2[CROMO_NAME][i] = individual2[CROMO_NAME][i]
        else:
            newindividual1[CROMO_NAME][i] = individual2[CROMO_NAME][i]
            newindividual2[CROMO_NAME][i] = individual1[CROMO_NAME][i]
    population.append(newindividual1)
    population.append(newindividual2)

def crossover_one_point(population, individual1, individual2):
    p = randint(0,CROMO_LEN-1)
    newindividual1 = copy.deepcopy(individual1)
    newindividual2 = copy.deepcopy(individual2)
    newindividual1[CROMO_FIT] = CROMO_FIT_INV
    newindividual2[CROMO_FIT] = CROMO_FIT_INV
    newindividual1[CROMO_NAME] = individual1[CROMO_NAME][:p] + individual2[CROMO_NAME][p:]
    newindividual2[CROMO_NAME] = individual2[CROMO_NAME][:p] + individual1[CROMO_NAME][p:]
    population.append(newindividual1)
    population.append(newindividual2)

def crossover_uniform_line(population, individual1, individual2):
    newindividual1 = {CROMO_NAME:[0]*CROMO_LEN, CROMO_FIT:CROMO_FIT_INV}
    newindividual2 = {CROMO_NAME:[0]*CROMO_LEN, CROMO_FIT:CROMO_FIT_INV}
    for i in range(N):
        if random() < 0.5:
            newindividual1[CROMO_NAME][i*N:i*N+N] = individual1[CROMO_NAME][i*N:i*N+N]
            newindividual2[CROMO_NAME][i*N:i*N+N] = individual2[CROMO_NAME][i*N:i*N+N]
        else:
            newindividual1[CROMO_NAME][i*N:i*N+N] = individual2[CROMO_NAME][i*N:i*N+N]
            newindividual2[CROMO_NAME][i*N:i*N+N] = individual1[CROMO_NAME][i*N:i*N+N]
    population.append(newindividual1)
    population.append(newindividual2)

def crossover_one_point_line(population, individual1, individual2):
    p = randint(0,N-1)
    newindividual1 = {CROMO_NAME:[0]*CROMO_LEN, CROMO_FIT:CROMO_FIT_INV}
    newindividual2 = {CROMO_NAME:[0]*CROMO_LEN, CROMO_FIT:CROMO_FIT_INV}

    newindividual1[CROMO_NAME] = individual1[CROMO_NAME][:p*N] + individual2[CROMO_NAME][p*N:]
    newindividual2[CROMO_NAME] = individual2[CROMO_NAME][:p*N] + individual1[CROMO_NAME][p*N:]

    population.append(newindividual1)
    population.append(newindividual2)

def crossover_uniform_column(population, individual1, individual2):
    newindividual1 = {CROMO_NAME:[0]*CROMO_LEN, CROMO_FIT:CROMO_FIT_INV}
    newindividual2 = {CROMO_NAME:[0]*CROMO_LEN, CROMO_FIT:CROMO_FIT_INV}
    for i in range(N):
        if random() < 0.5:
            newindividual1[CROMO_NAME][i::N] = individual1[CROMO_NAME][i::N]
            newindividual2[CROMO_NAME][i::N] = individual2[CROMO_NAME][i::N]
        else:
            newindividual1[CROMO_NAME][i::N] = individual2[CROMO_NAME][i::N]
            newindividual2[CROMO_NAME][i::N] = individual1[CROMO_NAME][i::N]
    population.append(newindividual1)
    population.append(newindividual2)

def crossover_one_line(population, individual1, individual2):
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

def crossover_one_column(population, individual1, individual2):
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

def crossover_one_box(population, individual1, individual2):
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
def crossover_cromossome(population, individual1, individual2):
    crossoverFunciton = [   crossover_one_point,
                            crossover_uniform,
                            crossover_uniform_line,
                            #crossover_uniform_column,
                            #crossover_one_point_line,
                            #crossover_one_line,
                            #crossover_one_column,
                            crossover_one_box,
                            ]
    choice(crossoverFunciton)(population, individual1, individual2)
    crossoverFunciton[0](population, individual1, individual2)
    crossoverFunciton[1](population, individual1, individual2)
    crossoverFunciton[2](population, individual1, individual2)

    crossoverFunciton[3](population, individual1, individual2)
    #crossoverFunciton[4](population, individual1, individual2)

def mutation_reset(individual):
    cromossome = copy.deepcopy(given_cromossome)
    for x in range(CROMO_LEN):
        if cromossome[x]==0 or cromossome[x]=='.':
            cromossome[x] = randint(1,N)
    population.append({CROMO_NAME:cromossome, CROMO_FIT:CROMO_FIT_INV})

def mutation_rotate(individual):
    newindividual = copy.deepcopy(individual)
    minimumNumberOfPoints = 2
    idx = randint(0,N)

    allPoints = [i for i, x in enumerate(given_cromossome[idx*N:idx*N+N]) if x == 0]

    numberOfAllPoints = len(allPoints)

    if (numberOfAllPoints < minimumNumberOfPoints): return

    randomNumberOfPoints = randrange(minimumNumberOfPoints, numberOfAllPoints + 1)

    points = sample(allPoints, randomNumberOfPoints)

    points.sort (reverse = (True if (random() < 0.5) else False))

    for i in points:
        xorValue = 0
        for j in points:
            xorValue ^= newindividual[CROMO_NAME][idx*N+j]

        newindividual[CROMO_NAME][idx*N+i] = xorValue

    xorValue = 0
    for point in points:
        xorValue ^= newindividual[CROMO_NAME][idx*N+point]
    newindividual[CROMO_NAME][idx*N+points[0]] = xorValue

    newindividual[CROMO_FIT] = CROMO_FIT_INV
    population.append(newindividual)

def mutation_swap_allele(individual):
    newindividual = copy.deepcopy(individual)
    p1 = randint(0,CROMO_LEN-1)
    p2 = randint(0,CROMO_LEN-1)
    if p1 != p2 \
            and given_cromossome[p1] == 0 \
            and given_cromossome[p2] == 0:
        v1 = newindividual[CROMO_NAME][p1]
        v2 = newindividual[CROMO_NAME][p2]
        newindividual[CROMO_NAME][p1] = v2
        newindividual[CROMO_NAME][p2] = v1
        newindividual[CROMO_FIT] = CROMO_FIT_INV
        population.append(newindividual)

def mutation_swap_2_line(individual):
    newindividual = copy.deepcopy(individual)
    i = randint(0, len(line_indexes)-1)
    idx1 = randint(0,N-1)
    idx2 = randint(0,N-1)
    if idx1 != idx2 \
            and given_cromossome[line_indexes[i][idx1]] == 0 \
            and given_cromossome[line_indexes[i][idx2]] == 0:
        newindividual[CROMO_NAME][line_indexes[i][idx1]] = individual[CROMO_NAME][line_indexes[i][idx2]]
        newindividual[CROMO_NAME][line_indexes[i][idx2]] = individual[CROMO_NAME][line_indexes[i][idx1]]
        newindividual[CROMO_FIT] = CROMO_FIT_INV
        population.append(newindividual)

def mutation_swap_3_line(individual):
    newindividual = copy.deepcopy(individual)
    i = randint(0, len(line_indexes)-1)
    idx1 = randint(0,N-1)
    idx2 = randint(0,N-1)
    idx3 = randint(0,N-1)
    if idx1 != idx2 != idx3 \
            and given_cromossome[line_indexes[i][idx1]] == 0 \
            and given_cromossome[line_indexes[i][idx2]] == 0  \
            and given_cromossome[line_indexes[i][idx3]] == 0:
        newindividual[CROMO_NAME][line_indexes[i][idx1]] = individual[CROMO_NAME][line_indexes[i][idx3]]
        newindividual[CROMO_NAME][line_indexes[i][idx2]] = individual[CROMO_NAME][line_indexes[i][idx1]]
        newindividual[CROMO_NAME][line_indexes[i][idx3]] = individual[CROMO_NAME][line_indexes[i][idx2]]
        newindividual[CROMO_FIT] = CROMO_FIT_INV
        population.append(newindividual)

def mutation_swap_lines(individual):
    newindividual = copy.deepcopy(individual)
    idx1 = randint(0, len(line_indexes)-1)
    idx2 = randint(0, len(line_indexes)-1)
    if (idx1 != idx2):
        for i in xrange(len(line_indexes)):
            if  given_cromossome[line_indexes[idx1][i]] == 0 \
                    and given_cromossome[line_indexes[idx2][i]] == 0:
                newindividual[CROMO_NAME][line_indexes[idx1][i]] = individual[CROMO_NAME][line_indexes[idx2][i]]
                newindividual[CROMO_NAME][line_indexes[idx2][i]] = individual[CROMO_NAME][line_indexes[idx1][i]]
        newindividual[CROMO_FIT] = CROMO_FIT_INV
        population.append(newindividual)

def mutation_swap_columns(individual):
    newindividual = copy.deepcopy(individual)
    idx1 = randint(0, len(column_indexes)-1)
    idx2 = randint(0, len(column_indexes)-1)
    if (idx1 != idx2):
        for i in xrange(len(column_indexes)):
            if  given_cromossome[line_indexes[idx1][i]] == 0 \
                    and given_cromossome[line_indexes[idx2][i]] == 0:
                newindividual[CROMO_NAME][column_indexes[idx1][i]] = individual[CROMO_NAME][column_indexes[idx2][i]]
                newindividual[CROMO_NAME][column_indexes[idx2][i]] = individual[CROMO_NAME][column_indexes[idx1][i]]
        newindividual[CROMO_FIT] = CROMO_FIT_INV
        population.append(newindividual)

def mutation_bit_wise(individual):
    newindividual = copy.deepcopy(individual)
    p = randint(0,CROMO_LEN-1)
    if  given_cromossome[p] == 0:
        newindividual[CROMO_NAME][p] = randint(1,N)
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
    mutationMethods = [mutation_reset,
                       mutation_rotate,
                       mutation_bit_wise,
                       #mutation_multi_values,
                       #mutation_swap_2_line,
                       #mutation_swap_3_line,
                       #mutation_swap_lines,
                       #mutation_swap_columns',
                       ]
    choice(mutationMethods)(individual)
    mutationMethods[0](individual)
    mutationMethods[1](individual)
    mutationMethods[2](individual)


#-----------------------------
# population functions
#-----------------------------
# calcule fitness of all individuals from population
def fitness_population(population):
    [p.update({CROMO_FIT:fitness_cromossome(p)}) for p in population if p[CROMO_FIT]==CROMO_FIT_INV or p[CROMO_FIT]==None]

def init_population_with_constraint_with_given_cromossome(population):
    values = range(1,N+1)
    for z in range(POP_LEN):
        ind = copy.deepcopy(given_cromossome)
        for x in range(N):
            missing = list(set(list_values) - set(given_cromossome[x*N:x*N+N]))
            shuffle(missing)
            for i in range(x*N,x*N+N):
                if ind[i]==0 and missing:
                    ind[i]= missing.pop()

        population.append({CROMO_NAME:ind, CROMO_FIT:CROMO_FIT_INV})

# init random population
def init_population_with_constraint(population):
    values = range(1,N+1)
    for z in range(POP_LEN):
        i = []
        [i.extend(sample(values, len(values))) for x in range(N)]
        population.append({CROMO_NAME:i, CROMO_FIT:CROMO_FIT_INV})

def init_population_random_with_given_cromossome(population):
    for i in range(POP_LEN):
        cromossome = copy.deepcopy(given_cromossome)
        for x in range(CROMO_LEN):
            if cromossome[x]==0:
                cromossome[x] = randint(1,N)
        population.append({CROMO_NAME:cromossome, CROMO_FIT:CROMO_FIT_INV})

# init random population
def init_population_random(population):
    [population.append({CROMO_NAME:[randint(1,N) for x in range(CROMO_LEN)], CROMO_FIT:CROMO_FIT_INV}) for i in range(POP_LEN)]

def init_population(population):
    init = {'R':init_population_random,
            'C':init_population_with_constraint,
            RANDOM:init_population_random_with_given_cromossome,
            CONSTRAINT:init_population_with_constraint_with_given_cromossome
            }

    init[INIT](population)

def selection_tournament(population):
    newpopulation = []
    for i in range(0, TOURNAMENT_SELECTION):
        indexes = [randrange(0, len(population)) for i in range(TOURNAMENT_SIZE)]
        ind = copy.deepcopy(min([population[i] for i in indexes], key = lambda o: o[CROMO_FIT]))
        newpopulation.append(ind)
    return newpopulation

# select population individuals
def selection_population(population):
    newpopulation = []
    spopulation = sorted(population, key=itemgetter(CROMO_FIT))
    #print(str(spopulation[0][CROMO_NAME]))
    if BEST_K_SELECTION >= 1:
        inserted = 1
        last = spopulation[0]
        for k in spopulation[1:]:
            if last[CROMO_NAME] != k[CROMO_NAME]:
                newpopulation.append(k)
                inserted += 1
                last = k
            if inserted >= BEST_K_SELECTION:
                break
    if WORST_K_SELECTION >= 1:
        newpopulation.extend(spopulation[-WORST_K_SELECTION:])
    newpopulation.extend(selection_tournament(population))
    newpopulation = sorted(newpopulation, key=itemgetter(CROMO_FIT))
    return newpopulation, \
           [newpopulation[0][CROMO_FIT], \
            newpopulation[-1][CROMO_FIT], \
            sum([d[CROMO_FIT] for d in newpopulation])/ len(newpopulation)]

# crossover individuals randomly
def crossover_population(population):
    [crossover_cromossome(population, population[randint(0,len(population)-1)], population[randint(0,len(population)-1)]) \
     for i in range(0,len(population)) \
     if random() <= CROSS_RATE]


# mutation individuals randomly
def mutation_population(population):
    [mutation_cromossome(population[randint(0,len(population)-1)]) \
     for i in range(0,len(population)) \
     if random() <= MUTAT_RATE]


def remove_duplicates_lines(population):
    r_CROMO_LEN = range(CROMO_LEN)
    r_N = range(N)
    for p in population:
        for i in r_N:
            content_idx = r_CROMO_LEN[i*N:i*N+N]
            content_value = p[CROMO_NAME][i*N:i*N+N]
            missings = list(set(list_values)-set(content_value))
            if missings:
                repetitions = []
                [repetitions.append(idx) for idx,item in enumerate(content_value) \
                 if (item in content_value[:idx] or item in content_value[idx+1:]) and (given_cromossome[content_idx[idx]] == 0)]
                for r in repetitions:
                    if missings:
                        shuffle(missings)
                        if missings[-1] not in content_value:
                            p[CROMO_NAME][content_idx[r]]  = missings.pop()
                    else:
                        break

def remove_duplicates_columns(population):
    r_CROMO_LEN = range(CROMO_LEN)
    r_N = range(N)
    for p in population:
        for i in r_N:
            content_idx = r_CROMO_LEN[i::N]
            content_value = p[CROMO_NAME][i::N]
            missings = list(set(list_values)-set(content_value))
            if missings:
                repetitions = []
                [repetitions.append(idx) for idx,item in enumerate(content_value) \
                 if (item in content_value[:idx] or item in content_value[idx+1:]) and (given_cromossome[content_idx[idx]] == 0)]
                for r in repetitions:
                    if missings:
                        shuffle(missings)
                        if missings[-1] not in content_value:
                            p[CROMO_NAME][content_idx[r]]  = missings.pop()
                    else:
                        break

def remove_duplicates(population):
    if REMOVE==True:
        remove_duplicates_lines(population)
        remove_duplicates_columns(population)
    return

def found_solution(population):
    return [p for p in population if p[CROMO_FIT]==CROMO_FIT_SOL]

def print_sudoku(individual):
    print("Found Solution")
    print("Sudoku")
    for i in range(N):
        print(individual[CROMO_NAME][i*N:i*N+N])

def print_parameters(param):
    for k,v in param.items():
        print(k, v)

def save_data(times, x, i, param):
    f = open(str(x)+'data.txt', 'a')
    f.write(str(({'times':times},{'iteration':i},param, {'fitness_calls':fitness_calls})) + '\n')
    print(({'times':times},{'iteration':i},param, {'fitness_calls':fitness_calls}))
    f.close()

def sudoku_solver(times, idx, param):
    global fitness_calls
    bestValue = 0
    restartCount = 0
    fitness_calls = 0
    population = []
    init_population(population)
    remove_duplicates(population)
    fitness_population(population)
    for i in range(GENERATIONS_LEN):
        crossover_population(population)
        mutation_population(population)
        remove_duplicates(population)
        fitness_population(population)
        population, statistics = selection_population(population)
        print('Iteration ' + str(i) + ', Best: ' + str(statistics[0]) + ', Worst: ' + str(statistics[1]) + ', Average: ' + str(statistics[2]))

        # check result
        s = found_solution(population)
        if len(s) > 0:
            print_sudoku(s[0])
            save_data(times, idx, i, param)
            break
        else:
            if bestValue == statistics[2]:
                restartCount = restartCount + 1
            else:
                restartCount = 0
                bestValue = statistics[2]

        if restartCount == RESTART_VALUE:
            restartCount = 0
            population = []
            init_population(population)

def from_file(filename, sep='\n'):
    "Parse a file into a list of strings, separated by sep."
    return [[0 if x=='.' else int(x) for x in l]for l in file(filename).read().strip().split(sep)]

idx = 2
params = parameters[idx]

POP_LEN = params[POPULATION]
GENERATIONS_LEN = params[GENERATIONS]
MUTAT_RATE = params[MUTATION_PROB]
CROSS_RATE = params[CROSSOVER_PROB]
TOURNAMENT_SIZE = params[TOURNAMENT]
FITNESS = params[FITNESS_F]
REMOVE = params[REMOVE_DUP]
INIT = params[INIT_POP]

BEST_K_SELECTION = 10#int(POP_LEN * BEST_K_PERCENTAGE)
WORST_K_SELECTION = 0#10#int(POP_LEN * WORST_K_PERCENTAGE)
TOURNAMENT_SELECTION = int(POP_LEN * TOURNAMENT_PERCENTAGE)

for s in from_file('./tests/9x9-medio.txt'):
#for s in from_file('./tests/top95.txt'):
    given_cromossome = s
    #given_cromossome = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    print(params, {'restart':RESTART_VALUE}, )
    for times in range(0,5):
        sudoku_solver(times,idx, params)






