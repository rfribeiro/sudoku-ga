import copy, math
from random import randint, random, sample
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

SLOT = 3
N = SLOT*SLOT
POP_LEN = 200
CROMO_LEN = N*N
GENERATIONS_LEN = 10000
MUTAT_RATE = 0.3
CROSS_RATE = 0.90
CROMO_FIT_INV = -1
CROMO_FIT_SOL = 0
BEST_K_PERCENTAGE = 0.01
WORST_K_PERCENTAGE = 0.0
TOURNAMENT_PERCENTAGE = 0.99
BEST_K_SELECTION = int(POP_LEN * BEST_K_PERCENTAGE)
WORST_K_SELECTION = int(POP_LEN * WORST_K_PERCENTAGE)
TOURNAMENT_SELECTION = int(POP_LEN * TOURNAMENT_PERCENTAGE)
TOURNAMENT_SIZE = 3
RESTART_VALUE = 500

population = []

line_indexes = [[N1*N+N2 for N2 in range(0,N)] for N1 in range(0,N)]
column_indexes = [[N1+N*N2 for N2 in range(0,N)] for N1 in range(0,N)]

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
                        2,0,0,0,
                        3,0,0,1,
                        4,0,0,0]
    given_cromossome = [0,0,3,0,
                        0,4,0,0,
                        0,0,0,2,
                        4,0,1,0]
else:
    given_cromossome = [  0, 3, 6, 0, 5, 0, 9, 7, 0,
                          5, 0, 9, 2, 0, 1, 0, 0, 3,
                          0, 1, 0, 0, 6, 9, 8, 2, 0,
                          4, 0, 8, 7, 0, 3, 0, 0, 5,
                          0, 2, 0, 9, 6, 0, 7, 0, 8,
                          7, 0, 3, 0, 8, 0, 0, 4, 2,
                          0, 0, 7, 5, 0, 9, 3, 1, 2,
                          3, 9, 0, 0, 0, 7, 0, 0, 6,
                          0, 5, 4, 0, 3, 6, 0, 0, 8]

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
def fitness_calculated(individual):
    phenotype = individual[CROMO_NAME]

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
    return fitness_cromossome_sum_of_sum(individual)

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
    #crossover_one_point(population, individual1, individual2)
    crossover_one_line(population, individual1, individual2)
    #crossover_one_column(population, individual1, individual2)
    #crossover_one_box(population, individual1, individual2)

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

def mutation_new_value(individual):
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
    mutation_new_value(individual)
    #mutation_bit_wise(individual)
    #mutation_multi_values(individual)
    mutation_swap_2_line(individual)
    mutation_swap_3_line(individual)
    #mutation_swap_lines(individual)
    #mutation_swap_columns(individual)


#-----------------------------
# population functions
#-----------------------------
# calcule fitness of all individuals from population
def fitness_population(population):
    [p.update({CROMO_FIT:fitness_cromossome(p)}) for p in population if p[CROMO_FIT]==CROMO_FIT_INV or p[CROMO_FIT]==None]

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
    #init_population_random(population)
    #init_population_with_constraint(population)
    init_population_random_with_given_cromossome(population)

def selection_tournament(population):
    newpopulation = []
    for i in range(0, TOURNAMENT_SELECTION):
        individual = population[randint(0,POP_LEN-1)].copy()
        for r in range(1,TOURNAMENT_SIZE):
            individual1 = population[randint(0,POP_LEN-1)]
            if (individual[CROMO_FIT] > individual1[CROMO_FIT]):
                individual = individual1.copy()
        newpopulation.append(individual)
    return newpopulation

# select population individuals
def selection_population(population):
    newpopulation = []
    spopulation = sorted(population, key=itemgetter(CROMO_FIT))
    #print(str(spopulation[0][CROMO_NAME]))
    newpopulation.extend(spopulation[:BEST_K_SELECTION])
    #newpopulation.extend(spopulation[-WORST_K_SELECTION:])
    newpopulation.extend(selection_tournament(population))
    return newpopulation, [newpopulation[0][CROMO_FIT], newpopulation[-1][CROMO_FIT], sum(d[CROMO_FIT] for d in newpopulation) / len(newpopulation)]

# crossover individuals randomly
def crossover_population(population):
    for i in range(0,len(population)):
        r = random()
        if r <= CROSS_RATE: # do crossover
            r1 = randint(0,POP_LEN-1)
            r2 = randint(0,POP_LEN-1)
            #print(r1,r2, len(population))
            crossover_cromossome(population, population[r1], population[r2])

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

    if restartCount == RESTART_VALUE:
        restartCount = 0
        population = []
        init_population(population)




