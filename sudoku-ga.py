from random import randint, random
from collections import Counter

SLOT = 3
N = SLOT*SLOT
POP_LEN = 100
CROMO_LEN = N*N
GENERATIONS_LEN = 1000
MUTAT_RATE = 0.05
CROSS_RATE = 0.7
CROMO_NAME = 'cromossome'
CROMO_FIT = 'fitness'
CROMO_FIT_INV = -1
CROMO_FIT_SOL = 0
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

# crossover two individuals
def crossover_cromossome(individual1, individual2):
    newindividual = individual1 # TODO need to change it only for test
    newindividual[CROMO_FIT] = CROMO_FIT_INV

def mutation_swap(individual):
    p1 = randint(0,CROMO_LEN)
    p2 = randint(0,CROMO_LEN)
    v1 = individual[CROMO_NAME][p1]
    v2 = individual[CROMO_NAME][p2]
    individual[CROMO_NAME][p1] = v2
    individual[CROMO_NAME][p2] = v1

def mutation_new_value(individual):
    individual[CROMO_NAME][randint(0,CROMO_LEN)] = randint(1,N)

# mutate individual
def mutation_cromossome(individual):
    mutation_new_value(individual)
    mutation_swap(individual)
    individual[CROMO_FIT] = CROMO_FIT_INV
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
    if random() <= CROSS_RATE: # do crossover
        crossover_cromossome(population[randint(1,POP_LEN)], population[randint(1,POP_LEN)])

# mutation individuals randomly
def mutation_population(population):
    if random() <= MUTAT_RATE: # do mutation
        mutation_cromossome(population[randint(1,POP_LEN)])

def found_solution(population):
    return [p for p in population if p[CROMO_FIT]==CROMO_FIT_SOL]

def print_sudoku(individual):
    print("Found Solution")
    print("Sudoku")
    for i in range(N):
        print(individual[CROMO_NAME][i*N:i*N+N])

print_sudoku(sample)
fitness_cromossome(sample)

init_population(population)
fitness_population(population)
for i in range(GENERATIONS_LEN):
    selection_population(population)
    crossover_population()
    mutation_population()
    fitness_population(population)
    # check result
    s = found_solution(population)
    if len(s) > 0:
        print_sudoku(s[0])
        break


