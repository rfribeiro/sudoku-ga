from random import randint

N = 9
POP_LEN = 100
CROMO_LEN = N*N
INTER_LEN = 1000
MUTAT_RATE = 0.05
CROSS_RATE = 0.7

population = []

# calulate all integer with same value
def fitness(population):
    return 0

def init(population):
    population = [[randint(1,9) for x in range(CROMO_LEN)] for y in range(POP_LEN)]
    return 0

def crossover():
    return 0

def mutation():
    return 0

def selection(population):
    return 0

init(population)
fitness(population)
for i in range(INTER_LEN):
    selection(population)
    crossover()
    mutation()
    fitness(population)
    # check result


