import numpy as np

# fitSolution {1,2,3,4}
fitSolution = 1

# how many times of experiment to conduct
EXTIMES = 1

# two rate (parameter)
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.08

# to record every max fitness in every iteration
maxFitnessIter = []

# in the crossover_rate experiment, record every max fitness
maxFitnessCrs = []

# in the mutation_rate experiment, record every max fitness
maxFitnessMut = []

# size of DNA (1111)
M = 4
# size of  population 
P = 100
# number of generations
N = 100

# domain of x1,x2,x3
xDomain = [0,15]

def getFitness(pop):
    x1, x2, x3 = encodeDNA(pop)
    ans = compFitness(x1, x2, x3)
    
    # Fitness solution1: move linearly 
    if (fitSolution == 1):
        return ans + 1e-3 - np.min(ans) 

    # Fitness solution2: square
    if (fitSolution ==2):
        return ans ** 2

    # Fitness solution3: root + move linearly
    if (fitSolution ==3):
        return (ans + 1e-3 - np.min(ans)) ** 0.5

    # Fitness solution4: minimal 
    if (fitSolution ==4):
        return -(ans - np.max(ans) - 1e-3)




def compFitness(x1, x2, x3):
    return (2*x1*x1 - 3*x2*x2 - 4*x1 + 5*x2 + x3)
    

# binary to decimal
def encodeDNA(pop):  
    #cut a line into 3 pieces, x1,x2,x3
    x1Pop = pop[:, 0:M]  
    x2Pop = pop[:, M:M * 2]  
    x3Pop = pop[:, M * 2:]  

    # pop:(P,M)*(M,1) --> (P,1)   2 to 10
    x1 = x1Pop.dot(2 ** np.arange(M)[::-1]) / float(2 ** M - 1) * (xDomain[1] - xDomain[0]) + xDomain[0]
    x2 = x2Pop.dot(2 ** np.arange(M)[::-1]) / float(2 ** M - 1) * (xDomain[1] - xDomain[0]) + xDomain[0]
    x3 = x3Pop.dot(2 ** np.arange(M)[::-1]) / float(2 ** M - 1) * (xDomain[1] - xDomain[0]) + xDomain[0]
    return x1,x2,x3


def mutation(child, MUTATION_RATE):
    if np.random.rand() < MUTATION_RATE:  
        # decide which point to be mutated by randomly pick up
        mutate_point = np.random.randint(0, M * 3)  
        # reverse binaryly
        child[mutate_point] = child[mutate_point] ^ 1  


def crossover_and_mutation(pop, CROSSOVER_RATE):
    new_pop = []
    # pick a father and a mother to corssover
    for father in pop:  
        child = father  
        # crossover happens under CROSS_RATE
        if np.random.rand() < CROSSOVER_RATE:  
            mother = pop[np.random.randint(P)]  
            # select the crossover point randomly
            cross_points = np.random.randint(low=0, high=M * 3)
            # the latter part is from mother  
            child[cross_points:] = mother[cross_points:]  
        # every child has possbility to mutate
        mutation(child,MUTATION_RATE) 
        # new population 
        new_pop.append(child)

    return new_pop


def roulette(pop, fitness):  
    # roulette by pick index with weight(fitness/fitness.sum())
    idx = np.random.choice(np.arange(P), size=P, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]

    # sumFits = sum(fitness)
    # # generate a random number
    # rndPoint = np.random.uniform(0, sumFits)
    # # calculate the index: O(N)
    # accumulator = 0.0
    # inx = []
    # for ind, val in enumerate(fitness):
    #     accumulator += val
    #     if accumulator >= rndPoint:
    #     	inx.append(ind)
    #     	return pop[inx]
    #     else:
    #     	inx.append(ind)


def printPop(pop):
    fitness = getFitness(pop)
    max_fitness_index = np.argmax(fitness)
    x1, x2, x3 = encodeDNA(pop)
    x1=x1[max_fitness_index]
    x2=x2[max_fitness_index]
    x3=x3[max_fitness_index]

    print(pop)
    print("x1: ",x1)
    print("x2: ",x2)
    print("x3: ",x3)
    print("Max Fitness: ", compFitness(x1,x2,x3))


if __name__ == "__main__":

    for _ in range(EXTIMES):
        # generate population matrix with random number
        pop = np.random.randint(0, 2, size=(P, M * 3)) 
        # run iterations
        for i in range(N): 
            pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
            fitness = getFitness(pop)
            pop = roulette(pop, fitness) 

            if EXTIMES==1:
                max_fitness_index = np.argmax(fitness)
                x1, x2, x3 = encodeDNA(pop)
                x1=x1[max_fitness_index]
                x2=x2[max_fitness_index]
                x3=x3[max_fitness_index]
                maxFitnessIter.append(compFitness(x1,x2,x3))

        if EXTIMES==1:
            print(maxFitnessIter)
        else:
            fitness = getFitness(pop)
            max_fitness_index = np.argmax(fitness)
            x1, x2, x3 = encodeDNA(pop)
            x1=x1[max_fitness_index]
            x2=x2[max_fitness_index]
            x3=x3[max_fitness_index]
            maxFitnessCrs.append(compFitness(x1,x2,x3))
    if EXTIMES!=1:
        print(maxFitnessCrs)




