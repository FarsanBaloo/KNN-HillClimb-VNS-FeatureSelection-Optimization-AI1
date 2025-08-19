def generaterandom():
    """ Function handels to generate a randome solution to explore in the search space """
    currentsolution = []
    for i in range(1, 14):
        currentsolution.append(random.randint(0, 1))
    return currentsolution


def getneighbors(currentsolution, num_variables=1):
    """Function generates all neighbours for current solution and returns them in a nested List
     Hill climb randome restart will use default amount of num_variabels = 1
     Variabel neighbor restart search will increase num_variabels in steps to search the search space more
     when trapped in a local maximum"""

    neighbors = []

    # Do same amount of changes as num_variabels informes and get all possible neighbor
    if num_variables == 1:
        for i in range(len(currentsolution)):
            neighbor = currentsolution.copy()
            # Change elements value to the opposit in range (0,1) at current index
            neighbor[i] = 1 - neighbor[i]
            # add the neighbour to the neighbors list for later evaluation
            neighbors.append(neighbor)

    # Do same amount of changes as num_variabels informes and get all possible neighbor
    elif num_variables == 2:
        for i in range(len(currentsolution)):
            for j in range(i + 1, len(currentsolution)):
                neighbor = currentsolution.copy()
                # Change elements value to the opposit in range (0,1) at current index
                neighbor[i] = 1 - neighbor[i]
                neighbor[j] = 1 - neighbor[j]
                # add the neighbour to the neighbors list for later evaluation
                neighbors.append(neighbor)

    # Do same amount of changes as num_variabels informes and get all possible neighbor
    elif num_variables == 3:
        for i in range(len(currentsolution)):
            for j in range(i + 1, len(currentsolution)):
                for k in range(j + 1, len(currentsolution)):
                    neighbor = currentsolution.copy()
                    # Change elements value to the opposit in range (0,1) at current index
                    neighbor[i] = 1 - neighbor[i]
                    neighbor[j] = 1 - neighbor[j]
                    neighbor[k] = 1 - neighbor[k]
                    # add the neighbour to the neighbors list for later evaluation
                    neighbors.append(neighbor)

    return neighbors


def getbestneighbor(neighbors, fitness_count):
    """ Function evaluates all neighbours against fitness function and return the best one """
    bestneighbor = 0
    bestneighborfitness = 0
    fitness_count = fitness_count
    # For each neigbour evaluate the best solution of them
    for neighbor in neighbors:
        neighborfitness = calculateFitness(neighbor)
        fitness_count += 1
        if neighborfitness > bestneighborfitness:
            bestneighbor = neighbor
            bestneighborfitness = neighborfitness

    return bestneighbor, bestneighborfitness, fitness_count


def variableneighborsearch(max_fitness_count):
    print('\n===== Variable Neighbour Search =====')
    fitness_count = 0
    # start with
    num_variables = 1

    # Start from a random initial state in the state search space
    currentsolution = generaterandom()

    # Set the random initial state as the initial best
    bestsolution = currentsolution
    curentsolutionfitness = calculateFitness(currentsolution)
    bestsolutionfitness = curentsolutionfitness
    fitness_count += 1

    # Begin investigate the search space with variable neighbour search
    while True:

        #  The base case stop the algorithm if reached max fitness count
        if fitness_count >= max_fitness_count:
            return bestsolution, bestsolutionfitness

        # Get and evaluate the neighbours of currentsolution
        neighbors = getneighbors(currentsolution, num_variables)
        bestneighbor, bestneighborfitness, fitness_count = getbestneighbor(neighbors, fitness_count)

        # Are we on right track upp to a hill?, maybe to a local / global maximum we dont know
        if bestneighborfitness > curentsolutionfitness:
            currentsolution = bestneighbor
            curentsolutionfitness = bestneighborfitness

            if curentsolutionfitness > bestsolutionfitness:
                bestsolution = currentsolution
                bestsolutionfitness = curentsolutionfitness
                num_variables = 0
                print("Best solution fitness ( ", fitness_count, "/", max_fitness_count, "):",
                      bestsolutionfitness)

        # It looks like we are stuck at a local maxima! and the way is only down now and cant find a bether view
        else:
            # try search in neighbours neighbour in the search space to search more of it.
            if num_variables < 3:
                num_variables += 1

            # if the neighbour neighbour neighbour still doesnt have a beather view
            else:
                # Go and search the state space from a diffren view at a "random" place
                currentsolution = generaterandom()
                num_variables = 0




if __name__ == "__main__":
    main()

