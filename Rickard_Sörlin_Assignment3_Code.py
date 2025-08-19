from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import random


# The dataset is uploaded
f = open("Assignment 3 medical_dataset.DATA")
dataset_X = []
dataset_y = []
line = " "
while line != "":
    line = f.readline()
    line = line[:-1]
    if line != "":
        line = line.split(",")
        floatList = []
        for i in range(len(line)):
            if i < len(line)-1:
                floatList.append(float(line[i]))
            else:
                value = float(line[i])
                if value == 0:
                    dataset_y.append(0)
                else:
                    dataset_y.append(1)
        dataset_X.append(floatList)
f.close()

# The dataset is splited into training and test.
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, test_size = 0.25, random_state = 0)


# The dataset is scaled

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# The model is created
""" Amount of input if training set is set to 75% = 222 if doing sqrt of 222 its about 14,89 change K=3 to 13? """

model = KNeighborsClassifier(n_neighbors=3)


# Function that calculates the fitness of a solution
def calculateFitness(solution):
    fitness = 0

    # The features are selected according to solution
    X_train_Fea_selc = []
    X_test_Fea_selc = []
    for example in X_train:
        X_train_Fea_selc.append([a*b for a,b in zip(example,solution)])
    for example in X_test:
        X_test_Fea_selc.append([a*b for a,b in zip(example,solution)])

    model.fit(X_train_Fea_selc, y_train)

    # We predict the test cases
    y_pred = model.predict(X_test_Fea_selc)
    # We calculate the Accuracy
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0][0] # True positives
    FP = cm[0][1] # False positives
    TN = cm[1][1] # True negatives
    FN = cm[1][0] # False negatives

    fitness = (TP + TN) / (TP + TN + FP + FN)

    return round(fitness *100,2)


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


def randomrestarthillclimb(max_fitness_count):
    print('\n===== Random Restart Hill climb =====')
    fitness_count = 0

    # Start from a random initial state in the state search space
    currentsolution = generaterandom()

    # Set the random initial state as the initial best
    bestsolution = currentsolution
    curentsolutionfitness = calculateFitness(currentsolution)
    bestsolutionfitness = curentsolutionfitness
    fitness_count += 1

    # Begin investigate the search space with random restart Hill climbing
    while True:

        #  The base case stop the algorithm if reached max fitness count
        if fitness_count >= max_fitness_count:
            return bestsolution, bestsolutionfitness

        # Get the neighbours of currentsolution and evaluate wich is the best one
        neighbors = getneighbors(currentsolution)
        bestneighbor, bestneighborfitness, fitness_count = getbestneighbor(neighbors, fitness_count)

        # Are we on right track upp to a hill?, maybe to a local / global maximum we dont know
        if bestneighborfitness > curentsolutionfitness:
            currentsolution = bestneighbor
            curentsolutionfitness = bestneighborfitness

            if curentsolutionfitness > bestsolutionfitness:
                bestsolution = currentsolution
                bestsolutionfitness = curentsolutionfitness
                print("Best solution fitness ( ", fitness_count, "/", max_fitness_count, "):",
                      bestsolutionfitness)

        # Else we look to be stuck and the way is only down now & cant find a bether view
        else:
            # start from random state in search space to check if there are a bether view.
            currentsolution = generaterandom()


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


def main():
    max_fitness_count = 5000
    totalfitnessrrhc = 0
    totalfitnessvns = 0
    executiontimes = 10

    # K = 3  for the Nearest Neighbours classifier (not changed) keep this during lab and report assignment
    # The amount of training data is 75% = 222 (sqrt of 222) is about 14,89
    # if setting K = 13 it looks like the future selection is more stable and has less variations



    # Start the feuture selection using Random Restart Hill Climbing Search & K Nearest Neighbours classifier
    for executiontime in range(executiontimes):
        print(f'\nExecution count: {executiontime + 1}')
        bestsolution_hcrrs, bestsolutionfitness = randomrestarthillclimb(max_fitness_count)
        totalfitnessrrhc += bestsolutionfitness
        print('\nBest solution:', bestsolution_hcrrs)
        print('Best solution fitness:', bestsolutionfitness)

    # Start the feuture selection using Variable neighbour Search & K Nearest Neighbours classifier
    for executiontime in range(executiontimes):
        print(f'\nExecution count: {executiontime + 1}')
        bestsolution_vnrs, bestsolutionfitness = variableneighborsearch(max_fitness_count)
        totalfitnessvns += bestsolutionfitness
        print('\nBest solution:', bestsolution_vnrs)
        print('Best solution fitness:', bestsolutionfitness)

    print('\n==================== Hill Climb Random Restart search =========================')
    print(f'Average performence fitness: {totalfitnessrrhc/executiontimes:.2f} %')
    print('===============================================================================')

    print('==================== Variable Neighbourhood Restart search ====================')
    print(f'Average performence fitness: {totalfitnessvns/executiontimes:.2f} %')
    print('===============================================================================')


if __name__ == "__main__":
    main()
