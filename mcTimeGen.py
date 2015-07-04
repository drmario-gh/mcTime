# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 12:43:35 2014
Genetic algorithm to build a timetable
@author: mariol
"""

import numpy as np
import matplotlib.pyplot as plt


minDurationShift = 4 * 2
maxDurationShift = 8 * 2
minDurationPartShift = 4 * 2
minRest = 0.5 * 2
maxRest = 2.5 * 2
#Other
numWorkers = 16
# En el ejemplo, 'workersHalfHours' comprende las medias horas entre las 8am
# y las 3am (38 posiciones)
workersHalfHours = np.array([3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 4, 7,
                             6, 5, 4, 3, 3, 2, 2, 2, 2, 2, 3, 6, 8,
                             10, 12, 10, 9, 9, 8, 4, 4, 3, 3, 3, 1])
finalTime = workersHalfHours.size
# Genetic algorithm parameters
popuSize = 200
probCross = 0.8
mutaSize = 2
probMutation = 0.1
numElitism = 20
probFullTime = 0.5
maxIt = 300
#-----------------------------
#Other
maxError = np.sum(np.maximum(workersHalfHours, numWorkers-workersHalfHours))

# Auxiliar functions
def generatePartShift():
    startTime1 = np.random.random_integers(0,
                                           finalTime - minDurationPartShift -
                                           minRest - minDurationPartShift)
    duration1 = np.random.random_integers(minDurationPartShift,
                                          min(finalTime - minDurationPartShift - 
                                              minRest - startTime1,
                                              maxDurationShift - minDurationPartShift))
    finalTime1 = startTime1 + duration1
    restDuration = np.random.random_integers(minRest, 
                                             min(finalTime - minDurationPartShift - 
                                                 finalTime1, maxRest))
    startTime2 = finalTime1 + restDuration
    duration2 = np.random.random_integers(minDurationPartShift, 
                                          min(finalTime - startTime2,
                                              maxDurationShift - duration1))
    finalTime2 = startTime2+duration2
    return(startTime1, finalTime1, startTime2, finalTime2)

    
def generateFullShift():
    startTime1 = np.random.random_integers(0, finalTime - minDurationShift)
    duration = np.random.random_integers(minDurationShift,
                                         min(finalTime-startTime1,
                                             maxDurationShift))
    finalTime1 = startTime1+duration    
    return(startTime1, finalTime1, 0, 0)
    

def integer2binaryShift(workerInteger):
    [startTime1, finalTime1, startTime2, finalTime2] = workerInteger
    workerBinary = np.zeros((workersHalfHours.size))
    workerBinary[startTime1:finalTime1+1] = 1
    if finalTime2 != 0:
        workerBinary[startTime2:finalTime2+1] = 1
    return workerBinary
    
    
def crossover(gen1,gen2):
    pointOfCross = np.random.random_integers(1,numWorkers-2)
    return(np.concatenate((gen1[0:pointOfCross,:],gen2[pointOfCross:numWorkers,:])), \
           np.concatenate((gen2[0:pointOfCross,:],gen1[pointOfCross:numWorkers,:])))

    
def mutation(gen):
    mutationIndices = np.random.permutation(numWorkers)
    mutationIndices = mutationIndices[0:mutaSize]
    for i in mutationIndices:
        if np.random.random([1]) < probFullTime:
            gen[i,:] = generateFullShift()
        else :
            gen[i,:] = generatePartShift()
    return gen 


def getGenWorkersHalfHours(genInteger):
    genBinary = np.zeros([numWorkers,workersHalfHours.size])
    for j in range(numWorkers-1):
        genBinary[j,:] = integer2binaryShift(genInteger[j,:])
    return np.sum(genBinary,0)


def computeFitness(genInteger,maxError):
    genError = np.sum(abs(getGenWorkersHalfHours(genInteger)-workersHalfHours))/maxError
    return 1-genError
    

# Generacion de poblacion aleatoria, cumpliendo con restricciones de trabajador
popuInteger = np.zeros((popuSize,numWorkers,4))
popuBinary = np.zeros((popuSize,numWorkers,workersHalfHours.size))
auxPopuInteger = popuInteger

for i in range(0,popuSize):
    for j in range(0,numWorkers):
        if np.random.random([1]) < probFullTime:
            popuInteger[i, j, :] = generateFullShift()
        else:
            popuInteger[i, j, :] = generatePartShift()

# Fitness function of each gen 
popuFitness = np.zeros([popuSize])
for i in range(popuSize):
    popuFitness[i] = computeFitness(popuInteger[i, :, :], maxError)
cumulativeFitness = np.cumsum(popuFitness)
it=0
maxPopuFitness = np.zeros([maxIt])
while it < maxIt:
    #Elitism
    sortedIndexPopuFitness = np.argsort(popuFitness)
    auxPopuInteger[0:numElitism-1,:,:] = popuInteger[sortedIndexPopuFitness[popuSize-numElitism+1:],:,:]
    #Crossover
    numCrossPairs = np.random.binomial((popuSize-numElitism)/2,probCross)
    numNoCrossGenes = popuSize - 2*numCrossPairs - numElitism
    for k in range(0,numCrossPairs-1):
        #Selection
        selected1 = np.argmax(cumulativeFitness >= np.random.random()*cumulativeFitness[-1])
        selected2 = np.argmax(cumulativeFitness >= np.random.random()*cumulativeFitness[-1])
        cross = crossover(popuInteger[selected1,:,:],popuInteger[selected2,:,:])
        auxPopuInteger[numElitism+2*k,:,:] = cross[0]
        auxPopuInteger[numElitism+2*k+1,:,:] = cross[1]
    for k in range(0,numNoCrossGenes-1):
        selected =  np.argmax(cumulativeFitness >= np.random.random()*cumulativeFitness[-1])
        auxPopuInteger[numElitism+2*numCrossPairs+k,:,:] = popuInteger[selected,:,:]
    #Mutation
    numMutation = np.random.binomial(popuSize,probMutation)
    indexToMutate = np.random.random_integers(numElitism,popuSize-1,numMutation)
    for k in range (0,numMutation-1):
           auxPopuInteger[indexToMutate[k],:,:] = mutation(auxPopuInteger[indexToMutate[k],:,:]);
    popuInteger = auxPopuInteger       
    # Fitness function
    for i in range(popuSize):
        popuFitness[i] = computeFitness(popuInteger[i, :, :], maxError)
    cumulativeFitness = np.cumsum(popuFitness)
    bestSolInd = np.argmax(popuFitness)
    maxPopuFitness[it] = popuFitness[bestSolInd]
    print it
    it = it+1
    
bestSolution = popuInteger[bestSolInd,:,:]*0.5+8
genWorkersHalfHours = getGenWorkersHalfHours(popuInteger[bestSolInd,:,:])
firstShift = bestSolution[:,0:2]
secondShift = bestSolution[:,2:4]
#broken_barh?
y = np.arange(numWorkers)
x = np.arange(workersHalfHours.size)
ax1 = plt.subplot(211)
plt.barh(y,firstShift[:,1]-firstShift[:,0],0.5,firstShift[:,0],hold=True)
plt.barh(y,secondShift[:,1]-secondShift[:,0],0.5,secondShift[:,0],hold=True)
plt.xticks(x)
plt.grid()
plt.subplot(212, sharex=ax1)
plt.grid()
plt.bar(x*0.5+8,genWorkersHalfHours-workersHalfHours,width=0.5)
plt.show()
print maxPopuFitness[it-1]







