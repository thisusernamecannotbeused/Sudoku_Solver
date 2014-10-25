import numpy as np
import pandas as pd
import random
from math import sqrt
import sys
import multiprocessing
import time
sys.setrecursionlimit(5000)

# basic helper function

def index_tableToArray(i,j,n) :
    # this function will convert two-dimentional index(i,j) to
    # the corresponding one-dimentional index
    return i * n + j

def index_arrayToTable(k,n) :
    # this function will convert one-dimentional index k into
    # the corresponding two-dimentional index (i,j) in a matrix
    return k/n, k%n

def member_smallSquare(k,n):
    # this function will take the index of the sudoku_array k and return
    # the the index of the 3*3 sub-grid which contains the kth small
    # squre

    i0,j0 = index_arrayToTable(k,n)         # get the index inthe sudoku table
    m = int(sqrt(n))
    i_start = i0/m*m                        # get the index of the 3*3 sub-grid
    j_start = j0/m*m
    res = []
    for i in range(m):                      # get all the elements in the sub-grid
        for j in range(m):
            res.append(index_tableToArray(i_start+i,j_start+j,n))
    return res

def getPositionArr(sudoku_arr,n) :
    # this function will return an array that marks the index of the unfilled grid
    arr = np.array([i == 0 for i in sudoku_arr])
    return arr






# basic functions in Genetic Algorithm Design

def fitness(sudoku_arr,n) :
    # the function will measure a "goodness" of an individual.
    # This function will count the number of repeated or non-present
    # integers in each row and each column.
    # Thanks to the setting, we do not need to check each sub-grid

    # A low fitness value indicates a good individual, a high
    # fitness value indicates a bad individual. If the fitness
    # value is zero, then the individual is the final solution
    val = 0
    m = int(sqrt(n))
    sudoku_mat = sudoku_arr.reshape(n,n)
    for k in range(1,n+1) :
        for i in range(n):
            row = sudoku_mat[i,:]
            col = sudoku_mat[:,i]
            val += abs(sum(row == k) - 1) + abs(sum(col == k) - 1)
    return val




def initialization(sudoku_arr,num_population,n) :
    # this function will set up the initial configuration for the genetic algorithm.
    # It will generate the population, the size of population is specified by the
    # variable num_population. Given the initial status of Sudoku table, we will
    # filled the blanks and make sure that every sub-grid contains all of digits
    # ranging between 1 and 9
    #
    # the function will return the population matrix and the positionArr,
    # the positionArr will record which grids are initially unfilled
    positionArr = getPositionArr(sudoku_arr,n)
    populationMat = np.zeros((num_population,n*n+1))

    m = int(sqrt(n))
    sudoku_array_cp = np.array(sudoku_arr)
    for i in range(m):
        for j in range(m):
            l = np.array([index_tableToArray(m*i+s1,m*j+s2,n) for s1 in range(m) for s2 in range(m)])
            numberLst = sudoku_arr[l]
            k = 1
            for s in l :
                if positionArr[s] :
                    while k in numberLst : k += 1
                    sudoku_array_cp[s] = k
                    k += 1

    fitnessVal = fitness(sudoku_array_cp,n)

    for i in range(num_population) :
        arr = np.array(sudoku_array_cp)
        populationMat[i,:-1] = arr
        populationMat[i,-1] = fitnessVal

    return populationMat,positionArr


def mutation(sudoku_arr,positionArr,n) :
    # the function will randomly choose a 3*3 sub-grid and randomly swap two
    # grids in this sub-grid
    # If the individual satisfies the constraint that each 3*3 sub-grid contains
    # all of digits ranging from 1 to 9, then the mutation process will preserve
    # this property
    m = int(sqrt(n))
    r1 = random.randrange(m)
    r2 = random.randrange(m)
    l = [index_tableToArray(r1*m+s1,r2*m+s2,n) for s1 in range(m) for s2 in range(m)]
    l2 = []
    for i in l:
        if positionArr[i] :
            l2.append(i)
    len_l2 = len(l2)
    if len(l2) > 1 :
        lst_rand = [random.random() for i in range(len(l2))]
        lst_index = np.argsort(lst_rand)
        # swap the first and second term
        t1 = l2[lst_index[0]]
        t2 = l2[lst_index[1]]
        ss = sudoku_arr[t1]
        sudoku_arr[t1] = sudoku_arr[t2]
        sudoku_arr[t2] = ss
    return 0



def crossover(sudoku_arr1,sudoku_arr2,positionArr,n):
    # this function perform the crossover process.
    # suppose sudoku_arr1 = AAA, sudoku_arr2 = BBB
    #                       AAA                BBB
    #                       AAA                BBB
    # after the crossover, we will get two individuals
    # AAA  and BBB
    # BBB      AAA
    # BBB      AAA

    m = int(sqrt(n))
    arr1 = sudoku_arr1[positionArr]
    arr2 = sudoku_arr2[positionArr]
    l = m * n
    p12 = arr1[l:]
    p22 = arr2[l:]
    arr1[l:] = p22
    arr2[l:] = p12
    return 0


def selection(populationMat) :
    # this function will randomly pick up a portion of population
    num_population = populationMat.shape[0]
    lst_rand = np.array([random.random() for i in range(num_population)])
    lst_index = np.argsort(lst_rand)
    selectedLst = lst_index[:int(num_population * 0.6)]
    fitnessValLst = populationMat[selectedLst,-1]
    sortedIndex = np.argsort(fitnessValLst)
    return selectedLst[sortedIndex[0]], selectedLst[sortedIndex[1]]


def notExistente(individual,populationMat) :
    # check if the individaul is already in the poopulation
    res = True
    num_population = populationMat.shape[0]
    for i in range(num_population):
        if  sum(abs(individual - populationMat[i,:-1])) == 0 :
            return False
    return True



def reproduction(populationMat,positionArr,n,level):
    # this function implements the reproduction process in a standard genetic algorithm
    # there are 4 steps:
    #   1. selection
    #   2. mutation
    #   3. crossover
    #   4. add the new individual to population or or not
    num_population, size = populationMat.shape
    # selection
    individualIndex_1, individualIndex_2 = selection(populationMat)
    individual_1 = np.array(populationMat[individualIndex_1,:-1])
    individual_2 = np.array(populationMat[individualIndex_2,:-1])
    # mutation
    r1 = random.random()
    r2 = random.random()
    if r1 < level :
        mutation(individual_1,positionArr,n)
    if r2 < level :
        mutation(individual_2,positionArr,n)
    # crossover
    crossover(individual_1,individual_2,positionArr,n)
    fitness_1 = fitness(individual_1,n)
    fitness_2 = fitness(individual_2,n)
    # evolution
    fitness_population = populationMat[:,-1]
    if fitness_1 < max(fitness_population) and notExistente(individual_1,populationMat):
        maxIndexArr, = np.where(fitness_population == max(fitness_population))
        maxIndex = maxIndexArr[0]
        populationMat[maxIndex,:-1] = individual_1
        populationMat[maxIndex,-1] = fitness_1
    if fitness_2 < max(fitness_population) and notExistente(individual_2,populationMat):
        maxIndexArr, = np.where(fitness_population == max(fitness_population))
        maxIndex = maxIndexArr[0]
        populationMat[maxIndex,:-1] = individual_2
        populationMat[maxIndex,-1] = fitness_2
    return 0




def GA(populationMat,positionArr,num_generation,num_population,mutation_rate) :
    k = 0
    while not min(populationMat[:,-1]) == 0 :
        #print("%d-th generation, fitness : %d, average fitness : %f" % (k, min(populationMat[:,-1]),populationMat[:,-1].mean()))
        reproduction(populationMat,positionArr,n,min(mutation_rate + 0.00005*(k%100000) , 0.9))
        k += 1

    if k < num_generation * num_population:
        #print "optimal generation found!"
        l = populationMat[:,-1]
        arr, = np.where(l==min(l))
        solutionArr = populationMat[arr[0],:-1].astype(int)

        #print_sudokuArr(solutionArr,n)
        sudoku_matrix0 = pd.DataFrame(solutionArr.reshape(n,n))
        sudoku_matrix0.to_csv("output.csv",sep=",",header=None,index=False)
        return True
    else:
        return False



if __name__ == "__main__" :
    df = pd.read_csv("input_1.csv",sep=",",header=None)
    sudoku_array = np.asarray(df).reshape(-1)
    n=9

    # parameter setting
    num_generation = 120
    num_population = 80
    populationMat = np.zeros((num_population,n*n+1))
    mutation_rate = 0.4
    populationMat,positionArr = initialization(sudoku_array,num_population,n)

    # run genetic algorithm
    #GA(populationMat,positionArr,num_generation,num_population,mutation_rate)
    p = multiprocessing.Process(target=GA,name="GA", args=(populationMat,positionArr,num_generation,num_population,mutation_rate))

    p.start()
    p.join(40)
    if p.is_alive() :
        print "Oooh! this puzzle is too hard for our algorithm, please give more hints and try again."
        p.terminate()
        p.join()
    p.join()
