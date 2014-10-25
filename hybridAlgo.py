

# Attention :
# ^^^^^^^^^
# This algorithm does not guarantee the termination of program in a reasonable time.
#                ^^^^^^^^




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

def getConstraintMemberLst(k,n):
    # this function will return a list of index which are in the same row, column or
    # sub-grid as the grid k
    constraintMemberLst = member_smallSquare(k,n)
    i0 = k/n
    j0 = k%n
    for s in range(n) :
        constraintMemberLst.append(index_tableToArray(i0,s,n))
        constraintMemberLst.append(index_tableToArray(s,j0,n))
    return constraintMemberLst


def getConstraintLinkMat(n):
    # this function will return a matrix that represents the constraint-member relationships
    # between two different index
    constraintLinkMat = np.zeros((n*n,n*n))
    for i in range(n*n) :
        constraintMemberlst_i = getConstraintMemberLst(i,n)
        for j in constraintMemberlst_i :
            constraintLinkMat[i,j] = 1
        constraintLinkMat[i,i] = 0
    return constraintLinkMat


def getPositionArr(sudoku_arr,n) :
    # this function will return an array that marks the index of the unfilled grid
    arr = np.array([i == 0 for i in sudoku_arr])
    return arr


def print_sudokuArr(sudoku_array,n):
    # this function will print out the sudoku solution
    # in matrix form
    print("-----------------------")
    for i in range(n) :
        lst = []
        for j in range(n):
            lst.append(str(int(sudoku_array[i*n+j])))
        print(','.join(lst))




# basic functions in Genetic Algorithm Design

def fitness(sudoku_arr,n) :
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


def addToPopulation(sudoku_array,populationMat,countLst,n):

    arr_cp = np.array(sudoku_array)
    unfilledLst = (arr_cp == 0)
    m = int(sqrt(n))
    # complete the sudoku_array
    for i in range(m):
        for j in range(m):
            l = np.array([index_tableToArray(m*i+s1,m*j+s2,n) for s1 in range(m) for s2 in range(m)])
            numberLst = arr_cp[l]
            k = 1
            for s in l :
                if unfilledLst[s] :
                    while k in numberLst : k += 1
                    arr_cp[s] = k
                    k += 1

    fitnessVal = fitness(arr_cp,n)
    populationMat[countLst[0],:-1] = arr_cp
    populationMat[countLst[0],-1] = fitnessVal
    countLst[0] += 1
    return 0



def GA(populationMat,positionArr,num_generation,num_population,mutation_rate) :
    k = 0
    while not min(populationMat[:,-1]) == 0 and k < num_generation * num_population:
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





def getOptimumIndex(userArr,constraintLinkMat) :
    C = constraintLinkMat[userArr,:][:,userArr]
    R = np.zeros(len(userArr))
    for i in range(len(userArr)) :
        linkArr, = np.where(C[i,:] == 1)
        R[i] = len(linkArr) + sum(C[linkArr,:][:,linkArr].reshape(-1)) / 2

    maxIndex_arr, = np.where(R == max(R))
    return userArr[maxIndex_arr[0]]








def process(sudoku_array,n,path,critical_ratio,constraintLinkMat,populationMat,positionArr,countLst,num_generation,num_population,mutation_rate) :
    # This function will search for the solution of the sudoku puzzle.
    # the sudoku_array represents the hints of the puzzle.
    # print_sudokuArr(sudoku_array,n)
    #print("p : %d, r : %d" % (len(path),sum(sudoku_array==0)))

    # initialization : get the unfilled_lst and compute the record matrix
    record_mat = np.zeros((n*n,n+1))
    unfilled_lst = np.array(range(n*n))[sudoku_array==0]                    # get the index of the blanks

    if len(unfilled_lst) == 0 :                                             # in this case, we have fund a solution
        sudoku_matrix0 = pd.DataFrame(sudoku_array.reshape(n,n))
        sudoku_matrix0.to_csv("output.csv",sep=",",header=None,index=False)
        return True

    else :

        sudoku_matrix = sudoku_array.reshape(n,n)
        for i in range(n) :
            for k in range(1,n+1):
                if sum(sudoku_matrix[i,:] == k) == 0  :
                    testCol = [sum(sudoku_matrix[:,j] == k) for j in range(n)]
                    availableCol = np.array([testCol[j]==0 and sudoku_matrix[i,j]== 0 and record_mat[index_tableToArray(i,j,n),k]==0 for j in range(n)])
                    if sum(availableCol) == 0 :

                        if countLst[0] > num_population - 1:
                            if random.random() < 0.25:
                                isSolved = GA(populationMat,positionArr,num_generation,num_population,mutation_rate)
                                if isSolved :
                                    return True
                            countLst[0] = max(countLst[0]*2/3, random.randrange(0,countLst[0]))

                        # add the failed case to the populationMat
                        addToPopulation(sudoku_array,populationMat,countLst,n)
                        r = random.random()
                        if r < 0.5 * critical_ratio  and r > 0.001:
                            sudoku_array_cp = np.array(sudoku_array)
                            l = max(int(len(path)* 0.6 * critical_ratio),1)
                            sudoku_array_cp[path[:l]]=0
                            path_cp = list(path[l:])


                            isSolved = process(sudoku_array_cp,n,path_cp,0.5*critical_ratio,constraintLinkMat,populationMat,positionArr,countLst,num_generation,num_population,mutation_rate)


                            if isSolved :
                                return True

                        return False

        # compute the record_mat
        for k in unfilled_lst :
            constraintMemberLst = member_smallSquare(k,n)                   # get the index of elements in sub-grid
            i0 = k/n
            j0 = k%n
            for  s in range(n) :
                constraintMemberLst.append(index_tableToArray(i0,s,n))      # get the index of elements that are in
                constraintMemberLst.append(index_tableToArray(s,j0,n))      # the same row or column as k

            for j in constraintMemberLst :                                  # add constraints to the record_mat
                if (not j == k) and (sudoku_array[j] > 0) :
                    record_mat[k,sudoku_array[j]] = 1

        for k in unfilled_lst :                                             # compute the number of possible choices
            record_mat[k,0] = n - sum(record_mat[k,:])                      # for blanks

        numberChoicesLst = record_mat[:,0]                                  # update the unfilled_lst
        unfilled_lst = np.array(range(n*n))[numberChoicesLst>0]

        if len(unfilled_lst) == 0 :                                         # in this case, there is no solution
            return False                                                    # because we still have blanks but we do
                                                                            # not have available number to fill them.

        else :

            # find out the blank with minimum possible choices.
            numberChoicesLst_unfilled = numberChoicesLst[unfilled_lst]
            minIndex_arr, = np.where(numberChoicesLst_unfilled==min(numberChoicesLst_unfilled))
            optIndex = getOptimumIndex(unfilled_lst[minIndex_arr],constraintLinkMat)

            # go through all the possibility for this blank
            start_index = optIndex
            path.append(start_index)
            threshold = sum(sudoku_array == 0) / float(n*n)
            r = random.random()
            if abs(r-0.5) > abs(threshold-0.5) :
                choiceLst = (np.array(range(n+1)))[record_mat[start_index,:]==0]
                occurrenceLst = [sum(sudoku_array == x) for x in choiceLst]
                choiceLst = choiceLst[np.argsort(occurrenceLst)][::-1]

            elif abs(r-0.5) < abs(threshold-0.5)-0.035 :
                choiceLst = (np.array(range(n+1)))[record_mat[start_index,:]==0]
                occurrenceLst = [sum(sudoku_array == x) for x in choiceLst]
                choiceLst = choiceLst[np.argsort(occurrenceLst)]
            else :
                choiceLst = (np.array(range(n+1)))[record_mat[start_index,:]==0]

            for k in choiceLst :
                record_mat[start_index,0] -= 1                 # update the number of possible choices
                record_mat[start_index,k] = 1                  # mark that we have tried k
                sudoku_array[start_index] = k                  # fill k into the blank
                # call recurseively the function
                isSolved = process(sudoku_array,n,path,critical_ratio,constraintLinkMat,populationMat,positionArr,countLst,num_generation,num_population,mutation_rate)
                if isSolved :
                    return True

            sudoku_array[start_index] = 0
            path.pop()
            return False                                        # failed to find a solution








### main ###

if __name__ == "__main__" :
    df = pd.read_csv("input_4.csv",sep=",",header=None)
    sudoku_array = np.asarray(df).reshape(-1)
    n=9

    # backtracking algorithm setting
    path = []
    constraintLinkMat = getConstraintLinkMat(n)

    # genetic algorithm setting
    countLst = np.array([0])
    num_generation = 120
    num_population = 80
    populationMat = np.zeros((num_population,n*n+1))
    mutation_rate = 0.4
    populationMat,positionArr = initialization(sudoku_array,num_population,n)

    #process(sudoku_array,n,path,1,constraintLinkMat,populationMat,positionArr,countLst,num_generation,num_population,mutation_rate)

    p = multiprocessing.Process(target=process,name="process", args=(sudoku_array,n,path,1,constraintLinkMat,populationMat,positionArr,countLst,num_generation,num_population,mutation_rate))

    p.start()
    p.join(40)
    if p.is_alive() :
        print "Oooh! this puzzle is too hard for our algorithm, please give more hints and try again."
        p.terminate()
        p.join()
    p.join()










