
# Script file : simpleBacktrackingAlgo.py

# In this script, we will implement a sudoku solver.
# The default size of the table is 9 * 9

# the input is a csv file. We use 0 to represent blanks.
# the output is also a csv file, which report the solution.


# the algorithm implemented in the problem is a straight forward
# recursive search algorithm. Basically, we will try every possibility
# for every blank.

# To make the algorithm efficient, when we perform the search we will add
# some constraints so that we can reduce the space of the possible solutions.
# The constraints are the direct translation of rules in a sudoku puzzle.

# We can further improve the efficiency of the algorithm by following some
# special order of search. In this algorithm, we will use a heuristic approach
# to determine the order of search. Given a sudoku puzzle table, we will always
# start to fill the blanks with minimum possibilities.



import numpy as np
import pandas as pd
import random
from math import sqrt
import sys
import multiprocessing
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



def simpleBacktrackingAlgo(sudoku_array,n) :
    # This function will search for the solution of the sudoku puzzle.
    # the sudoku_array represents the hints of the puzzle.

    # initialization : get the unfilled_lst and compute the record matrix
    record_mat = np.zeros((n*n,n+1))
    unfilled_lst = np.array(range(n*n))[sudoku_array==0]                    # get the index of the blanks

    if len(unfilled_lst) == 0 :                                             # in this case, we have fund a solution
        sudoku_matrix0 = pd.DataFrame(sudoku_array.reshape(n,n))
        sudoku_matrix0.to_csv("output.csv",sep=",",header=None,index=False)
        return True

    else :

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


            numberChoicesLst_unfilled = numberChoicesLst[unfilled_lst]
            minIndex_arr, = np.where(numberChoicesLst_unfilled==min(numberChoicesLst_unfilled))
            minIndex = minIndex_arr[0]



            # go through all the possibility for this blank
            start_index = unfilled_lst[minIndex]
            choiceLst = (np.array(range(n+1)))[record_mat[start_index,:]==0]
            occurrenceLst = [sum(sudoku_array == x) for x in choiceLst]
            choiceLst = choiceLst[np.argsort(occurrenceLst)][::-1]

            for k in choiceLst :
                record_mat[start_index,0] -= 1                 # update the number of possible choices
                record_mat[start_index,k] = 1                  # mark that we have tried k
                sudoku_array[start_index] = k                  # fill k into the blank
                # call recurseively the function
                isSolved = simpleBacktrackingAlgo(sudoku_array,n)
                if isSolved :
                    return True

            sudoku_array[start_index] = 0
            return False                                        # failed to find a solution



if __name__ == "__main__" :
    df = pd.read_csv("input_1.csv",sep=",",header=None)
    sudoku_array = np.asarray(df).reshape(-1)
    n=9

    #simpleBacktrackingAlgo(sudoku_array,n)
    p = multiprocessing.Process(target=simpleBacktrackingAlgo,name="sba",args=(sudoku_array,n))

    p.start()
    p.join(40)
    if p.is_alive() :
        print "Oooh! this puzzle is too hard for our algorithm, please give more hints and try again."
        p.terminate()
        p.join()
    p.join()


