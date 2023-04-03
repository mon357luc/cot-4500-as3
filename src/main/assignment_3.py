<<<<<<< HEAD
=======
"Assignment 3"

import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

func : str = "t - y**2"
bounds = (0,2)
iterations = 10
initial_condition = (0,1)
gauss_matrix = np.array([[2,-1,1,6],[1,3,1,0],[-1,5,4,-3]])
LU_matrix = np.array([[1,1,0,3],[2,1,-1,1],[3,-1,-1,2],[-1,2,3,-1]])
dominant_matrix = np.array([[9,0,5,2,1],[3,9,1,2,1],[0,1,7,2,3],[4,2,3,12,2],[3,2,4,0,8]])
pos_def_matrix = np.array([[2,2,1],[2,3,0],[1,0,2]])

def eulers_method(e_func : str, rng, steps : int, initial_vals):
    """Performs Euler's method on a given function given a range (on the dependent variable),
    number fo iterations, and inital point"""
    
    t = initial_vals[0]
    y = initial_vals[1]
    step = (rng[1] - rng[0]) / steps
    dydt = eval(e_func)

    for i in range(steps):
        y = y + dydt * step
        t = t + step
        dydt = eval(e_func)

    return y

def runge_kutta_method(e_func : str, rng, steps : int, initial_vals):
    """Performs Euler's method on a given function given a range (on the dependent variable),
    number fo iterations, and inital point"""
    
    t = initial_vals[0]
    w = initial_vals[1]
    y = initial_vals[1]
    step = (rng[1] - rng[0]) / steps
    k_1 = step * eval(e_func)
    t = t + step  / 2
    y = w + k_1  / 2
    k_2 = step * eval(e_func)
    y = w + k_2 / 2
    k_3 = step * eval(e_func)
    t = t + step  / 2
    y = w + k_3
    k_4 = step * eval(e_func)

    for i in range(steps):
        w = w + (1/6) * (k_1 + 2 * (k_2 + k_3) + k_4)
        y = w
        k_1 = step * eval(e_func)
        t = t + step  / 2
        y = w + k_1  / 2
        k_2 = step * eval(e_func)
        y = w + k_2 / 2
        k_3 = step * eval(e_func)
        t = t + step  / 2
        y = w + k_3
        k_4 = step * eval(e_func)

    return w

def gaussian_elimination(original_matrix : np.array):
    row_does_not_exists = 1
    ex_c = 0
    matrix = original_matrix.copy()
    pivots = np.zeros((matrix.shape[0],2), dbtype:= int)
    for j in range(0, matrix.shape[1]-1):
        pivot_row = j - ex_c
        if matrix[(pivot_row,j)] == 0:
            for i in range(pivot_row + 1,matrix.shape[0]):
                if matrix[(i,j)] != 0:
                    matrix[[pivot_row, i], :] = matrix[[i, pivot_row], :]
                    row_does_not_exists = 0
                    break
            if row_does_not_exists:
                ex_c = ex_c + 1
                pivots[[j]] = [-1,-1]
                continue
        pivots[[j]] = [pivot_row, j]
        for i in range(pivot_row + 1,matrix.shape[0]):
            if matrix[(i, j)] == 0:
                continue
            matrix[[i]] = matrix[[i]] - \
                (matrix[(i, j)] / matrix[(pivot_row, j)]) * matrix[[pivot_row]]
    for j in range(matrix.shape[0] - 1, -1, -1):
        if pivots[j][0] == -1:
            continue
        for i in range(pivots[j][0] - 1, -1, -1):
            matrix[[i]] = matrix[[i]] - \
                (matrix[(i, j)] / matrix[(pivots[j][0], pivots[j][1])]) * matrix[[j]]
        matrix[[j]] = (1 / matrix[(pivots[j][0], pivots[j][1])]) * matrix[[j]]
    return matrix.T[matrix.shape[1] - 1]

def determinant(matrix):
    det = matrix[(0,0)]
    for i in range(1, matrix.shape[0]):
        det = det * matrix[(i,i)]
    return det

def LU_factorization(original_matrix : np.array):
    row_does_not_exists = 1
    ex_c = 0
    L_matrix = original_matrix.copy()
    U_matrix = np.identity(L_matrix.shape[0])
    pivots = np.zeros((L_matrix.shape[0],2), dbtype:= int)
    for j in range(0, L_matrix.shape[1]-1):
        pivot_row = j - ex_c
        if L_matrix[(pivot_row,j)] == 0:
            for i in range(pivot_row + 1,L_matrix.shape[0]):
                if L_matrix[(i,j)] != 0:
                    L_matrix[[pivot_row, i], :] = L_matrix[[i, pivot_row], :]
                    row_does_not_exists = 0
                    break
            if row_does_not_exists:
                ex_c = ex_c + 1
                pivots[[j]] = [-1,-1]
                continue
        pivots[[j]] = [pivot_row, j]
        for i in range(pivot_row + 1,L_matrix.shape[0]):
            if L_matrix[(i, j)] == 0:
                continue
            U_matrix[(i,j)] = L_matrix[(i, j)] / L_matrix[(pivot_row, j)]
            L_matrix[[i]] = L_matrix[[i]] - \
                (L_matrix[(i, j)] / L_matrix[(pivot_row, j)]) * L_matrix[[pivot_row]]
    print("%.5f" % determinant(L_matrix), end:="\n\n")
    print(L_matrix, "\n\n", U_matrix, end:="\n\n")

def diagonally_dominant_test(matrix):
    test = True
    for i in range(matrix.shape[0]):
        row_total = 0
        for j in range(matrix.shape[1]):
            if i == j:
                continue
            row_total = row_total + abs(matrix[(i, j)])
        if matrix[(i,i)] < row_total:
            test = False
    return test

def pos_def_test(matrix):
    if np.all(matrix - matrix.T) == 0:
        if np.all(np.linalg.eigvals(matrix)) > 0:
            return True
    return False

print("%.5f" % eulers_method(func, bounds, iterations, initial_condition), end:="\n\n")
print("%.5f" % runge_kutta_method(func, bounds, iterations, initial_condition), end:="\n\n")
print(gaussian_elimination(gauss_matrix), end:="\n\n")
LU_factorization(LU_matrix)
print(diagonally_dominant_test(dominant_matrix), end:="\n\n")
print(pos_def_test(pos_def_matrix))
>>>>>>> 43765e7 (	modified:   src/main/assignment_3.py)
