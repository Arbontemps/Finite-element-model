""" 
author: Arthur Bontemps
finite element function for composite floor model

"""

import numpy as np
import scipy.integrate as integrate


#%%
""" Form function and relation matrices """
def matrices_N(x, l):
    """ N is 3*6 matrix for 3 dof/nodes and EB hypothesis 
    x is the position in the element and l the length of the element"""

    N = np.array([[1-x/l, 0, 0, x/l, 0, 0], \
                  [0, 1 - 3*x**2/l**2 + 2*x**3/l**3, x - 2*x**2/l + x**3/l**2, 0, 3*x**2/l**2 - 2*x**3/l**3,  -x**2/l + x**3/l**2], \
                  [0, -6*x/l**2 + 6*x**2/l**3, 1 - 4*x/l + 3*x**2/l**2, 0, 6*x/l**2 - 6*x**2/l**3, -2*x/l + 3*x**2/l**2]])
    
    dN = np.array([[-1/l, 0, 0, 1/l, 0, 0], \
                  [0, -6*x/l**2 + 6*x**2/l**3, 1 - 4*x/l + 3*x**2/l**2, 0, 6*x/l**2 - 6*x**2/l**3,  -2*x/l + 3*x**2/l**2], \
                  [0, -6/l**2 + 12*x/l**3, -4/l + 6*x/l**2, 0, 6/l**2 - 12*x/l**3, -2/l + 6*x/l**2]])
    return N, dN

def matrices_N2e(x, l):
    """ N is 3*6 matrix for 3 dof/nodes and EB hypothesis 
    x is the position in the element and l the length of the element"""

    N = np.array([[1-x/l, 0, 0, x/l, 0, 0, 0, 0], \
                  [0, 1 - 3*x**2/l**2 + 2*x**3/l**3, x - 2*x**2/l + x**3/l**2, 0, 3*x**2/l**2 - 2*x**3/l**3,  -x**2/l + x**3/l**2, 0, 0], \
                  [0, -6*x/l**2 + 6*x**2/l**3, 1 - 4*x/l + 3*x**2/l**2, 0, 6*x/l**2 - 6*x**2/l**3, -2*x/l + 3*x**2/l**2, 0, 0], \
                  [0, 0, 0, 0, 0, 0, 1-x/l, x/l]])
    
    dN = np.array([[-1/l, 0, 0, 1/l, 0, 0, 0, 0], \
                  [0, -6*x/l**2 + 6*x**2/l**3, 1 - 4*x/l + 3*x**2/l**2, 0, 6*x/l**2 - 6*x**2/l**3,  -2*x/l + 3*x**2/l**2, 0, 0], \
                  [0, -6/l**2 + 12*x/l**3, -4/l + 6*x/l**2, 0, 6/l**2 - 12*x/l**3, -2/l + 6*x/l**2, -1/l, 1/l]])
    return N, dN


def matrix_a_s(y):
    """ a_s is a 2*3 matrix that relates u(x,y) = u0(x) - y*beta(x) and v(x,y) = v0(x) 
    y is the ordinate position into the section"""
    a_s = np.array([[1, 0, -y], \
                    [0, 1, 0]])
    return a_s

def matrix_dkeij(y, x, i, j, l, E):
    """ elementary matrix element that has to be integrated to have kelem 
    x and y position, i and j the indices to retrieve a value from the matrix, l = l_elem 
    and E elastic modulus of the element """
    dke = E * matrices_N(x, l)[1].T @ matrix_a_s(y).T @ np.array([[1, 0], [0, 0]]) @ np.array([[1, 0], [0, 0]]) @ matrix_a_s(y) @ matrices_N(x, l)[1]
    return dke[i][j]

def U0(x, l, u):
    U0 = matrices_N(x, l)[0] @ u
    return U0

def node_number(list, number):
    return np.where(list == min(list, key=lambda x:abs(x-number)))[0][0]

#%% 
""" Elementary stiffness matrix"""
# Calculate your own kelem 
def matrix_ke(l, E, h, e, ndof):
    ke = np.zeros((ndof, ndof))
    for i, k in enumerate(ke):
        for j, dk in enumerate(k):
            ke[i][j] = e * integrate.dblquad(matrix_dkeij, 0,  l, lambda x: -h/2, lambda x:h/2, args=(i, j, l, E))[0]
    return ke

def true_Ke_matrix(E, S, I, l):
    """ Common stiffness matrix for a 1D-element with 2 nodes and 3dof/node"""
    true_Ke =  np.array([[E*S/l, 0, 0, -E*S/l, 0, 0], \
                        [0, 12*E*I/l**3, 6*E*I/l**2, 0, -12*E*I/l**3, 6*E*I/l**2], \
                        [0, 6*E*I/l**2, 4*E*I/l, 0, -6*E*I/l**2, 2*E*I/l], \
                        [-E*S/l, 0, 0, E*S/l, 0, 0], \
                        [0, -12*E*I/l**3, -6*E*I/l**2, 0, 12*E*I/l**3, -6*E*I/l**2], \
                        [0, 6*E*I/l**2, 2*E*I/l, 0, -6*E*I/l**2, 4*E*I/l]])
    return true_Ke

def matrix_Kr(k, e):
    """Stiffness matrix of the spring, take excentricity and spring's stiffness as input """ 
    # pi is a 4*6 matrix relating the 4 dof of the spring ends with the 6 dof of each node
    pi = np.array([[1, 0, e, 0, 0, 0],\
                [0, 1, 0, 0, 0, 0],\
                [0, 0, 0, 1, 0, e], \
                [0, 0, 0, 0, 1, 0]])
    a = np.array([[-1, 0, 1, 0]]) # 1*4 vector relating relative displacement with displacements of the spring ends, hence Urel = a.pi.U
    Kr = k * pi.T @ a.T @ a @ pi
    return Kr

def matrix_K2e(Eb, hb, eb, Et, ht, et, l):
    """Stiffness matrix of a 1D-element with 4 nodes defining 2 layer with 
    3 dof/bottom node (u,v,theta) and 1 dof/top node (u) as they share same v and theta 
    The function first calculate a 12*12 matrix using 3 dof/node (bot and top) and then transform it
    into a 8*8 matrix as the element has in total 8 dof
    Eb, Sb, Ib are the properties of the bot layer and Eh, Sh, Ih the top layer
    l is the length of the element"""
    P = np.array([[1, 0, 0, 0, 0, 0, 0, 0],\
                  [0, 1, 0, 0, 0, 0, 0, 0], \
                  [0, 0, 1, 0, 0, 0, 0, 0], \
                  [0, 0, 0, 1, 0, 0, 0, 0], \
                  [0, 0, 0, 0, 1, 0, 0, 0], \
                  [0, 0, 0, 0, 0, 1, 0, 0], \
                  [0, 0, 0, 0, 0, 0, 1, 0], \
                  [0, 1, 0, 0, 0, 0, 0, 0], \
                  [0, 0, 1, 0, 0, 0, 0, 0], \
                  [0, 0, 0, 0, 0, 0, 0, 1], \
                  [0, 0, 0, 0, 1, 0, 0, 0], \
                  [0, 0, 0, 0, 0, 1, 0, 0]])
    Ketot = np.zeros((12,12))
    Ketot[0:6,0:6], Ketot[-6:,-6:] = matrix_ke(l, Eb, hb, eb, 6), matrix_ke(l, Et, ht, et, 6)
    K2e = P.T @ Ketot @ P
    return K2e

def matrix_Kc(kc, hb, ht, xi, l_el):
    """ Shear stiffness of the connector between the 2 layers. kc is the scalar value in N/mm """
    s = np.array([[1, 0, (hb+ht)/2, -1]])
    Kc = kc * matrices_N2e(xi, l_el)[0].T @ s.T @ s @ matrices_N2e(xi, l_el)[0]
    return Kc