""" 
author: Arthur Bontemps
finite element function for composite floor model

"""

import numpy as np
import scipy.integrate as integrate


#%%
""" Form function and relation matrices """
def matrices_N(x, l):
    """ N is 3*6 matrix for 3 dof/nodes and EB hypothesis for 1 layer beam element
    x is the position in the element and l the length of the element"""

    N = np.array([[1-x/l, 0, 0, x/l, 0, 0], \
                  [0, 1 - 3*x**2/l**2 + 2*x**3/l**3, x - 2*x**2/l + x**3/l**2, 0, 3*x**2/l**2 - 2*x**3/l**3,  -x**2/l + x**3/l**2], \
                  [0, -6*x/l**2 + 6*x**2/l**3, 1 - 4*x/l + 3*x**2/l**2, 0, 6*x/l**2 - 6*x**2/l**3, -2*x/l + 3*x**2/l**2]])
    
    dN = np.array([[-1/l, 0, 0, 1/l, 0, 0], \
                  [0, -6*x/l**2 + 6*x**2/l**3, 1 - 4*x/l + 3*x**2/l**2, 0, 6*x/l**2 - 6*x**2/l**3,  -2*x/l + 3*x**2/l**2], \
                  [0, -6/l**2 + 12*x/l**3, -4/l + 6*x/l**2, 0, 6/l**2 - 12*x/l**3, -2/l + 6*x/l**2]])
    return N, dN

def matrices_N2e(x, l):
    """ N is 6*12 matrix for 3 dof/nodes and EB hypothesis for 2-layers beam element
    x is the position in the element and l the length of the element"""

    N = np.array([[1-x/l, 0, 0, x/l, 0, 0, 0, 0, 0, 0, 0, 0], \
                  [0, 1 - 3*x**2/l**2 + 2*x**3/l**3, x - 2*x**2/l + x**3/l**2, 0, 3*x**2/l**2 - 2*x**3/l**3,  -x**2/l + x**3/l**2, 0, 0, 0, 0, 0, 0], \
                  [0, -6*x/l**2 + 6*x**2/l**3, 1 - 4*x/l + 3*x**2/l**2, 0, 6*x/l**2 - 6*x**2/l**3, -2*x/l + 3*x**2/l**2, 0, 0, 0, 0, 0, 0], \
                  [0, 0, 0, 0, 0, 0, 1-x/l, 0, 0, x/l, 0, 0], \
                  [0, 0, 0, 0, 0, 0, 0, 1 - 3*x**2/l**2 + 2*x**3/l**3, x - 2*x**2/l + x**3/l**2, 0, 3*x**2/l**2 - 2*x**3/l**3,  -x**2/l + x**3/l**2], \
                  [0, 0, 0, 0, 0, 0, 0, -6*x/l**2 + 6*x**2/l**3, 1 - 4*x/l + 3*x**2/l**2, 0, 6*x/l**2 - 6*x**2/l**3, -2*x/l + 3*x**2/l**2]])
    
    dN = np.array([[-1/l, 0, 0, 1/l, 0, 0, 0, 0, 0, 0, 0, 0], \
                  [0, -6*x/l**2 + 6*x**2/l**3, 1 - 4*x/l + 3*x**2/l**2, 0, 6*x/l**2 - 6*x**2/l**3,  -2*x/l + 3*x**2/l**2, 0, 0, 0, 0, 0, 0], \
                  [0, -6/l**2 + 12*x/l**3, -4/l + 6*x/l**2, 0, 6/l**2 - 12*x/l**3, -2/l + 6*x/l**2, 0, 0, 0, 0, 0, 0],  \
                  [0, 0, 0, 0, 0, 0, -1/l, 0, 0, 1/l, 0, 0], \
                  [0, 0, 0, 0, 0, 0, 0, -6*x/l**2 + 6*x**2/l**3, 1 - 4*x/l + 3*x**2/l**2, 0, 6*x/l**2 - 6*x**2/l**3,  -2*x/l + 3*x**2/l**2], \
                  [0, 0, 0, 0, 0, 0, 0, -6/l**2 + 12*x/l**3, -4/l + 6*x/l**2, 0, 6/l**2 - 12*x/l**3, -2/l + 6*x/l**2]])
    return N, dN


def matrix_a_s(y):
    """ a_s is a 2*3 matrix that relates u(x,y) = u0(x) - y*beta(x) and v(x,y) = v0(x) 
    y is the ordinate position into the section"""
    a_s = np.array([[1, 0, -y], \
                    [0, 1, 0]])
    return a_s

def matrix_a_s_i(y, layer):
    """ a_s is a 2*6 matrix that relates ub,t(x,y) = u0b,t(x) - yb,t*beta(x) and v(x,y) = v0(x) for the bot or top layer element 
    y is the ordinate position into the section"""
    if layer=='bot':
        a_s = np.array([[1, 0, -y, 0, 0, 0], \
                        [0, 1, 0, 0, 0, 0]])
    elif layer=='top':
        a_s = np.array([[0, 0, 0, 1, 0, -y], \
                    [0, 0, 0, 0, 1, 0]])
    else:
        print('specify the layer (bot or top)')
    return a_s

def U0(x, l, u):
    U0 = matrices_N(x, l)[0] @ u
    return U0

def U0_bilayer(x,l,u):
    U0bi = matrices_N2e(x,l)[0] @ u
    return U0bi


def node_number(list, number):
    return np.where(list == min(list, key=lambda x:abs(x-number)))[0][0]

#%% 
""" Elementary stiffness matrix"""

def matrix_dkeij(y, x, i, j, l, E):
    """ elementary matrix element that has to be integrated to have kelem 
    x and y position, i and j the indices to retrieve a value from the matrix, l = l_elem 
    and E elastic modulus of the element """
    # add np.array([[1,0],[0,0]]) in the calculation to represent the D.T @ D (although the derivative is then in N)
    dke = E * matrices_N(x, l)[1].T @ matrix_a_s(y).T @ np.array([[1,0],[0,0]]) @ \
            np.array([[1,0],[0,0]]) @ matrix_a_s(y) @ matrices_N(x, l)[1]
    return dke[i][j]

def matrix_dk2eij(y, x, i, j, l, E, layer):
    """ elementary matrix element that has to be integrated to have kelem for 2-layers beam
    x and y position, i and j the indices to retrieve a value from the matrix, l = l_elem 
    and E elastic modulus of the element """
    # add np.array([[1,0],[0,0]]) in the calculation to represent the D.T @ D (although the derivative is then in N)
    dk2e = E * matrices_N2e(x, l)[1].T @ matrix_a_s_i(y, layer).T @ np.array([[1,0],[0,0]]) @ \
          np.array([[1,0],[0,0]]) @ matrix_a_s_i(y, layer) @ matrices_N2e(x, l)[1]
    return dk2e[i][j]

def matrix_ke(l, E, h, e, eldof):
    ke = np.zeros((eldof, eldof))
    for i, k in enumerate(ke):
        for j, dk in enumerate(k):
            ke[i][j] = e * integrate.dblquad(matrix_dkeij, 0,  l, lambda x: -h/2, lambda x:h/2, args=(i, j, l, E))[0]
    return ke

def true_Ke_matrix(E, S, I, l):
    """ Common stiffness matrix for a 1D-element with 2 nodes and 3 dof/node"""
    true_Ke =  np.array([[E*S/l, 0, 0, -E*S/l, 0, 0], \
                        [0, 12*E*I/l**3, 6*E*I/l**2, 0, -12*E*I/l**3, 6*E*I/l**2], \
                        [0, 6*E*I/l**2, 4*E*I/l, 0, -6*E*I/l**2, 2*E*I/l], \
                        [-E*S/l, 0, 0, E*S/l, 0, 0], \
                        [0, -12*E*I/l**3, -6*E*I/l**2, 0, 12*E*I/l**3, -6*E*I/l**2], \
                        [0, 6*E*I/l**2, 2*E*I/l, 0, -6*E*I/l**2, 4*E*I/l]])
    return true_Ke

def true_K2e_matrix(Eb, Sb, Ib, Et, St, It, l, eldof):
    """ Common stiffness matrix for a 1D bilayer element with 4 nodes and 3 dof/node"""
    k2e = np.zeros((eldof,eldof))
    k2e[:6,:6], k2e[:6,6:], k2e[6:,:6], k2e[6:,6:] = \
        true_Ke_matrix(Eb, Sb, Ib, l), np.zeros((6,6)), np.zeros((6,6)), true_Ke_matrix(Et, St, It, l)
    return k2e


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

def matrix_P():
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
    return P

def matrix_K2e(Eb, hb, eb, Et, ht, et, l, eldof):
    """Stiffness matrix of a 1D-element with 4 nodes defining 2 layers with 
    3 dof/node (u,v,theta) 
    The function calculate a 12*12 matrix using 3 dof/node
    Eb, Sb, hb are the properties of the bot layer and Et, St, ht the top layer
    l is the length of the element"""
    K2e = np.zeros((eldof, eldof))
    for i, k in enumerate(K2e):
        for j, dk in enumerate(k):
            keb = eb * integrate.dblquad(matrix_dk2eij, 0, l, lambda x:-hb/2, lambda x:hb/2, args=(i, j, l, Eb, 'bot'))[0]
            ket = et * integrate.dblquad(matrix_dk2eij, 0, l, lambda x:-ht/2, lambda x:ht/2, args=(i, j, l, Et, 'top'))[0]
            K2e[i][j] = keb + ket
    return K2e

def dkinter(x, i, j, kc, hb, ht, l):
    """ elementary interface stiffness matrix that has to be integrated to have Kinter 
    x position, i and j indices to retrieve a value from the matrix, l = length of the interface 
    hb and ht the bot and top layer's height """
    s = np.array([[1, 0, -(hb+ht)/2, -1, 0, 0]])
    dki = kc * matrices_N2e(x, l)[0].T @ s.T @ s @ matrices_N2e(x,l)[0]
    return dki[i][j]

def matrix_Kinter(kc, hb, ht, ti, li, eldof):
    """ Shear stiffness of the connector between the 2 layers. kc is the scalar value in N/mm """
    Kinter = np.zeros((eldof, eldof))
    Si = ti*li
    for i, k in enumerate(Kinter):
        for j, dk in enumerate(k):
            Kinter[i][j] = (ti/Si) * integrate.quad(dkinter, 0,  li, args=(i, j, kc, hb, ht, li))[0]
    return Kinter
