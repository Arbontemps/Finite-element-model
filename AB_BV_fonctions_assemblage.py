#! /usr/bin/env python3
# _*_ coding: utf8 _*_

#######################################
### DIVERSES FONCTIONS D'ASSEMBLAGE ###
#######################################

import numpy as np
from math import *


def connectivity(nnodes_per_el, nel, ndof, keyword):
    """ Define the node and dof connectivity tables """
    if keyword == 'monolayer':
        TC_elnodes = []
        for i in range(nel):
            TC_elnodes.append([i+k for k in range(nnodes_per_el)])
        TC_ndof = []
        for n in range(nel*nnodes_per_el-2):
            TC_ndof.append([ndof*n+k for k in range(ndof)])
        TC_nloc = np.array([0, 1]) # local numerotation of nodes association for global numerotation [ne1, ne2, ne3, ne4]
        TC_dofloc = np.array([0, 1, 2, 3, 4, 5]) # local numerotation of dof association for global numerotation [dof1, dof2, ..., dof7]
    
    elif keyword == 'bilayer':
        TC_elnodes = []
        for i in range(nel):
            TC_elnodes.append([2*i+k for k in range(nnodes_per_el)])
        TC_ndof = []
        for n in range(nel*nnodes_per_el-2):
            TC_ndof.append([ndof*n+k for k in range(ndof)])
        TC_nloc = np.array([0, 2, 1, 3]) # local numerotation of nodes association for global numerotation [ne1, ne2, ne3, ne4]
        TC_dofloc = np.array([0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]) # local numerotation of dof association for global numerotation [dof1, dof2, ..., dof7]
    else:
        print('use keyword among monolayer, bilayer')
    return np.array(TC_elnodes), np.array(TC_ndof), TC_nloc, TC_dofloc


def constraints(list_dof):
    """ Return dof number where dof equality is needed (because v3=v1 and beta3=beta1) """
    cons = {}
    k = 0
    for i in list_dof:
        if k==4:
            cons[i] = i-3
        elif k==5:
            cons[i] = i-3
            k=0
            continue
        k+=1
    return cons

def matrix_T(eldof_tot, constraints):
    """ Apply constraints to the system, because dof are linked
     based on N.B hypothesis, v3 = v1, v4=v2, beta3 = beta1 and beta4=beta2
     T matrix can transform global matrices on constrained matrices with all 
     v3, v4, beta3 and beta4 removed """
    indep = [i for i in range(eldof_tot) if i not in constraints]
    n_red = len(indep)
    T = np.zeros((eldof_tot, n_red))
    # independent dof 
    for col, dof in enumerate(indep):
        T[dof, col] = 1
    # constraints
    for slave, master in constraints.items():
        col = indep.index(master)
        T[slave, col] = 1
    return T, indep

def apply_dirichlet_BC(K, F, BC):
    """ Apply dirichlet BC (ui = 0) on K and F """
    for dof in BC:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1
        F[dof] = 0
    return K, F

def map_BC_to_reduced(BC, indep):
    """
    Convertit les ddl bloqués du système complet
    vers les indices du système réduit
    """
    BC_red = []
    for dof in BC:
        if dof in indep:  # only if independent
            BC_red.append(indep.index(dof))
    return BC_red



def assemblage_K(nel, list_el, ndof, ndof_el):
    """
    Assemblage des matrices carrées élémentaires (type mel en K)
    """
    K = np.zeros((ndof,ndof))
    for iel in range(nel):
        dof_globaux = list_el[iel]['dof_el']
        for i in range(ndof_el):
            linK = dof_globaux[i]
            for j in range(ndof_el):
                colK = dof_globaux[j]
                K[linK, colK] = K[linK, colK] + list_el[iel]['Kel'][i,j]
    return K


def assemblage_F(nel, list_el, ndof, ndof_el):
    """
    Assemblage des vecteurs élémentaires
    """
    F = np.zeros((ndof,1))
    for iel in range(nel):
        dof_globaux = list_el[iel]['dof_el']
        for i in range(ndof_el):
            linF = dof_globaux[i]
            F[linF,0] = F[linF,0] + list_el[iel]['Fel'][i,0]
    return F


def assemblage_B(nel, n_pts_g, list_el, ndof, ndof_el):
    """
    Assemblage des matrices rectangles élémentaires (type bel en B)
    """
    B = np.zeros((2*n_pts_g, ndof)) #2 *nptsg car N et M a chaque point
    B_equ = np.zeros((2*n_pts_g, ndof)) #2 *nptsg car N et M a chaque point
    for iel in range(nel):
	    dof_globaux = list_el[iel]['dof_el']
	    for i in range(4):#N et M a chaque point de Gauss
		    linB = 4*iel + i
		    for j in range(ndof_el):#Pour chaque dof
			    colB = dof_globaux[j]
			    B[linB, colB] = B[linB, colB] + list_el[iel]['Bel'][i][j]
			    B_equ[linB,colB] = B_equ[linB,colB] + list_el[iel]['Bel_eq'][i][j]
    return B, B_equ

def assemb_J(nel, n_pts_g, list_el, J, J_type):
    """
    Assemblage des matrices rectangles d'écrouissage élémentaires (J_sigma ou J_a)
    """
    nb_col_J_elem = int(np.shape(J)[1]/(n_pts_g))
    nb_lig_J_elem =int(np.shape(J)[0]/(n_pts_g))
    for iel in range(nel):
        for pt_g_el in range(list_el[iel]['n_g_el']): #2 gauss point for each elment
            for ii in range(nb_col_J_elem): #columns in elementary J
                colJ = nb_col_J_elem*(2*iel+pt_g_el) + ii
                for ij in range(nb_lig_J_elem): #4 rows in elementary J
                    ligJ = 4*(2*iel+pt_g_el) + ij
                    J[ligJ,colJ] = list_el[iel][J_type][pt_g_el][ij][ii]
    return J

