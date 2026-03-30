#! /usr/bin/env python3
# _*_ coding: utf8 _*_

#######################################
### DIVERSES FONCTIONS D'ASSEMBLAGE ###
#######################################

import numpy as np

from math import *
import csv

import os

# from fonctions_generales import *

def connectivity():
    """ Define the node and dof connectivity tables for the bilayer element with 8 element and 8 dof/elem"""
    TCn0, TCn1 = [0,1,2,3], [1,4,3,5]
    TCn2, TCn3 = [4,6,5,7], [6,8,7,9]
    TCn4, TCn5 = [8,10,9,11], [10,12,11,13]
    TCn6, TCn7 = [12,14,13,15], [14,16,15,17]
    TC_nodes = [TCn0, TCn1,TCn2,TCn3,TCn4,TCn5,TCn6,TCn7]
    TCd0, TCd1, TCd2, TCd3 = [0,1,2], [3,4,5], [6], [7] 
    TCd4, TCd5, TCd6, TCd7 = [8,9,10], [11], [12,13,14], [15] 
    TCd8, TCd9, TCd10, TCd11 = [16,17,18], [19], [20,21,22], [23]
    TCd12, TCd13, TCd14, TCd15 = [24,25,26], [27], [28,29,30], [31]
    TCd16, TCd17 = [32,33,34], [35]
    TC_dof = [TCd0, TCd1, TCd2, TCd3, TCd4, TCd5, TCd6, TCd7, TCd8, TCd9, TCd10, \
              TCd11, TCd12, TCd13, TCd14, TCd15, TCd16, TCd17]
    return TC_nodes, TC_dof

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

def assemblage_K(nel, list_el, ndof, ndof_el):
    """
    Assemblage des matrices carrées élémentaires (type mel en M)
    """
    M = np.zeros((ndof,ndof))
    for iel in range(nel):
        dof_globaux = list_el[iel]['dof_el']
        for i in range(ndof_el):
            linM = dof_globaux[i]
            for j in range(ndof_el):
                colM = dof_globaux[j]
                M[linM, colM] = M[linM, colM] + list_el[iel]['Kel'][i,j]
    return M

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

