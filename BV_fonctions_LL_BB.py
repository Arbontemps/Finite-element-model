#! /usr/bin/env python3
# _*_ coding: utf8 _*_

###########################################################
### FONCTIONS POUR LA GESTION DES DDL LIBRES ET BLOQUES ###
###########################################################

### LISTE

# list_dof_l : Fonction qui cree 1/la liste des dof libres a partir du nbre de dof et de la liste des dof bloques/a dplcmt impose 2/assemblage des listes des dof libres et bloques pour avoir les indexs en vue de la reorganisation

# matrices_LL_BB : Fonction qui separe une matrice matr (type masse ou rigidite) en ses sous matrices matr_LL, matr_LB, matr_BL et matr_BB et qui assemble ces sous matrices en une matrice reoganisee matr_LL_BB

# vecteur_LL_BB :  Fonction qui separe un vecteur vect (type deplacement ou vitesse) en ses sous vecteurs vect_LL et vect_BB et qui assemble ces sous vecteurs en unvecteur reoganisee vect_LL_BB

#reorganisation_dof_mat : Fonction qui reorganise dans l'ordre ordonne des dof une matrice reorganisee _LL_BB

#reorganisation_dof_vec : Fonction qui reorganise dans l'ordre ordonne des dof un vecteur reorganisee _LL_BB

import numpy as np
import os

def list_dof_l(dof_b,ndof):
    """
    Fonction qui cree
    1/la liste des dof libres a partir du nbre de dof et de la liste des dof bloques/a dplcmt impose
    2/assemblage des listes des dof libres et bloques pour avoir les indexs en vue de la reorganisation
    """
    dof_l = list(range(ndof)) #creation de la liste complete des dof
    for i in dof_b:#suppression des dof_b
        dof_l.remove(i)
    dof_l_b = dof_l + dof_b
    return(dof_l, dof_b, dof_l_b)

def matrices_LL_BB(matr, dof_libres, dof_bloques):
    """
    Fonction qui separe une matrice matr (type masse ou rigidite)
    en ses sous matrices matr_LL, matr_LB, matr_BL et matr_BB
    et qui assemble ces sous matrices en une matrice reoganisee matr_LL_BB
    """
    matr_LL = matr
    matr_LB = matr
    matr_BL = matr
    matr_BB = matr
    matr_LL = np.delete(matr_LL, dof_bloques, axis = 1)
    matr_LL = np.delete(matr_LL, dof_bloques, axis = 0)
    matr_LB = np.delete(matr_LB, dof_bloques, axis = 0)
    matr_LB = np.delete(matr_LB, dof_libres, axis = 1)
    matr_BL = np.delete(matr_BL, dof_bloques, axis = 1)
    matr_BL = np.delete(matr_BL, dof_libres, axis = 0)
    matr_BB = np.delete(matr_BB, dof_libres, axis = 1)
    matr_BB = np.delete(matr_BB, dof_libres, axis = 0)
    matr_L = np.concatenate((matr_LL,matr_LB),axis = 1)
    matr_B = np.concatenate((matr_BL,matr_BB),axis = 1)
    matr_LL_BB = np.concatenate((matr_L,matr_B), axis = 0)
    return(matr_LL,matr_LB,matr_BL,matr_BB,matr_LL_BB)

def matrices_rect_LL_BB(matr,dof_libres, dof_bloques):
    """
    Fonction qui separe une matrice matr (rectangulaire type L en elements mixtes)
    en ses sous matrices matr_LL, matr_LB, matr_BL et matr_BB
    et qui assemble ces sous matrices en une matrice reoganisee matr_LL_BB
    A VOIR UTILITE SI DIFFERENT DE VECTEURS OU NON (fonction vecteur_LL_BB)
    """
    matr_LL = matr
    matr_BB = matr
    matr_LL = np.delete(matr_LL, dof_bloques, axis = 0)
    matr_BB = np.delete(matr_BB, dof_libres, axis = 0)
    matr_LL_BB = np.concatenate((matr_LL,matr_BB), axis = 0)
    return(matr_LL,matr_BB,matr_LL_BB)

def vecteur_LL_BB(vect,dof_libres, dof_bloques):
    """
    Fonction qui separe un vecteur vect (type deplacement ou vitesse)
    en ses sous vecteurs vect_LL et vect_BB
    et qui assemble ces sous vecteurs en unvecteur reoganisee vect_LL_BB
    """
    vect_LL = vect
    vect_BB = vect
    vect_LL = np.delete(vect_LL, dof_bloques)
    vect_BB = np.delete(vect_BB, dof_libres)
    vect_LL_BB = np.concatenate((vect_LL, vect_BB))
    return(vect_LL, vect_BB, vect_LL_BB)

def reorganisation_dof_mat(ndof,dof,matrTriee):
    """
    Fonction qui reorganise dans l'ordre ordonne des dof une matrice reorganisee _LL_BB
    """
    matrReconstituee = np.zeros((ndof,ndof))
    for i in range(ndof):
        n = dof[i]
        for j in range(ndof):
            m = dof[j]
            matrReconstituee[n,m] = matrTriee[i,j]
    return(matrReconstituee)

def reorganisation_dof_vec(ndof,dof,vecTriee):
    """
    Fonction qui reorganise dans l'ordre ordonne des dof un vecteur reorganisee _LL_BB
    """
    vecReconstitue = np.zeros(ndof)
    for i in range(ndof):
        n = dof[i]
        vecReconstitue[n] = vecTriee[i]
    return(vecReconstitue)
