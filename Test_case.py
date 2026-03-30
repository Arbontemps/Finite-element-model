#%%
""" 
Test cas 1 : One element loaded in pure bending with 2 torques. Is the displacement symetric ?
Test cas 3 : Three element loaded in pure bending with 2 torques. Is the displacement symetric ?
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from BV_fonctions_assemblage import assemblage_K, connectivity
from BV_fonctions_LL_BB import list_dof_l, matrices_LL_BB, vecteur_LL_BB, reorganisation_dof_vec
from FEM_functions import matrices_N, matrix_ke, matrix_Kr, matrix_K2e, matrix_Kc, U0, matrices_N2e
from Analytical_beam_formulations import beam_3pt_deflection
from test import BIEN LE BONJOUR CAMARADE

#%% Arbitrary geometry
hb = 100
eb = 20
Eb = 10e3
Ib = eb*hb**3/12
Sb = eb * hb
yb = hb/2 #0 at bottom

ht = 50
et = 50
Et = 2e3
It = et*ht**3/12
St = et * ht
yt = hb + ht/2

n = Et/Eb
ygtot = (yb*Sb + yt*St) / (Sb+St) 
#%% 
""" Test case 1: application on a bi-supported beam loaded in 3-pt bending with 3 elements """
# Length, node vector and load definition
L = 1500
f = 200*9.81
# Define the dof of both bot (_b) and top (_t) layer
ndof_b, ndof_t = 3, 1
eldof = (ndof_b + ndof_t) * 2
nnodes = 3
nel = nnodes - 1
eldof_tot = nnodes * ndof_b + nnodes * ndof_t
list_dof = np.linspace(0, eldof_tot - 1, eldof_tot, dtype=int)

l_el = L / nel

# Boundary conditions
BC = [0, 1, 7, 9] # dof number that is equal to 0 
eldof_L = list_dof_l(BC, eldof_tot)[0] # free dof number
load = np.zeros(eldof_tot)
load[1], load[4], load[9] = -f/2, f, -f/2

# Stiffness matrix element and assembly
# Be careful, dof notation is particular : bot node 1 : 3,4,5; bot node 2 : 8,9,10; top node 1 :7; top node 2: 11
kel = matrix_K2e(Eb, hb, eb, Et, ht, et, l_el)
TC_nodes, TC_dof = connectivity()
list_el = [{'dof_el':[0,1,2,3,4,5,6,7], 'Kel':kel}, {'dof_el':[3, 4, 5, 8, 9, 10, 7, 11], 'Kel':kel}]
Ktot = assemblage_K(nel, list_el, eldof_tot, eldof)
Ktot_LL = matrices_LL_BB(Ktot, eldof_L, BC)[0]

# Compute displacement
U = np.zeros(eldof_tot)
U[np.delete(list_dof, BC)] = np.linalg.solve(Ktot_LL, load[np.delete(list_dof, BC)])

# Plot the result
def U0(x, l, u):
    U0 = matrices_N2e(x, l)[0] @ u
    return U0

xtot =  [0]
vtot = [0]
for i in range(nel):
    uel = U[list_el[i]['dof_el']]
    xel = np.linspace(0, l_el, 100)
    xtot.extend(xel+xtot[-1])
    for x in xel:
        v = U0(x, l_el, uel)[1]
        vtot.append(-v)
vtot = np.array(vtot)
EI_analytic = Eb*Ib + Et*It
x_analytic, v_analytic = beam_3pt_deflection(L, EI_analytic, f)

plt.plot(x_analytic, -v_analytic, c='red', label='Analytical solution')
plt.plot(xtot, vtot, c='blue', label='Model')
plt.title('Test case 1 : Unconnected bi-layer loaded in 3-pt bending')
plt.legend()
plt.show()

#%%
""" Test case 2: application on a bi-supported beam loaded in 3-pt bending with 3 elements, with infinite stiffness at interface """
# Length, node vector and load definition
L = 1500
f = 200*9.81
# Define the dof of both bot (_b) and top (_t) layer
ndof_b, ndof_t = 3, 1
eldof = (ndof_b + ndof_t) * 2
nnodes = 3
nel = nnodes - 1
eldof_tot = nnodes * ndof_b + nnodes * ndof_t
list_dof = np.linspace(0, eldof_tot - 1, eldof_tot, dtype=int)

l_el = L / nel

# Boundary conditions
BC = [0, 1, 7, 9] # dof number that is equal to 0 
eldof_L = list_dof_l(BC, eldof_tot)[0] # free dof number
load = np.zeros(eldof_tot)
load[1], load[4], load[9] = -f/2, f, -f/2

# Stiffness matrix element and assembly
# Be careful, dof notation is particular : bot node 1 : 3,4,5; bot node 2 : 8,9,10; top node 1 :7; top node 2: 11
kel = matrix_K2e(Eb, hb, eb, Et, ht, et, l_el) + matrix_Kc(1e10, hb, ht, 0, l_el)
TC_nodes, TC_dof = connectivity()
list_el = [{'dof_el':[0,1,2,3,4,5,6,7], 'Kel':kel}, {'dof_el':[3, 4, 5, 8, 9, 10, 7, 11], 'Kel':kel}]
Ktot = assemblage_K(nel, list_el, eldof_tot, eldof)
Ktot_LL = matrices_LL_BB(Ktot, eldof_L, BC)[0]

# Compute displacement
U = np.zeros(eldof_tot)
U[np.delete(list_dof, BC)] = np.linalg.solve(Ktot_LL, load[np.delete(list_dof, BC)])

xtot =  [0]
vtot = [0]
for i in range(nel):
    uel = U[list_el[i]['dof_el']]
    xel = np.linspace(0, l_el, 100)
    xtot.extend(xel+xtot[-1])
    for x in xel:
        v = U0(x, l_el, uel)[1]
        vtot.append(-v)
vtot = np.array(vtot)
EI_analytic = Eb*Ib + Et*It + Eb*Sb*np.abs(ygtot - yb)**2 + Et*St*np.abs(ygtot - yt)**2
x_analytic, v_analytic = beam_3pt_deflection(L, EI_analytic, f)

plt.plot(x_analytic, -v_analytic, c='red', label='Analytical solution')
plt.plot(xtot, vtot, c='blue', label='Model')
plt.title('Test case 2 : connected bi-layer loaded in 3-pt bending')
plt.legend()
plt.show()