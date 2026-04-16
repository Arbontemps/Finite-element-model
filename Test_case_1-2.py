#%%
"""
Arthur Bontemps
Test case 1: Bi-supported unconnected bi-layer beam in 3-pt bending with 2 elements. Comparison with analytical results
Test case 2: Bi-supported connected bi-layer beam in 3-pt bending with infinite stiffness at the interface and same elastic modulus for each layer. 
    Comparison with a monolytic beam and its analytical results.
"""
import numpy as np
import matplotlib.pyplot as plt
from AB_BV_fonctions_assemblage import *
from FEM_functions import *
from Analytical_beam_formulations import beam_3pt_deflection

#%% Arbitrary geometry and load definition(N, mm, MPa)
hb = 100
eb = 20
Eb = 10e3
Ib = eb*hb**3/12
Sb = eb * hb
yb = hb/2 # origin at bottom

ht = 50
et = 50
Et = 2e3
It = et*ht**3/12
St = et * ht
yt = hb + ht/2 # origin at bottom

n = Et/Eb
ygtot = (yb*Sb + yt*St) / (Sb+St) 

# Length and load definition
L = 1500
f = 200*9.81

# Meshing : define the dof, nodes and elements
ndof_per_node = 3
nnodes_per_el = 4
ndof_per_el = nnodes_per_el * ndof_per_node
nnodes = 6
nel = int(nnodes/2 - 1)
ndof = nnodes * ndof_per_node
list_dof = np.linspace(0, ndof - 1, ndof, dtype=int)
l_el = L / nel


# definition of the stiffness element matrix and assembly global matrix
kel = true_K2e_matrix(Eb, Sb, Ib, Et, St, It, l_el, ndof_per_el)
kel_int = true_K2e_matrix(Eb, Sb, Ib, Eb, Sb, Ib, l_el, ndof_per_el) + matrix_Kinter(1e10, hb, ht, min(eb,et), l_el, ndof_per_el)

_, _, _, local_dof = connectivity(nnodes_per_el, nel, ndof_per_node, 'bilayer')
list_el = [{'dof_el':local_dof, 'Kel':kel}, \
           {'dof_el':local_dof+6, 'Kel':kel}]
list_el_int = [{'dof_el':local_dof, 'Kel':kel_int}, \
           {'dof_el':local_dof+6, 'Kel':kel_int}]
Ktot = assemblage_K(nel, list_el, ndof, ndof_per_el)
Ktot_int = assemblage_K(nel, list_el_int, ndof, ndof_per_el)

# Boundary conditions
# dof number that is equal to 0
BC = [0, 1, 9, 13]
cons = constraints(list_dof)

# reduce global stiffness matrix to the constrained K_tilde (because dof linked between the bot and top layer)
T, indep = matrix_T(ndof, cons)
BC_red = map_BC_to_reduced(BC, indep)

K_tilde = T.T @ Ktot @ T
K_tilde_int = T.T @ Ktot_int @ T

# definition of the global force vector
Ftot = np.zeros(ndof)
Ftot[1], Ftot[10], Ftot[13] = -f/2, f, -f/2
F_tilde = T.T @ Ftot

# add BC on the matrix system
K_tilde, F_tilde = apply_dirichlet_BC(K_tilde, F_tilde, BC_red)
K_tilde_int, _ = apply_dirichlet_BC(K_tilde_int, F_tilde, BC_red)

# Resolution and reconstruction
U_tilde = np.linalg.solve(K_tilde, F_tilde)
U_tilde_int = np.linalg.solve(K_tilde_int, F_tilde)
Utot = T @ U_tilde
Utot_int = T @ U_tilde_int

# prepare the plots by extraction the deflection along the beam with form functions 
xtot =  [0]
vtot, vtot_int = [0], [0]
for i in range(nel):
    uel = Utot[list_el[i]['dof_el']]
    uel_int = Utot_int[list_el_int[i]['dof_el']]
    xel = np.linspace(0, l_el, 100)
    xtot.extend(xel+xtot[-1])
    for x in xel:
        v = U0_bilayer(x, l_el, uel)[1]
        v_int = U0_bilayer(x, l_el, uel_int)[1]
        vtot.append(-v)
        vtot_int.append(-v_int)
vtot, vtot_int = np.array(vtot), np.array(vtot_int)

# calculate analytical solutions
EI_analytic_unconnected = Eb*Ib + Et*It
EI_analytic_int = Eb*(eb*(2*hb)**3)/12
x_analytic, v_analytic_unconnected = beam_3pt_deflection(L, EI_analytic_unconnected, f)
x_analytic, v_analytic_int = beam_3pt_deflection(L, EI_analytic_int, f)

# plot results and compare with an analytical solution
plt.plot(x_analytic, -v_analytic_unconnected, c='red', label='Analytical solution kc=0')
plt.plot(x_analytic, -v_analytic_int, c='purple', label='Analytical solution kc=inf')
plt.plot(xtot, vtot, c='blue', label='Model kc=0')
plt.plot(xtot, vtot_int, c='black', label='Model kc=inf')
plt.xlabel('x-position (mm)')
plt.ylabel('deflection (mm)')
plt.title('Test case 1&2 : Unconnected and perfectly connected bi-layer in 3-pt bending')
plt.legend()
plt.show()

