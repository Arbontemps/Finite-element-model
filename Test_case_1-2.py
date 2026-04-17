#%%
"""
Arthur Bontemps
Test case 1: 2 elements bi-supported unconnected bi-layer beam in 3-pt bending. Comparison with analytical results
Test case 2: 2 elements bi-supported connected bi-layer beam in 3-pt bending with infinite stiffness at the interface and same elastic modulus for each layer. 
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
nnodes_per_elem = 4
ndof_per_elem = nnodes_per_elem * ndof_per_node
nnodes = 6
nelem = int(nnodes/2 - 1)
ndof = nnodes * ndof_per_node
list_dof = np.linspace(0, ndof - 1, ndof, dtype=int)
l_elem = L / nelem


# definition of the stiffness element matrix and assembly global matrix
kelem = true_k2B_matrix(Eb, Sb, Ib, Et, St, It, l_elem, ndof_per_elem)
kelem_i = true_k2B_matrix(Eb, Sb, Ib, Eb, Sb, Ib, l_elem, ndof_per_elem) + matrix_ki(1e10, hb, hb, eb, l_elem, ndof_per_elem)

_, _, _, local_dof = connectivity(nnodes_per_elem, nelem, ndof_per_node, 'bilayer')
list_elem = [{'dof_elem':local_dof, 'kelem':kelem}, \
           {'dof_elem':local_dof+6, 'kelem':kelem}]
list_elem_i = [{'dof_elem':local_dof, 'kelem':kelem_i}, \
           {'dof_elem':local_dof+6, 'kelem':kelem_i}]
Ktot = assemblage_K(nelem, list_elem, ndof, ndof_per_elem)
Ktot_i = assemblage_K(nelem, list_elem_i, ndof, ndof_per_elem)

# Boundary conditions
# dof number that is equal to 0
BC = [0, 1, 9, 13]
cons = constraints(list_dof)

# reduce global stiffness matrix to the constrained K_tilde (because dof linked between the bot and top layer)
T, indep = matrix_T(ndof, cons)
BC_red = map_BC_to_reduced(BC, indep)

K_tilde = T.T @ Ktot @ T
K_tilde_i = T.T @ Ktot_i @ T

# definition of the global force vector
Ftot = np.zeros(ndof)
Ftot[1], Ftot[10], Ftot[13] = -f/2, f, -f/2
F_tilde = T.T @ Ftot

# add BC on the matrix system
K_tilde, F_tilde = apply_dirichlet_BC(K_tilde, F_tilde, BC_red)
K_tilde_i, _ = apply_dirichlet_BC(K_tilde_i, F_tilde, BC_red)

# Resolution and reconstruction
U_tilde = np.linalg.solve(K_tilde, F_tilde)
U_tilde_i = np.linalg.solve(K_tilde_i, F_tilde)
Utot = T @ U_tilde
Utot_i = T @ U_tilde_i

# prepare the plots by extraction the deflection along the beam with form functions 
xtot =  [0]
vtot, vtot_i = [0], [0]
for i in range(nelem):
    uelem = Utot[list_elem[i]['dof_elem']]
    uelem_i = Utot_i[list_elem_i[i]['dof_elem']]
    xel = np.linspace(0, l_elem, 10)
    xtot.extend(xel+xtot[-1])
    for x in xel:
        v = U_NA_bilayer(x, l_elem, uelem)[1]
        v_i = U_NA_bilayer(x, l_elem, uelem_i)[1]
        vtot.append(-v)
        vtot_i.append(-v_i)
vtot, vtot_i = np.array(vtot), np.array(vtot_i)

# calculate analytical solutions
EI_analytic_unconnected = Eb*Ib + Et*It
EI_analytic_i = Eb*(eb*(2*hb)**3)/12
x_analytic, v_analytic_unconnected = beam_3pt_deflection(L, EI_analytic_unconnected, f)
x_analytic, v_analytic_i = beam_3pt_deflection(L, EI_analytic_i, f)

# plot results and compare with an analytical solution
plt.plot(x_analytic, -v_analytic_unconnected, c='red', label='Analytical solution kc=0')
plt.plot(x_analytic, -v_analytic_i, c='blue', label='Analytical solution kc=inf')
plt.plot(xtot, vtot, c='red', label='Model kc=0', linestyle='none', marker='.')
plt.plot(xtot, vtot_i, c='blue', label='Model kc=inf', linestyle='none', marker='.')
plt.xlabel('x-position (mm)')
plt.ylabel('deflection (mm)')
plt.title('Test case 1&2 : Unconnected and perfectly connected bi-layer in 3-pt bending')
plt.legend()
plt.show()


