#%%
"""
Arthur Bontemps
Test case 3: Unconnected bi-layer beam with axial load on each layer (2 calculations). Comparison with analytical results
Test case 4: Connected bi-layer beam with axial load on the top layer, infinite stiffness for each layer to show rigid body motion. 
    One can calculate the force at the interface analytically and compare to what shows the model.
  
"""
import numpy as np
import matplotlib.pyplot as plt
from AB_BV_fonctions_assemblage import *
from FEM_functions import *
from Analytical_beam_formulations import beam_clamped_tension

#%% Arbitrary geometry (N, mm, MPa)
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
f = 100*9.81

# Meshing : define the dof, nodes and elements
ndof_per_node = 3
nnodes_per_el = 4
ndof_per_el = nnodes_per_el*ndof_per_node
nnodes = 8
nel = int(nnodes/2 - 1)
ndof = nnodes * ndof_per_node
list_dof = np.linspace(0, ndof - 1, ndof, dtype=int)
l_el = L / nel


# Boundary conditions
# dof number that is equal to 0 -> clamped beam
BC = [0, 1, 2, 3]
cons = constraints(list_dof)

# definition of the stiffness element matrix and assembly global matrix
kel = true_K2e_matrix(Eb, Sb, Ib, Et, St, It, l_el, ndof_per_el)
kel_int = true_K2e_matrix(Eb, Sb, Ib, Eb, Sb, Ib, l_el, ndof_per_el) + matrix_Kinter(1e10, hb, ht, min(eb,et), l_el, ndof_per_el)

_, _, _, local_dof = connectivity(nnodes_per_el, nel, ndof, 'bilayer')
list_el = [{'dof_el':local_dof + 6*i, 'Kel':kel} for i in range(nel)]
list_el_int = [{'dof_el':local_dof + 6*i, 'Kel':kel_int} for i in range(nel)]

Ktot = assemblage_K(nel, list_el, ndof, ndof_per_el)
Ktot_int = assemblage_K(nel, list_el_int, ndof, ndof_per_el)

# reduce global stiffness matrix to the constrained K_tilde (because dof linked between the bot and top layer)
T, indep = matrix_T(ndof, cons)
BC_red = map_BC_to_reduced(BC, indep)

K_tilde = T.T @ Ktot @ T
K_tilde_int = T.T @ Ktot_int @ T

# definition of the global force vector
Ftot_b, Ftot_t = np.zeros(ndof), np.zeros(ndof)
Ftot_b[0], Ftot_b[18] = -f, f
Ftot_t[3], Ftot_t[21] = -f, f
F_tilde_b = T.T @ Ftot_b
F_tilde_t = T.T @ Ftot_t

# add BC on the matrix system
K_tilde, F_tilde_b = apply_dirichlet_BC(K_tilde, F_tilde_b, BC_red)
K_tilde_int, F_tilde_t = apply_dirichlet_BC(K_tilde_int, F_tilde_t, BC_red)

# Resolution and reconstruction
U_tilde_b = np.linalg.solve(K_tilde, F_tilde_b)
U_tilde_t = np.linalg.solve(K_tilde, F_tilde_t)
U_tilde_int = np.linalg.solve(K_tilde_int, F_tilde_t)
Utot_b = T @ U_tilde_b
Utot_t = T @ U_tilde_t
Utot_int = T @ U_tilde_int

# prepare the plots by extraction the deflection along the beam with form functions 
xtot =  [0]
utot_b, utot_t, utot_int = [0], [0], [0]
for i in range(nel):
    uel_b = Utot_b[list_el[i]['dof_el']]
    uel_t = Utot_t[list_el[i]['dof_el']]
    uel_int = Utot_int[list_el_int[i]['dof_el']]
    xel = np.linspace(0, l_el, 100)
    xtot.extend(xel+xtot[-1])
    for x in xel:
        u_b, u_t = U0_bilayer(x, l_el, uel_b)[0], U0_bilayer(x, l_el, uel_t)[3]
        u_int = U0_bilayer(x, l_el, uel_int)[3]
        utot_b.append(-u_b)
        utot_t.append(-u_t)
        utot_int.append(-u_int)
utot_b, utot_t, utot_int = np.array(utot_b), np.array(utot_t), np.array(utot_int)

# calculate analytical solutions u(x) = (F*x) / (E*S) 
x_analytic, u_analytic_b = beam_clamped_tension(f, Eb, Sb, L)
x_analytic, u_analytic_t = beam_clamped_tension(f, Et, St, L)
x_analytic, u_analytic_int = beam_clamped_tension(f, Eb, Sb+St, L)

# plot results and compare with an analytical solution
plt.plot(x_analytic, -u_analytic_b, c='red', label='Analytical solution ux_b')
plt.plot(x_analytic, -u_analytic_t, c='purple', label='Analytical solution ux_t')
plt.plot(x_analytic, -u_analytic_int, c='brown', label='Analytical solution kc=inf ux_t')
plt.plot(xtot, utot_b, c='blue', label='Model ux_b')
plt.plot(xtot, utot_t, c='black', label='Model ux_t')
plt.plot(xtot, utot_int, c='orange', label='Model kc=inf ux_t')
plt.xlabel('x-position (mm)')
plt.ylabel('axial displacement (mm)')
plt.title('Test case 3&4 : Clamped beam with axial load and '+str(nel)+' elements')
plt.legend()
plt.show()

