#%%
"""
Arthur Bontemps
Test case 3: Unconnected bi-layer beam with axial load on each layer (2 calculations), with 3 elements. Comparison with analytical results
Test case 4: Connected bi-layer beam with axial load on the top layer, with 3 elements and infinite stiffness at interface.
    Comparison with analytical results
 for each layer to show rigid body motion. One can calculate the force at the interface analytically and compare to what shows the model.
  
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
nnodes_per_elem = 4
ndof_per_elem = nnodes_per_elem * ndof_per_node
nnodes = 8
nelem = int(nnodes/2 - 1)
ndof = nnodes * ndof_per_node
list_dof = np.linspace(0, ndof - 1, ndof, dtype=int)
l_elem = L / nelem


# Boundary conditions
# dof number that is equal to 0 -> clamped beam
BC = [0, 1, 2, 3]
cons = constraints(list_dof)

# definition of the stiffness element matrix and assembly global matrix
kelem = true_k2B_matrix(Eb, Sb, Ib, Et, St, It, l_elem, ndof_per_elem)
kelem_i = true_k2B_matrix(Eb, Sb, Ib, Eb, Sb, Ib, l_elem, ndof_per_elem) + matrix_ki(1e10, hb, hb, eb, l_elem, ndof_per_elem)

_, _, _, local_dof = connectivity(nnodes_per_elem, nelem, ndof, 'bilayer')
list_elem = [{'dof_elem':local_dof + 6*i, 'kelem':kelem} for i in range(nelem)]
list_elem_i = [{'dof_elem':local_dof + 6*i, 'kelem':kelem_i} for i in range(nelem)]

Ktot = assemblage_K(nelem, list_elem, ndof, ndof_per_elem)
Ktot_i = assemblage_K(nelem, list_elem_i, ndof, ndof_per_elem)

# reduce global stiffness matrix to the constrained Ktot_tilde (because dof linked between the bot and top layer)
T, indep = matrix_T(ndof, cons)
BC_red = map_BC_to_reduced(BC, indep)

Ktot_tilde = T.T @ Ktot @ T
Ktot_i_tilde = T.T @ Ktot_i @ T

# definition of the global force vector
Ftot_b, Ftot_t = np.zeros(ndof), np.zeros(ndof)
Ftot_b[0], Ftot_b[18] = -f, f
Ftot_t[3], Ftot_t[21] = -f, f
Ftot_b_tilde = T.T @ Ftot_b
Ftot_t_tilde = T.T @ Ftot_t

# add BC on the matrix system
Ktot_tilde, Ftot_b_tilde = apply_dirichlet_BC(Ktot_tilde, Ftot_b_tilde, BC_red)
Ktot_i_tilde, Ftot_t_tilde = apply_dirichlet_BC(Ktot_i_tilde, Ftot_t_tilde, BC_red)

# Resolution and reconstruction
Utot_b_tilde = np.linalg.solve(Ktot_tilde, Ftot_b_tilde)
Utot_t_tilde = np.linalg.solve(Ktot_tilde, Ftot_t_tilde)
Utot_i_tilde = np.linalg.solve(Ktot_i_tilde, Ftot_t_tilde)
Utot_b = T @ Utot_b_tilde
Utot_t = T @ Utot_t_tilde
Utot_i = T @ Utot_i_tilde

# prepare the plots by extraction the deflection along the beam with form functions 
xtot =  [0]
utot_b, utot_t, utot_i = [0], [0], [0]
for i in range(nelem):
    uelem_b = Utot_b[list_elem[i]['dof_elem']]
    uelem_t = Utot_t[list_elem[i]['dof_elem']]
    uelem_i = Utot_i[list_elem_i[i]['dof_elem']]
    xelem = np.linspace(0, l_elem, 10)
    xtot.extend(xelem+xtot[-1])
    for x in xelem:
        u_b, u_t = U_NA_bilayer(x, l_elem, uelem_b)[0], U_NA_bilayer(x, l_elem, uelem_t)[3]
        u_i = U_NA_bilayer(x, l_elem, uelem_i)[3]
        utot_b.append(-u_b)
        utot_t.append(-u_t)
        utot_i.append(-u_i)
utot_b, utot_t, utot_i = np.array(utot_b), np.array(utot_t), np.array(utot_i)

# calculate analytical solutions u(x) = (F*x) / (E*S) 
x_analytic, u_analytic_b = beam_clamped_tension(f, Eb, Sb, L)
x_analytic, u_analytic_t = beam_clamped_tension(f, Et, St, L)
x_analytic, u_analytic_i = beam_clamped_tension(f, Eb, 2*Sb, L)

# plot results and compare with an analytical solution
plt.plot(x_analytic, -u_analytic_b, c='red', label='Analytical solution ux_b')
plt.plot(x_analytic, -u_analytic_t, c='blue', label='Analytical solution ux_t')
plt.plot(x_analytic, -u_analytic_i, c='brown', label='Analytical solution kc=inf ux_t')
plt.plot(xtot, utot_b, c='red', label='Model ux_b', linestyle='none', marker='.')
plt.plot(xtot, utot_t, c='blue', label='Model ux_t', linestyle='none', marker='.')
plt.plot(xtot, utot_i, c='brown', label='Model kc=inf ux_t', linestyle='none', marker='.')
plt.xlabel('x-position (mm)')
plt.ylabel('axial displacement (mm)')
plt.title('Test case 3&4 : Clamped beam with axial load and '+str(nelem)+' elements')
plt.legend()
plt.show()

