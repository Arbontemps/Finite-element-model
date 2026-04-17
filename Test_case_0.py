#%%
"""
Arthur Bontemps
Test case 0 : Application on a clamped beam loaded at one end and both 1 and 3 elements 
_1 : 1 element model; _3: 3 elements model
"""

import numpy as np
import matplotlib.pyplot as plt
from AB_BV_fonctions_assemblage import *
from FEM_functions import *


#%% Arbitrary geometry and loading: 50 kg (N, mm, MPa)
L = 1500
hb = 100
eb = 20
Eb = 10e3
Ib = eb*hb**3/12
Sb = eb * hb
f = 50*9.81 

# mesh definition
x_nodes_1 = np.linspace(0, L, 2)
x_nodes_3 = np.linspace(0, L, 4)
ndof_per_node = 3
ndof_per_elem = ndof_per_node * 2
nelem_1, nelem_3 = 1, 3 
nnode_1, nnode_3 = nelem_1 + 1, nelem_3 + 1
ndof_1, ndof_3 = ndof_per_node * nnode_1, ndof_per_node * nnode_3
l_elem = L / nelem_3


# global force vector definition 
Ftot_1 = np.array([0, -f, -f*L, 0, f, 0])
Ftot_3 = np.array([0]*ndof_3)
Ftot_3[1:3], Ftot_3[-2] = np.array([-f, -f*L]), f

# Stiffness matrix element and assembly the 3 element model 
kelem_1 = matrix_kB(L, Eb, hb, eb, ndof_per_elem)
kelem_3 = matrix_kB(l_elem, Eb, hb, eb, ndof_per_elem)
list_elem_3 = [{'dof_elem':[0,1,2,3,4,5], 'kelem':kelem_3}, {'dof_elem':[3,4,5,6,7,8], 'kelem':kelem_3}, {'dof_elem':[6,7,8,9,10,11], 'kelem':kelem_3}]
Ktot_3 = assemblage_K(nelem_3, list_elem_3, ndof_3, ndof_per_elem)

# apply BC on Ktot and Ftot
BC = [0, 1, 2] # dof number that is equal to 0 
Ktot_1, Ftot_1 = apply_dirichlet_BC(kelem_1, Ftot_1, BC)
Ktot_3, Ftot_3 = apply_dirichlet_BC(Ktot_3, Ftot_3, BC)

# Compute displacement
Utot_1 = np.linalg.solve(Ktot_1, Ftot_1)
Utot_3 = np.linalg.solve(Ktot_3, Ftot_3)

# prepare the plots by extraction the deflection along the beam with form functions 
xtot_1, xtot_3 =  np.linspace(0, L, 100), [0]
vtot_1, vtot_3 = [-U_NA(x, L, Utot_1)[1] for x in xtot_1], [0]
for i in range(nelem_3):
    uelem = Utot_3[list_elem_3[i]['dof_elem']]
    xelem = np.linspace(0, l_elem, 10)
    xtot_3.extend(xelem+xtot_3[-1])
    for x in xelem:
        v_3 = U_NA(x, l_elem, uelem)[1]
        vtot_3.append(-v_3)
vtot_1, vtot_3 = np.array(vtot_1), np.array(vtot_3)
v_true = f*L**3/(3*Eb*Ib)

print('v true (mm) = ', -v_true, '\n''v calc 1 elem (mm) = ', vtot_1[-1], '\n''v calc 3 elem (mm) = ', vtot_3[-1])
print('The exact solution is found with 1 element as the form functions are able to reconstruct the kinematic')
plt.plot(xtot_1, vtot_1, color='blue', label='1 elem')
plt.plot(xtot_3, vtot_3, color='red', label='3 elem')
plt.scatter(x_nodes_1, [0, vtot_1[-1]], marker='.', c='red', s=50, label='1 elem nodes')
plt.scatter(x_nodes_3, [vtot_3[np.where(np.array(xtot_3)==x)[0]][0] for x in x_nodes_3], \
                                        marker='s', c='blue', s=50, label='3 elem nodes')
plt.scatter(x_nodes_1[-1], -v_true, marker=(5,1), s=100, color='black', label='Exact solution')
plt.xlabel('x-position (mm)')
plt.ylabel('Deflection (mm)')
plt.legend()
plt.show()
# %%
