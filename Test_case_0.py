#%%
"""
Arthur Bontemps
Test case 0 : Application on a clamped beam loaded at one end and both 1 and 3 elements 
_1 : 1 element model; _3: 3 elements model
"""

import numpy as np
import matplotlib.pyplot as plt
from BV_fonctions_LL_BB import list_dof_l, matrices_LL_BB
from BV_fonctions_assemblage import *
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
ndof = 3
eldof = ndof * 2
nel1, nel3 = 1, 3 
nnode_1, nnode_3 = nel1 + 1, nel3 + 1
el1dof_tot, el3dof_tot = ndof * nnode_1, ndof * nnode_3
l_el = L / nel3

# global force vector definition 
Ftot_1 = np.array([0, -f, -f*L, 0, f, 0])
Ftot_3 = np.array([0]*el3dof_tot)
Ftot_3[1:3], Ftot_3[-2] = np.array([-f, -f*L]), f

# Stiffness matrix element and assembly the 3 element model 
kelem_1 = matrix_ke(L, Eb, hb, eb, eldof)
kelem_3 = matrix_ke(l_el, Eb, hb, eb, eldof)
list_el_3 = [{'dof_el':[0,1,2,3,4,5], 'Kel':kelem_3}, {'dof_el':[3,4,5,6,7,8], 'Kel':kelem_3}, {'dof_el':[6,7,8,9,10,11], 'Kel':kelem_3}]
Ktot_3 = assemblage_K(nel3, list_el_3, el3dof_tot, eldof)

# apply BC on Ktot and Ftot
BC = [0, 1, 2] # dof number that is equal to 0 
Ktot_1, Ftot_1 = apply_dirichlet_BC(kelem_1, Ftot_1, BC)
Ktot_3, Ftot_3 = apply_dirichlet_BC(Ktot_3, Ftot_3, BC)

# Compute displacement
Utot_1 = np.linalg.solve(Ktot_1, Ftot_1)
Utot_3 = np.linalg.solve(Ktot_3, Ftot_3)

# prepare the plots by extraction the deflection along the beam with form functions 
xtot_1, xtot_3 =  np.linspace(0, L, 100), [0]
vtot_1, vtot_3 = [-U0(x, L, Utot_1)[1] for x in xtot_1], [0]
for i in range(nel3):
    uel = Utot_3[list_el_3[i]['dof_el']]
    xel = np.linspace(0, l_el, 10)
    xtot_3.extend(xel+xtot_3[-1])
    for x in xel:
        v_3 = U0(x, l_el, uel)[1]
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