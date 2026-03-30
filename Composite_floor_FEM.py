#%%
""" FEM resolution of a loaded beam"""
import numpy as np
import matplotlib.pyplot as plt
from BV_fonctions_assemblage import assemblage_K, connectivity
from BV_fonctions_LL_BB import list_dof_l, matrices_LL_BB, vecteur_LL_BB, reorganisation_dof_vec
from FEM_functions import matrices_N, true_Ke_matrix, matrix_Kr, matrix_K2e, matrix_Kc, U0, matrix_ke
from Analytical_beam_formulations import beam_4pt_deflection
#%%
""" Geometry and material properties (MPa, N, mm) """
# Wooden beam (double section because 2 beams in 1 floor element) -> index b
L_b = 3000
h_b = 150
e_b = 50*2
S_b = e_b*h_b
IGz_b = (e_b*h_b**3)/12
E_b = 11500

# Earth block and wooden sleepers -> index es
L_es = 300 + 63 # this also define the length of one element
h_es = 100
e_es = 450 
S_es = e_es*h_es
IGz_es = (e_es*h_es**3)/12
E_es = 1150 
k_es = E_es*S_es/L_es

# Connectors -> index c
N_c = 9 # total number of connector / 2 as there is 2 connectors per sleepers
k_c = 13000 # stiffness of 2 connectors in N/mm
exc = h_b/2 + h_es/2 # excentricity of the spring modelizing earth-wooden sleepers element (at its gravity center)

#%%
""" Application on a clamped beam loaded at one end and both 1 and 3 elements """
# Length, node vector and load definition: 50 kg at one end and clamped at the other end
L, f = 1500, 50*9.81 # 
x_nodes_el1 = np.linspace(0, L, 2)
x_nodes_el3 = np.linspace(0, L, 4)
ndofnode = 3
ndofel = ndofnode * 2
n_el1, n_el3 = 1, 3 
nnode_el1, nnode_el3 = n_el1 + 1, n_el3 + 1
ndof_el1, ndof_el3 = ndofnode * nnode_el1, ndofnode * nnode_el3
list_ndof_el1, list_ndof_el3 = np.linspace(0, ndof_el1 - 1,ndof_el1, dtype=int), np.linspace(0, ndof_el3 - 1,ndof_el3, dtype=int)
Fext_el1 = np.array([0, -f, -f*L, 0, f, 0])
Fext_el3 = np.array([0]*ndof_el3)
Fext_el3[1:3], Fext_el3[-2] = np.array([-f, -f*L]), f
l_el = L / n_el3
# Boundary conditions
BC = [0, 1, 2] # dof number that is equal to 0 
dof_L_el1, dof_L_el3 = list_dof_l(BC, ndof_el1)[0], list_dof_l(BC, ndof_el3)[0] # free dof number
# Stiffness matrix element and assembly
k_el1 = matrix_ke(L, E_b, h_b, e_b, ndofel)
k_el3 = matrix_ke(l_el, E_b, h_b, e_b, ndofel)

list_el3 = [{'dof_el':[0,1,2,3,4,5], 'Kel':k_el3}, {'dof_el':[3,4,5,6,7,8], 'Kel':k_el3}, {'dof_el':[6,7,8,9,10,11], 'Kel':k_el3}]
Ktot_el3 = assemblage_K(n_el3, list_el3, ndof_el3, ndofel)
Ktot_el1_LL = matrices_LL_BB(k_el1, dof_L_el1, BC)[0]
Ktot_el3_LL = matrices_LL_BB(Ktot_el3, dof_L_el3, BC)[0]
# Compute displacement
U_el1 = np.array([0.]*ndof_el1)
U_el3 = np.array([0.]*ndof_el3)
U_el1[np.delete(list_ndof_el1, BC)] = np.linalg.solve(Ktot_el1_LL, Fext_el1[np.delete(list_ndof_el1, BC)])
U_el3[np.delete(list_ndof_el3, BC)] = np.linalg.solve(Ktot_el3_LL, Fext_el3[np.delete(list_ndof_el3, BC)])

xtot_el1, xtot_el3 =  np.linspace(0, L, 100), [0]
vtot_el1, vtot_el3 = [-U0(x, L, U_el1)[1] for x in xtot_el1], [0]
for i in range(n_el3):
    uel = U_el3[list_el3[i]['dof_el']]
    xel = np.linspace(0, l_el, 10)
    xtot_el3.extend(xel+xtot_el3[-1])
    for x in xel:
        v_el3 = U0(x, l_el, uel)[1]
        vtot_el3.append(-v_el3)
vtot_el1, vtot_el3 = np.array(vtot_el1), np.array(vtot_el3)
v_true = f*L**3/(3*E_b*IGz_b)

print('v true (mm) = ', -v_true, '\n''v calc 1 elem (mm) = ', vtot_el1[-1], '\n''v calc 3 elem (mm) = ', vtot_el3[-1])
print('The exact solution is found with 1 element as the form functions are able to reconstruct the kinematic')
plt.plot(xtot_el1, vtot_el1, color='blue', label='1 elem')
plt.plot(xtot_el3, vtot_el3, color='red', label='3 elem')
plt.scatter(x_nodes_el1, [0, vtot_el1[-1]], marker='.', c='red', s=50, label='1 elem nodes')
plt.scatter(x_nodes_el3, [vtot_el3[np.where(np.array(xtot_el3)==x)[0]][0] for x in x_nodes_el3], \
                                        marker='s', c='blue', s=50, label='3 elem nodes')
plt.scatter(x_nodes_el1[-1], -v_true, marker=(5,1), s=100, color='black', label='Exact solution')
plt.xlabel('x-position (mm)')
plt.ylabel('Deflection (mm)')
plt.legend()
plt.show()

#%%
""" Try to calculate the stiffness gain using beam+spring (_bs), double layer beam (_db) and double layer beam + interface (_dbi)
Mesh definition: one element is defined between two connectors -0.13556292869565217"""
f = 1000 # load in N
l_el = L_es
# Define the dof of both bot (_b) and top (_t) layer
ndofnodes_b = 3
ndofnodes_t = 1
ndofel_bs, ndofel_db = ndofnodes_b * 2, (ndofnodes_b + ndofnodes_t) * 2
nnodes_b, nnodes_t = N_c, N_c 
nel = N_c - 1
ndof_bs, ndof_db = nnodes_b * ndofnodes_b, (nnodes_b * ndofnodes_b) + (nnodes_t * ndofnodes_t)
dof_bs, dof_db = np.linspace(1, ndof_bs, ndof_bs), np.linspace(1, ndof_db, ndof_db)
list_ndof_bs, list_ndof_db = np.linspace(0, ndof_bs-1, ndof_bs, dtype=int), np.linspace(0, ndof_db - 1, ndof_db, dtype=int)
L = l_el * nel
x_nodes = np.linspace(0, L, N_c) # position of each nodes in the global reference frame with one end as origin (in mm)
# Define load vector and BC for 4-pts bending
BC_bs, BC_db = [0, 1, ndof_bs-2], [0, 1, 19, ndof_db-3]
F_bs, F_db = np.array([0]*ndof_bs), np.array([0]*ndof_db)
F_bs[1], F_bs[10], F_bs[16], F_bs[25] = -f/2, f/2, f/2, -f/2
F_db[1], F_db[13], F_db[21], F_db[33] = -f/2, f/2, f/2, -f/2

#k_c = 1e10
kel_bs = matrix_ke(l_el, E_b, h_b, e_b, ndofel_bs) + matrix_Kr(k_es, exc) 
kel_db = matrix_K2e(E_b, h_b, e_b, E_es, h_es, e_es, l_el)
kel_dbi = matrix_K2e(E_b, h_b, e_b, E_es, h_es, e_es, l_el) + matrix_Kc(k_c, h_b, h_es, 0, l_el)

list_el_bs, list_el_db, list_el_dbi = [{'dof_el':np.array([0,1,2,3,4,5]), 'Kel':kel_bs}], [], []
TC_nodes_db, TC_dof_db = connectivity()

for i in range(nel):
    nodes = TC_nodes_db[i]
    dof_db = [TC_dof_db[nodes[j]][k] for j in range(len(nodes)) for k in range(len(TC_dof_db[nodes[j]]))]
    list_el_db.append({'dof_el':dof_db, 'Kel': kel_db})
    list_el_dbi.append({'dof_el':dof_db, 'Kel': kel_dbi})
    if i < nel - 1:
        list_el_bs.append({'dof_el':list_el_bs[i]['dof_el']+3, 'Kel':kel_bs})

Ktot_bs, Ktot_db, Ktot_dbi = assemblage_K(nel, list_el_bs, ndof_bs, ndofel_bs), \
                             assemblage_K(nel, list_el_db, ndof_db, ndofel_db), \
                             assemblage_K(nel, list_el_dbi, ndof_db, ndofel_db)

dof_L_bs, dof_B_bs  = list_dof_l(BC_bs, ndof_bs)[:-1]
dof_L_db, dof_B_db = list_dof_l(BC_db, ndof_db)[:-1]
Ktot_bs_LL, Ktot_db_LL, Ktot_dbi_LL = matrices_LL_BB(Ktot_bs, dof_L_bs, BC_bs)[0], \
                                      matrices_LL_BB(Ktot_db, dof_L_db, BC_db)[0], \
                                      matrices_LL_BB(Ktot_dbi, dof_L_db, BC_db)[0]
F_bs_LL, F_db_LL = vecteur_LL_BB(F_bs, dof_L_bs, dof_B_bs)[0], vecteur_LL_BB(F_db, dof_L_db, dof_B_db)[0]

U_bs, U_db, U_dbi = np.zeros(ndof_bs), np.zeros(ndof_db), np.zeros(ndof_db)
U_bs_LL, U_db_LL, U_dbi_LL = np.linalg.solve(Ktot_bs_LL, F_bs_LL), np.linalg.solve(Ktot_db_LL, F_db_LL), np.linalg.solve(Ktot_dbi_LL, F_db_LL)
U_bs[np.delete(list_ndof_bs, BC_bs)] = U_bs_LL
U_db[np.delete(list_ndof_db, BC_db)] = U_db_LL
U_dbi[np.delete(list_ndof_db, BC_db)] = U_dbi_LL

xtot =  [0]
vtot_bs, vtot_db, vtot_dbi = [0], [0], [0]
for i in range(nel):
    uel_bs, uel_db, uel_dbi = U_bs[list_el_bs[i]['dof_el']], U_db[list_el_db[i]['dof_el']], U_dbi[list_el_dbi[i]['dof_el']]
    xel = np.linspace(0, l_el, 10)
    xtot.extend(xel+xtot[-1])
    for x in xel:
        v_bs, v_db, v_dbi = U0(x, l_el, uel_bs)[1], U0(x, l_el, uel_db[:-2])[1], U0(x, l_el, uel_dbi[:-2])[1]
        vtot_bs.append(-v_bs)
        vtot_db.append(-v_db)
        vtot_dbi.append(-v_dbi)
vtot_bs, vtot_db, vtot_dbi = np.array(vtot_bs), np.array(vtot_db), np.array(vtot_dbi)
v_analytic = []
for x in xtot:
    v_analytic.append(beam_4pt_deflection(x, f, E_b, IGz_b, L, 1089, 1815))
plt.plot(xtot, v_analytic, c='black', label='Analytical results with bot layer only')
plt.plot(xtot, vtot_bs, color='blue', label='Beam + offset spring')
plt.plot(xtot, vtot_db, color='red', label='2-unconnected layer')
plt.plot(xtot, vtot_dbi, color='orange', label='2-connected layer (kc=13kN/mm)')
plt.xlabel('x-position (mm)')
plt.ylabel('Deflection (mm)')
plt.legend()
#plt.savefig('FEM_Results_04-03-2026.jpg')
plt.show()
gain_bs = np.min(v_analytic)/np.min(vtot_bs)
gain_dbi = np.min(v_analytic)/np.min(vtot_dbi)
print('v analytic / v beam+spring (gain_bs) = ', '{:.4}'.format(gain_bs))
print('v analytic / v 2-connected layer (gain_dbi) = ', '{:.4}'.format(gain_dbi))


#%%
""" Use the same 4-points bending definition and compare to experimental results"""
import pandas as pd
cbtc3 = pd.read_excel('CBTC3.xlsx', sheet_name='Comparaison modèle FEM')
l_el = L_es
# Define the dof of both bot (_b) and top (_t) layer
ndofnodes_b, ndofnodes_t = 3, 1
ndofel_bois, ndofel_collab = ndofnodes_b * 2, (ndofnodes_b + ndofnodes_t) * 2
nnodes_b, nnodes_t = N_c, N_c 
nel = N_c - 1
ndof_bois, ndof_collab = nnodes_b * ndofnodes_b, (nnodes_b * ndofnodes_b) + (nnodes_t * ndofnodes_t)
dof_bois, dof_collab = np.linspace(1, ndof_bois, ndof_bois), np.linspace(1, ndof_collab, ndof_collab)
list_ndof_bois, list_ndof_collab = np.linspace(0, ndof_bois-1, ndof_bois, dtype=int), np.linspace(0, ndof_collab - 1, ndof_collab, dtype=int)
L = l_el * nel
x_nodes = np.linspace(0, L, N_c) # position of each nodes in the global reference frame with one end as origin (in mm)
# Define load vector and BC for 4-pts bending
BC_bois, BC_collab = [0, 1, ndof_bois-2], [0, 1, 19, ndof_collab-3]
kel_bois =  matrix_ke(l_el, E_b, h_b, e_b, ndofel_bois)
kel_collab = matrix_K2e(E_b, h_b, e_b, E_es, h_es, e_es, l_el) + matrix_Kc(k_c, h_b, h_es, 0, l_el)

list_el_bois, list_el_collab = [{'dof_el':np.array([0,1,2,3,4,5]), 'Kel':kel_bois}], []
TC_nodes_collab, TC_dof_collab = connectivity()

for i in range(nel):
    nodes = TC_nodes_collab[i]
    dof_collab = [TC_dof_collab[nodes[j]][k] for j in range(len(nodes)) for k in range(len(TC_dof_collab[nodes[j]]))]
    list_el_collab.append({'dof_el':dof_collab, 'Kel': kel_collab})
    if i < nel - 1:
        list_el_bois.append({'dof_el':list_el_bois[i]['dof_el']+3, 'Kel':kel_bois})

Ktot_bois, Ktot_collab = assemblage_K(nel, list_el_bois, ndof_bois, ndofel_bois), \
                             assemblage_K(nel, list_el_collab, ndof_collab, ndofel_collab)

dof_L_bois, dof_B_bois  = list_dof_l(BC_bois, ndof_bois)[:-1]
dof_L_collab, dof_B_collab = list_dof_l(BC_collab, ndof_collab)[:-1]
Ktot_bois_LL, Ktot_collab_LL = matrices_LL_BB(Ktot_bois, dof_L_bois, BC_bois)[0], \
                                      matrices_LL_BB(Ktot_collab, dof_L_collab, BC_collab)[0]

Utot_bois, Utot_collab = [], []
Uxtot_A, Uxtot_C = [], []
for i, fbois in enumerate(cbtc3['F bois'][1:]):
    fbois = fbois * 1e3
    fcollab = cbtc3['F collab'][i+1]*1e3
    Fbois, Fcollab = np.array([0]*ndof_bois), np.array([0]*ndof_collab)
    Fbois[1], Fbois[10], Fbois[16], Fbois[25] = -fbois/2, fbois/2, fbois/2, -fbois/2
    Fcollab[1], Fcollab[13], Fcollab[21], Fcollab[33] = -fcollab/2, fcollab/2, fcollab/2, -fcollab/2
    F_bois_LL, F_collab_LL = vecteur_LL_BB(Fbois, dof_L_bois, dof_B_bois)[0], vecteur_LL_BB(Fcollab, dof_L_collab, dof_B_collab)[0]
    U_bois, U_collab = np.zeros(ndof_bois), np.zeros(ndof_collab)
    U_bois_LL, U_collab_LL = np.linalg.solve(Ktot_bois_LL, F_bois_LL), np.linalg.solve(Ktot_collab_LL, F_collab_LL)
    U_bois[np.delete(list_ndof_bois, BC_bois)] = U_bois_LL
    U_collab[np.delete(list_ndof_collab, BC_collab)] = U_collab_LL
    Utot_bois.append(U_bois[13])
    Utot_collab.append(U_collab[17])
    Uxtot_A.append(U_collab[6])
    Uxtot_C.append(-U_collab[35])

fig, axs = plt.subplots(2,1,figsize=(6,6), layout='constrained')
axs[0].plot(cbtc3['U bois'][1:], cbtc3['F bois'][1:], label='exp bois', marker='.', linestyle='none')
axs[0].plot(cbtc3['U collab'][1:], cbtc3['F collab'][1:], label='exp collab', marker='.', linestyle='none')
axs[0].plot(Utot_bois, cbtc3['F bois'][1:], label='num bois')
axs[0].plot(Utot_collab, cbtc3['F collab'][1:], label='num collab')
axs[0].set_xlabel('U (mm)')
axs[0].set_ylabel('F (kN)')
axs[0].legend()

axs[1].plot(cbtc3['F collab'][1:], cbtc3['GlissCMoyen'][1:], label='gliss C')
axs[1].plot(cbtc3['F collab'][1:], cbtc3['GlissAMoyen'][1:], label='gliss A')
axs[1].plot(cbtc3['F collab'][1:], Uxtot_A, label='Ux 0')
axs[1].plot(cbtc3['F collab'][1:], Uxtot_C, label='Ux L')
axs[1].set_xlabel('F (kN)')
axs[1].set_ylabel('glissement (mm)')
axs[1].legend()

plt.show()
""" Pourquoi on a pas Uxtot_a et Uxtot_c parfaitement superposé ? Dans le modèle
la symétrie devrait les égaliser """



# %%
