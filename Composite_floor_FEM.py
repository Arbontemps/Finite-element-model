#%%
""" FEM resolution of a loaded beam"""
import numpy as np
import matplotlib.pyplot as plt
from BV_fonctions_LL_BB import list_dof_l, matrices_LL_BB, vecteur_LL_BB
from AB_BV_fonctions_assemblage import *
from FEM_functions import *
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
""" Try to calculate the stiffness gain using beam+spring (_bs), double layer beam (_db) and double layer beam + interface (_dbi)
Mesh definition: one element is defined between two connectors -0.13556292869565217"""
f = 1000 # load in N
l_el = L_es

# Meshing : define the dof, nodes and elements
ndof_per_node = 3
nnodes_per_el_bs, nnodes_per_el_db = 2, 4
ndof_per_el_bs, ndof_per_el_db = nnodes_per_el_bs * ndof_per_node, nnodes_per_el_db * ndof_per_node
nnodes_bs, nnodes_db = N_c, 2 * N_c
nel = int(nnodes_db/2 - 1)
ndof_bs, ndof_db = nnodes_bs * ndof_per_node, nnodes_db * ndof_per_node
list_dof_bs, list_dof_db = np.linspace(0, ndof_bs - 1, ndof_bs, dtype=int), np.linspace(0, ndof_db - 1, ndof_db, dtype=int)
L =  l_el * nel

# definition of the stiffness element matrix 
kel_bs = matrix_ke(l_el, E_b, h_b, e_b, ndof_per_el_bs) + matrix_Kr(k_es, exc) 
kel_db = true_K2e_matrix(E_b, S_b, IGz_b, E_es, S_es, IGz_es, l_el, ndof_per_el_db) 
kel_dbi = true_K2e_matrix(E_b, S_b, IGz_b, E_es, S_es, IGz_es, l_el, ndof_per_el_db) + matrix_Kinter(k_c, h_b, h_es, min(e_b, e_es), l_el, ndof_per_el_db)

# load connectivity tables for assembly
TC_elnodes_bs, TC_ndof_bs, TC_nloc_bs, local_dof_bs = connectivity(nnodes_per_el_bs, nel, ndof_per_node, 'monolayer')
TC_elnodes_db, TC_ndof_db, TC_nloc_db, local_dof_db = connectivity(nnodes_per_el_db, nel, ndof_per_node, 'bilayer')
list_el_bs, list_el_db, list_el_dbi = [{'dof_el':np.array([0,1,2,3,4,5]), 'Kel':kel_bs}], [], []
for i in range(nel):
    nodes = TC_elnodes_db[i]
    dof_db = [TC_ndof_db[nodes[j]][k] for j in range(len(nodes)) for k in range(len(TC_ndof_db[nodes[j]]))]
    list_el_db.append({'dof_el':local_dof_db+dof_db[0], 'Kel': kel_db})
    list_el_dbi.append({'dof_el':local_dof_db+dof_db[0], 'Kel': kel_dbi})
    if i < nel - 1:
        list_el_bs.append({'dof_el':list_el_bs[i]['dof_el']+3, 'Kel':kel_bs})

# assembly
Ktot_bs = assemblage_K(nel, list_el_bs, ndof_bs, ndof_per_el_bs)
Ktot_db = assemblage_K(nel, list_el_db, ndof_db, ndof_per_el_db)
Ktot_dbi = assemblage_K(nel, list_el_dbi, ndof_db, ndof_per_el_db)

# Boundary conditions: dof number that is equal to 0
BC_bs, BC_db = [0, 1, ndof_bs-2], [0, 1, 27, ndof_db-5]

# constraints: dof equality for the reduced system (because dof linked between the bot and top layer)
cons_db = constraints(list_dof_db)

# reduce global stiffness matrix to the constrained Ktot_tilde (because dof linked between the bot and top layer)
T, indep = matrix_T(ndof_db, cons_db)
BC_db_red = map_BC_to_reduced(BC_db, indep)
Ktot_tilde_db =  T.T @ Ktot_db @ T
Ktot_tilde_dbi = T.T @ Ktot_dbi @ T

# definition of the global force vector
Ftot_bs, Ftot_db = np.zeros(ndof_bs), np.zeros(ndof_db)
Ftot_bs[1], Ftot_bs[10], Ftot_bs[16], Ftot_bs[25] = -f/2, f/2, f/2, -f/2
Ftot_db[1], Ftot_db[19], Ftot_db[31], Ftot_db[49] = -f/2, f/2, f/2, -f/2
Ftot_tilde_db = T.T @ Ftot_db

# add BC on the matrix system
Ktot_bs, Ftot_bs = apply_dirichlet_BC(Ktot_bs, Ftot_bs, BC_bs)
Ktot_tilde_db, Ftot_tilde_db = apply_dirichlet_BC(Ktot_tilde_db, Ftot_tilde_db, BC_db_red)
Ktot_tilde_dbi, _ = apply_dirichlet_BC(Ktot_tilde_dbi, Ftot_tilde_db, BC_db_red)

# Resolution and reconstruction
Utot_bs = np.linalg.solve(Ktot_bs, Ftot_bs)
Utot_tilde_db = np.linalg.solve(Ktot_tilde_db, Ftot_tilde_db)
Utot_tilde_dbi = np.linalg.solve(Ktot_tilde_dbi, Ftot_tilde_db)
Utot_db = T @ Utot_tilde_db
Utot_dbi = T @ Utot_tilde_dbi

# prepare the plots by extraction the deflection along the beam with form functions 
xtot =  [0]
vtot_bs, vtot_db, vtot_dbi = [0], [0], [0]
for i in range(nel):
    uel_bs, uel_db, uel_dbi = Utot_bs[list_el_bs[i]['dof_el']], Utot_db[list_el_db[i]['dof_el']], Utot_dbi[list_el_dbi[i]['dof_el']]
    xel = np.linspace(0, l_el, 100)
    xtot.extend(xel+xtot[-1])
    for x in xel:
        v_bs, v_db, v_dbi = U0(x, l_el, uel_bs)[1], U0_bilayer(x, l_el, uel_db)[1], U0_bilayer(x, l_el, uel_dbi)[1]
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
