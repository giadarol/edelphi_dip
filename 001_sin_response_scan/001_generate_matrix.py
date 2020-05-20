import PyECLOUD.myfilemanager as mfm

import numpy as np

plane = 'y'
folder_sims = 'simulations_' + plane

N_samples = 200
ref_ampl = 1e-4
assert(N_samples % 2 ==0)

cos_ampl_list = []
sin_ampl_list = []
n_osc_list = []

for ii in range(N_samples//2):
    cos_ampl_list.append(ref_ampl)
    sin_ampl_list.append(0.)
    n_osc_list.append(ii)

    cos_ampl_list.append(0.)
    sin_ampl_list.append(ref_ampl)
    n_osc_list.append(ii+1)

r_meas_mat = []
r_mat = []
dpr_mat = []
for ii in range(len(n_osc_list)):

    cos_ampl = cos_ampl_list[ii]
    sin_ampl = sin_ampl_list[ii]
    n_osc = n_osc_list[ii]

    current_sim_ident= f'n_{n_osc:.1f}_c{cos_ampl:.2e}_s{sin_ampl:.2e}'
    ob = mfm.myloadmat_to_obj(folder_sims + '/' + current_sim_ident + '/response.mat')

    r_mat.append(ob.r_ideal)
    r_meas_mat.append(ob.r_slices)
    dpr_mat.append(ob.dpr_slices_all_clouds)

z_slices = ob.z_slices

import scipy.io as sio
sio.savemat(f'response_data_{plane}.mat',{
    'r_mat': r_mat,
    'z_slices': z_slices,
    'dpr_mat': dpr_mat,
    'sin_ampl_list':np.array(sin_ampl_list),
    'cos_ampl_list':np.array(cos_ampl_list),
    'n_osc_list':np.array(n_osc_list),
    })


r_mat = np.array(r_mat)

r_mat[np.isnan(r_mat)] = 0.

f_mat = r_mat

N_base = f_mat.shape[0]

w_mat = f_mat

M_mat = np.dot(f_mat, w_mat.T)

r_test = 5e-3 * z_slices

b_test = np.dot(w_mat, r_test.T)

a_test = np.linalg.solve(M_mat, b_test)

r_check = np.dot(a_test, f_mat)
b_check = np.dot(f_mat, r_check.T)
