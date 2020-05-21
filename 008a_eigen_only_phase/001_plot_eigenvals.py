import pickle

import numpy as np
from scipy.constants import c as clight
import scipy.io as sio

import PyECLOUD.mystyle as ms

strength_scan = np.arange(0., 2., 0.01)[1::]

omega0 = 2*np.pi*clight/27e3 # Revolution angular frquency
omega_s = 4.9e-3*omega0
simulation_folder = './simulations_fine'
pkl_fname = 'mode_coupling_matrix.pkl'

l_min = -7
l_max = 7
m_max = 39
N_max = 49
min_imag_unstab = 1.
flag_correct_tune = False
flag_tilt_lines = False
temp_rescale_DQ = .85

# Plot settings
l_min_plot = -5
l_max_plot = 3
min_strength_plot = 0
max_strength_plot = 1.55
tau_min_plot = 0
tau_max_plot = 300

Omega_mat = []
DQ_0_list = []
M00_list = []
for ii in range(0, len(strength_scan)):
    print(f'{ii}/{len(strength_scan)}')
    pkl_fname = simulation_folder+(f'/strength_{strength_scan[ii]:.3f}'
        '/mode_coupling_matrix.pkl')
    with open(pkl_fname, 'rb') as fid:
        MM_orig = pickle.load(fid)

    MM_obj = MM_orig.get_sub_matrix(l_min, l_max, m_max, N_max)

    Omega_array = MM_obj.compute_mode_complex_freq(omega_s)
    if flag_correct_tune or flag_tilt_lines:
        if len(MM_obj.alpha_p)>0:
            print('alpha_p:')
            print(MM_obj.alpha_p)
            DQ_0 = -(MM_obj.beta_fun_rescale
                    * MM_obj.alpha_p[0] /(4*np.pi)*temp_rescale_DQ)
            if flag_correct_tune:
                Omega_array += DQ_0 * MM_obj.omega0
            DQ_0_list.append(DQ_0)
    Omega_mat.append(Omega_array)
    i_l0 = np.argmin(np.abs(MM_obj.l_vect))
    M00_list.append(MM_obj.MM[i_l0,0,i_l0,0])

Omega_mat = np.array(Omega_mat)
import matplotlib.pyplot as plt
plt.close('all')
ms.mystyle(fontsz=14, traditional_look=False)

mask_unstable = np.imag(Omega_mat) > min_imag_unstab
Omega_mat_unstable = Omega_mat.copy()
Omega_mat_unstable[~mask_unstable] = np.nan+1j*np.nan

i_mode = -1
mask_mode = np.abs(np.real(Omega_mat)/omega_s - i_mode)<0.5
plt.plot(strength_scan, np.imag(Omega_mat), '.b')
plt.plot(strength_scan, np.imag(Omega_mat), '.b')
Omega_mat_mode = Omega_mat.copy()
Omega_mat_mode[~mask_mode] = np.nan

title = f'l_min={l_min}, l_max={l_max}, m_max={m_max}, N_max={N_max}'


plt.figure(200)
plt.plot(strength_scan, np.real(Omega_mat)/omega_s, '.b')
plt.plot(strength_scan, np.real(Omega_mat_unstable)/omega_s, '.r')
plt.suptitle(title)

plt.figure(201)
plt.plot(strength_scan, np.imag(Omega_mat), '.b')
plt.plot(strength_scan, np.imag(Omega_mat_mode), '.g')
plt.suptitle(title)

plt.figure(300)
plt.plot(np.imag(Omega_mat).flatten(),
        np.real(Omega_mat).flatten()/omega_s, '.b')
plt.suptitle(title)


fig400 = plt.figure(400)
ax = fig400.add_subplot(111)
ax.set_facecolor('grey')
for ii in range(len(strength_scan)):
    plt.scatter(x=strength_scan[ii]+0*np.imag(Omega_mat[ii, :]),
            y=np.imag(Omega_mat[ii, :]),
            c = np.real(Omega_mat[ii, :])/omega_s,
            cmap=plt.cm.seismic, s=3)
plt.suptitle(title)
plt.colorbar()

sio.savemat('eigenvalues.mat', {
    'Omega_mat': Omega_mat,
    'strength_scan': strength_scan,
    'DQ_0_vect': np.array(DQ_0_list),
    'M00_array': np.array(M00_list),
    'omega0': omega0,
    'omega_s': omega_s,
    'l_min': l_min,
    'l_max': l_max,
    'm_max': m_max,
    'N_max': N_max})

# fig500 = plt.figure(500, figsize=(1.3*6.4, 1.3*4.8))
# ax = fig500.add_subplot(111)
# ax.set_facecolor('grey')
# im_min_col = 0.1
# im_max_col = 100
# im_min_size = 5
# im_max_size = 50
# import matplotlib
# for ii in range(len(strength_scan)):
#     Omega_ii = Omega_mat[ii, :]
#     ind_sorted = np.argsort(-np.imag(Omega_ii))
#     re_sorted = np.take(np.real(Omega_ii), ind_sorted)
#     im_sorted = np.take(np.imag(Omega_ii), ind_sorted)
#     plt.scatter(x=strength_scan[ii]+0*np.imag(Omega_mat[ii, :]),
#             y=re_sorted/omega_s,
#             c = np.clip(-im_sorted, im_min_col, im_max_col),
#             cmap=plt.cm.jet,
#             s=np.clip(-im_sorted, im_min_size, im_max_size),
#             vmin=im_min_col, vmax=im_max_col,
#             norm=matplotlib.colors.LogNorm())
# plt.suptitle(title)
# plt.colorbar()

fig500 = plt.figure(500)#, figsize=(1.3*6.4, 1.3*4.8))
ax = fig500.add_subplot(111)
ax.set_facecolor('grey')
im_min_col = 5
im_max_col = 200
im_min_size = 10
im_max_size = 50
import matplotlib
for ii in range(len(strength_scan)):
    Omega_ii = Omega_mat[ii, :]
    ind_sorted = np.argsort(-np.imag(Omega_ii))
    re_sorted = np.take(np.real(Omega_ii), ind_sorted)
    im_sorted = np.take(np.imag(Omega_ii), ind_sorted)
    plt.scatter(x=strength_scan[ii]+0*np.imag(Omega_mat[ii, :]),
            y=re_sorted/omega_s,
            c = np.clip(-im_sorted, im_min_col, im_max_col),
            cmap=plt.cm.jet,
            s=np.clip(-im_sorted, im_min_size, im_max_size),
            vmin=im_min_col, vmax=im_max_col,
            norm=matplotlib.colors.LogNorm())
plt.suptitle(title)
ax.set_xlim(min_strength_plot, max_strength_plot)
ax.set_ylim(l_min_plot, l_max_plot)
fig500.subplots_adjust(right=1.)
for lll in range(l_min_plot-10, l_max_plot+10):
    if flag_tilt_lines:
        add_to_line = np.array(DQ_0_list)*omega0/omega_s
    else:
        add_to_line = 0.
    ax.plot(strength_scan,
            0*strength_scan + lll + add_to_line,
            color='w',
            alpha=.5, linestyle='--')
ax.tick_params(right=True, top=True)
plt.colorbar()

figtau = plt.figure(600)
axtau = figtau.add_subplot(111)
axtau.plot(strength_scan, np.imag(Omega_mat),
        '.', color='grey', alpha=0.5)
axtau.plot(strength_scan, np.max(-np.imag(Omega_mat), axis=1),
        linewidth=2, color='b')
axtau.set_xlim(min_strength_plot, max_strength_plot)
axtau.set_ylim(tau_min_plot, tau_max_plot)
axtau.grid(True, linestyle=':')
figtau.suptitle(title)

plt.show()

