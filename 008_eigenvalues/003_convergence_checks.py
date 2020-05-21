import pickle

import numpy as np
from scipy.constants import c as clight
import scipy.io as sio

import PyECLOUD.mystyle as ms

pkl_fname = 'mode_coupling_matrix.pkl'

l_min = -7
l_max = 7
m_max = 39
N_max = 49
abs_min_imag_unstab = 1.

# Plot settings
l_min_plot = -5
l_max_plot = 3
min_strength_plot = 0
max_strength_plot = 2.0
tau_min_plot = 0
tau_max_plot = 300

with open(pkl_fname, 'rb') as fid:
    MM_orig = pickle.load(fid)

MM_obj = MM_orig.get_sub_matrix(l_min, l_max, m_max, N_max)

omega0 = 2*np.pi*clight/27e3 # Revolution angular frquency
omega_s = 4.9e-3*omega0

# Mode coupling test
strength_scan = np.arange(min_strength_plot, max_strength_plot, 0.01)
Omega_mat = MM_obj.compute_mode_complex_freq(omega_s, rescale_by=strength_scan)

i_l0 = np.argmin(np.abs(MM_obj.l_vect))
M00_array = MM_obj.MM[i_l0,0,i_l0,0] * strength_scan

sio.savemat('eigenvalues.mat', {
    'Omega_mat': Omega_mat,
    'strength_scan': strength_scan,
    'DQ_0_vect': 0*strength_scan,
    'M00_array': M00_array,
    'omega0': omega0,
    'omega_s': omega_s,
    'l_min': l_min,
    'l_max': l_max,
    'm_max': m_max,
    'N_max': N_max})


import matplotlib.pyplot as plt
plt.close('all')
ms.mystyle(fontsz=13, traditional_look=False)

mask_unstable = np.imag(Omega_mat) < -abs_min_imag_unstab
Omega_mat_unstable = Omega_mat.copy()
Omega_mat_unstable[~mask_unstable] = np.nan+1j*np.nan

i_mode = -1
mask_mode = np.abs(np.real(Omega_mat)/omega_s - i_mode)<0.5
plt.plot(strength_scan, np.imag(Omega_mat), '.b')
plt.plot(strength_scan, np.imag(Omega_mat), '.b')
Omega_mat_mode = Omega_mat.copy()
Omega_mat_mode[~mask_mode] = np.nan

title = f'l_min={l_min}, l_max={l_max}, m_max={m_max}, N_max={N_max}'

figre = plt.figure(200)
plt.plot(strength_scan, np.real(Omega_mat)/omega_s, '.b')
plt.plot(strength_scan, np.real(Omega_mat_unstable)/omega_s, '.r')
plt.suptitle(title)
plt.grid(True, linestyle=':', alpha=.8)
plt.xlabel('Strength')
plt.ylabel(r'Re($\Omega$)/$\omega_s$')
plt.subplots_adjust(bottom=.12)

figim = plt.figure(201)
plt.plot(strength_scan, np.imag(Omega_mat), '.b')
plt.plot(strength_scan, np.imag(Omega_mat_unstable), '.r')
plt.suptitle(title)
plt.grid(True, linestyle=':', alpha=.8)
plt.xlabel('Strength')
plt.ylabel(r'Im($\Omega$)')
plt.subplots_adjust(bottom=.12)

plt.figure(300)
plt.plot(np.imag(Omega_mat).flatten(),
        np.real(Omega_mat).flatten()/omega_s, '.b')
plt.suptitle(title)


plt.figure(400)
for ii in range(len(strength_scan)):
    plt.scatter(x=strength_scan[ii]+0*np.imag(Omega_mat[ii, :]),
            y=np.imag(Omega_mat[ii, :]),
            c = np.real(Omega_mat[ii, :])/omega_s,
            cmap=plt.cm.seismic)
plt.suptitle(title)
plt.colorbar()

fig500 = plt.figure(500)#, figsize=(1.3*6.4, 1.3*4.8))
ax = fig500.add_subplot(111)
ax.set_facecolor('grey')
im_min_col = 5
im_max_col = 200
im_min_size = 5
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
    ax.plot(strength_scan, 0*strength_scan+lll, color='w',
            alpha=.5, linestyle='--')
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

