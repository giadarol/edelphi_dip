import pickle

import numpy as np
import matplotlib.pyplot as plt

import PyECLOUD.myfilemanager as mfm

frac_tune_0 = .27
max_strength_plot = 0.9

fname_pic = './processed_data/compact_pic_fit.mat'
fname_vlasov_dipquad = '../008a1_scan_strength_wlampldet/eigenvalues.mat'
fname_vlasov_diponly = '../008_eigenvalues/eigenvalues.mat'

obpic = mfm.myloadmat_to_obj(fname_pic)
obdipquad = mfm.myloadmat_to_obj(fname_vlasov_dipquad)
obdiponly= mfm.myloadmat_to_obj(fname_vlasov_diponly)

i_ref = 5

stren_array = np.linspace(0, max_strength_plot, 1000)
tune_shift_dipquad = (obdipquad.M00_array[i_ref].real/obdipquad.omega0
        /obdipquad.strength_scan[i_ref]*stren_array)
tune_shift_diponly = (obdiponly.M00_array[i_ref].real/obdiponly.omega0
        /obdiponly.strength_scan[i_ref]*stren_array)

plt.close('all')
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.plot(stren_array, tune_shift_diponly, lw=3, alpha=.6, color='C1',
        label='Dipolar forces')
ax1.plot(stren_array, tune_shift_dipquad - tune_shift_diponly,
        lw=3, alpha=.6, color='C2', label='Quadrupolar forces')
ax1.plot(stren_array, tune_shift_dipquad, lw=3, alpha=.6, color='C0',
        label='Dipolar + quadrupolar forces')
ax1.plot(obpic.strength_list, obpic.freq_list[:, 0]-frac_tune_0, '.',
        color='C0', label='PIC simulations')
ax1.plot(stren_array, stren_array*0, ':', color='grey')
ax1.set_xlabel('e-cloud strength')
ax1.set_ylabel('Coherent tune shift')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax1.set_xlim(0, max_strength_plot)
ax1.set_ylim(-1e-2, 1e-2)
ax1.set_yticks(np.arange(-1e-2, 1.0001e-2, 0.5e-2))
ax1.legend(loc='lower left', fontsize='medium', frameon=False,
        bbox_to_anchor=(-0.0,-0.02))

fig1.subplots_adjust(bottom=0.12, left=0.13, right=0.85)

plt.show()
