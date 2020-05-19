import pickle

import numpy as np
import scipy
from scipy.constants import c as clight
from scipy.interpolate import interp2d
import scipy.io as sio

import PyECLOUD.mystyle as ms

# strength = 1.35
# pkl_fname = '../008_eigenvalues/mode_coupling_matrix.pkl'

strength_in_fname= 1.24 # 1.18
strength_rescale = 1.
pkl_fname = ('../008a1_scan_strength_wlampldet/simulations/'
            f'strength_{strength_in_fname:.3f}/mode_coupling_matrix.pkl')

# strength_in_fname = 0.6 # 1.18
# strength_rescale = 1.
# pkl_fname = ('../008b_scan_strength_wlampldet_Qp5/simulations/'
#             f'strength_{strength_in_fname:.3f}/mode_coupling_matrix.pkl')

omega0 = 2*np.pi*clight/27e3 # Revolution angular frquency
omega_s = 4.9e-3*omega0
omega_mode_plot = None # -omega_s

l_min = -7
l_max = 7
m_max = 29
N_max = 29#9
abs_min_imag_unstab = 30.
n_r = 200
n_phi = 210
n_z = 220
n_delta = 250
n_traces = 15
n_modes_plot = 6
flag_with_phase_shift = True

with open(pkl_fname, 'rb') as fid:
    MM_orig = pickle.load(fid)

MM_obj = MM_orig.get_sub_matrix(l_min, l_max, m_max, N_max)


# Mode coupling test
Omega_mat, evect = MM_obj.compute_mode_complex_freq(omega_s,
        rescale_by=[strength_rescale], flag_eigenvectors=True)
i_most_unstable = np.argmin(np.imag(Omega_mat))
i_sorted_by_risetime = np.argsort(np.imag(Omega_mat))

i_selected = i_sorted_by_risetime[:n_modes_plot]
temp_index = np.argsort(Omega_mat.real[i_selected])
i_selected_sorted = np.take(i_selected, temp_index)

# # Plot intrabunch
# if omega_mode_plot is None:
#     i_intrab = i_most_unstable
# else:
#     # Select unstable modes
#     Omega_mat_temp = Omega_mat.copy()
#     Omega_mat_temp[np.imag(Omega_mat)>-abs_min_imag_unstab] = 1e10
#     i_intrab = np.argmin(np.abs(
#         np.real(Omega_mat_temp) - omega_mode_plot))

r_max = 4*MM_obj.sigma_b
r_vect = np.linspace(0, r_max, n_r)
phi_vect = np.linspace(0, 2*np.pi, n_phi)
l_vect = MM_obj.l_vect
m_vect = MM_obj.m_vect
n_l = len(l_vect)

import matplotlib.pyplot as plt
plt.close('all')
fig2 = plt.figure(2, figsize=(6.4*1.6, 4.8*1.4))
axlist = [fig2.add_subplot(3, 2, ii+1) for ii in range(6)]
for i_plot in range(n_modes_plot):
    i_intrab = i_selected_sorted[i_plot]

    R_L_R = np.zeros((n_l, n_r), dtype=np.complex)
    for i_ll, ll in enumerate(l_vect):
        sumRl = (0 + 0j) * r_vect
        for i_mm, mm in enumerate(m_vect):
            sumRl += (evect[i_ll, i_mm, i_intrab]
                   * scipy.special.eval_genlaguerre(mm, abs(ll),
                           MM_obj.a_param * r_vect  ** 2))

        Rl = (sumRl * (r_vect / MM_obj.r_b)**np.abs(ll)
                  * np.exp(-MM_obj.a_param * r_vect**2))

        R_L_R[i_ll, :] = Rl

    z_vect = np.linspace(-r_max, r_max, n_z)
    delta_vect = np.linspace(-r_max, r_max, n_delta)
    distr_Z_Delta = np.zeros((n_delta, n_z), dtype=np.complex)
    ZZ, DD = np.meshgrid(z_vect, delta_vect)
    RR = np.sqrt(ZZ**2 +DD**2)
    PP = np.arctan2(DD, ZZ)
    PP[PP<0] += 2*np.pi

    # Resample DPhi
    if flag_with_phase_shift:
        DPhi_func = interp2d(MM_orig.r_vect,
            MM_orig.phi_vect, MM_orig.dPhi_R_PHI.T)
        DPhi_Z_Delta = np.array(
                [DPhi_func(rr, pp) for rr, pp in zip(RR.flatten(), PP.flatten())]).reshape(
                    RR.shape)
    else:
        DPhi_Z_Delta = 0.

    for i_ll, ll in enumerate(l_vect):
        Rl_this = np.interp(RR.flatten(), r_vect, R_L_R[i_ll, :]).reshape(RR.shape)
        distr_Z_Delta += Rl_this*np.exp(-1j*ll*PP-1j*DPhi_Z_Delta)

    distr_Z = np.sum(distr_Z_Delta, axis=0)

    phase_osc = np.linspace(0, 2*np.pi, n_traces+1)[:-1]

    T_rev = 2 * np.pi / omega0
    ax = axlist[i_plot]
    for i_trace in range(n_traces):
        #ax.plot(z_vect, np.real(distr_Z*np.exp(1j*phase_osc[i_trace])))
        ax.plot(z_vect, np.exp(Omega_mat[i_intrab].imag * i_trace * T_rev) *
                   np.real(distr_Z *
                    np.exp(1j * 2*np.pi
                    * (MM_obj.Q_full + Omega_mat[i_intrab].real/omega0)*i_trace)))
    ax.set_title(
            r'Q'+ f'={np.modf(MM_obj.Q_full)[0] + Omega_mat[i_intrab].real/omega0:.3}'
            r' $\Delta$Q/Q$_s$'+ f'={Omega_mat[i_intrab].real/omega_s:.3}'
            f' risetime={-Omega_mat[i_intrab].imag:.1f}')

   # fig3 = plt.figure(3)
   # for i_trace in range(n_traces):
   #     plt.plot(z_vect, np.imag(distr_Z*np.exp(1j*phase_osc[i_trace])))

fig2.suptitle(f'strength: {strength_in_fname:.3f} strength_rescale: {strength_rescale:.3f}')
fig2.subplots_adjust(
        top=0.91,
        bottom=0.07,
        left=0.075,
        right=0.96,
        hspace=0.385,
        wspace=0.2)

plt.show()
