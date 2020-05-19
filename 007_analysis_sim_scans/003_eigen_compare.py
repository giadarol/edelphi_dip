import pickle

import numpy as np
from scipy.constants import c as clight
import scipy.io as sio

import PyECLOUD.mystyle as ms
import PyECLOUD.myfilemanager as mfm

T_rev=88.9e-6

# Plot settings
l_min_plot = -6
l_max_plot = 4
min_strength_plot = 0
max_strength_plot = 1.5
tau_min_plot = 0
tau_max_plot = 300
max_strength_tau_plot = 1.5001
flag_tilt_lines = False
flag_mode0 = True
fig_fname = 'vlas'
flag_intra = False

colorlist = None
colorlist = 'C0 C1 C2'.split()


dict_plot = {
        'onlydip':{'fname':'../008_eigenvalues/eigenvalues.mat',
            'mpsim_fname': './processed_data/compact_t1_fit.mat',
            'label': r'$\Delta$Q$_\Phi=$0, $\Delta$Q$_R=$0'},
        'phase':{'fname':'../008a_scan_strength/eigenvalues.mat',
            'mpsim_fname': './processed_data/compact_t2_fit.mat',
            'label': r'$\Delta$Q$_\Phi\neq$0, $\Delta$Q$_R=$0'},
        'wquad':{'fname':'../008a1_scan_strength_wlampldet/eigenvalues.mat',
            'mpsim_fname': './processed_data/compact_t3_fit.mat',
            'label': r'$\Delta$Q$_\Phi\neq$0, $\Delta$Q$_R\neq$0'},
        }

import matplotlib
import matplotlib.pyplot as plt
plt.close('all')
ms.mystyle_arial(fontsz=14, dist_tick_lab=5, traditional_look=False)
fig1 = plt.figure(1, figsize=(6.4*1.2, 4.8))
ax1 = fig1.add_subplot(111)
fig_re_list = []
fig_im_list = []
for ill, ll in enumerate(dict_plot.keys()):

    fname = dict_plot[ll]['fname']

    ob = mfm.myloadmat_to_obj(fname)
    Omega_mat = ob.Omega_mat
    strength_scan = ob.strength_scan
    strength_scan_0 = strength_scan
    # Omega_mat = np.zeros((ob.Omega_mat.shape[0]*2, ob.Omega_mat.shape[1]),
    #         dtype=np.complex)
    # strength_scan = np.zeros(Omega_mat.shape[0])
    # strength_scan_0 = ob.strength_scan
    # # Sort by real part
    # for iss in range(ob.Omega_mat.shape[0]):
    #     isorted = np.argsort(ob.Omega_mat[iss, :].real
    #             + 1e-10 * ob.Omega_mat[iss, :].imag)
    #     Omega_mat[2*iss] = np.take(ob.Omega_mat[iss, :], isorted)
    #     strength_scan[2*iss] = ob.strength_scan[iss]
    # # Interpolate
    # for iss in range(ob.Omega_mat.shape[0]-2):
    #     Omega_mat[2*iss+1,:] = 0.5 * (Omega_mat[2*iss + 2, :] + Omega_mat[2*iss, :])
    #     strength_scan[2*iss+1] = 0.5 * (strength_scan[2*iss + 2]
    #             + strength_scan[2*iss])
    DQ_0_vect = ob.DQ_0_vect
    M00_array = ob.M00_array
    omega0 = ob.omega0
    omega_s = ob.omega_s
    l_min = ob.l_min
    l_max = ob.l_max
    m_max = ob.m_max
    N_max = ob.N_max

    fig500 = plt.figure(500+ill)#, figsize=(1.3*6.4, 1.3*4.8))
    ax = fig500.add_subplot(111)
    ax.set_facecolor('darkgrey')
    im_min_col = 5
    im_max_col = 200
    im_min_size = 3
    im_max_size = 50
    if flag_mode0:
        maskmode0 = ((np.real(M00_array)/omega_s>-0.9) & (strength_scan<0.75))
        ax.plot(strength_scan_0[maskmode0],
                np.real(M00_array)[maskmode0]/omega_s, '--',
                linewidth=2, color='orange', alpha=1.)
    for jj in range(len(strength_scan)):
        Omega_jj = Omega_mat[jj, :]
        ind_sorted = np.argsort(-np.imag(Omega_jj))
        re_sorted = np.take(np.real(Omega_jj), ind_sorted)
        im_sorted = np.take(np.imag(Omega_jj), ind_sorted)
        sctr = plt.scatter(x=strength_scan[jj]+0*np.imag(Omega_mat[jj, :]),
                y=re_sorted/omega_s,
                c = np.clip(-im_sorted, im_min_col, im_max_col),
                cmap=plt.cm.jet,
                s=np.clip(-im_sorted, im_min_size, im_max_size),
                vmin=im_min_col, vmax=im_max_col,
                norm=matplotlib.colors.LogNorm(),
                )
    plt.suptitle(ll)
    ax.set_xlim(min_strength_plot, max_strength_plot)
    ax.set_ylim(l_min_plot, l_max_plot)
    ax.set_yticks(np.arange(l_min_plot+1, l_max_plot-.1))
    ax.set_xlabel('e-cloud strength')
    ax.set_ylabel(r'(Q - Q$_0$)/Q$_s$')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="3%", pad=0.1, pack_start=False)
    fig500.add_axes(cax)
    cbar = fig500.colorbar(sctr, cax=cax, orientation="vertical")
    cbar.ax.set_ylabel(r'Growth rate [s$^{-1}$]', labelpad = 0.1)
    fig500.subplots_adjust(right=.88, bottom=.12)

    for lll in range(l_min_plot-10, l_max_plot+10):
        if flag_tilt_lines:
            add_to_line = np.array(DQ_0_list)*omega0/omega_s
        else:
            add_to_line = 0.
        ax.plot(strength_scan,
                0*strength_scan + lll + add_to_line,
                color='w',
                alpha=.5, linestyle='--')
    #ax.tick_params(right=True, top=True)

    figtau = plt.figure(600+ill)
    axtau = figtau.add_subplot(111)
    axtau.plot(strength_scan, np.imag(Omega_mat),
            '.', color='grey', alpha=0.5)
    axtau.plot(strength_scan, np.max(-np.imag(Omega_mat), axis=1),
            linewidth=2, color='b')
    axtau.set_xlim(min_strength_plot, max_strength_plot)
    axtau.set_ylim(tau_min_plot, tau_max_plot)
    axtau.grid(True, linestyle=':')
    figtau.suptitle(ll)

    kwargs = {}
    if colorlist is not None:
        kwargs['color'] = colorlist[ill]
    ax1.plot(strength_scan, np.max(-np.imag(Omega_mat), axis=1),
            label=dict_plot[ll]['label'],
            linewidth=3, alpha=.5, **kwargs)

    fig_re_list.append(fig500)
    fig_im_list.append(figtau)

    fig500.savefig(fig_fname+f'_{ll}_re.png', dpi=200)
    figtau.savefig(fig_fname+f'_{ll}_im.png', dpi=200)

    mpsim_fname = dict_plot[ll]['mpsim_fname']
    if mpsim_fname is not None:
        oo = mfm.myloadmat_to_obj(mpsim_fname)
        if flag_intra:
            gr_sim = oo.p_list_intra/T_rev
        else:
            gr_sim = oo.p_list_centroid/T_rev
        ax1.plot(oo.strength_list, gr_sim, '.', alpha=.5,
            markeredgewidth=0, **kwargs)
        from scipy.signal import savgol_filter
        mask_plot = oo.strength_list < max_strength_tau_plot
        smooth_gr = savgol_filter(gr_sim[mask_plot], 31, 5)
        ax1.plot(oo.strength_list[mask_plot],
                smooth_gr, '--', linewidth=3, **kwargs)

ax1.legend(loc='upper left', fontsize='medium', frameon=False)
#ax1.grid(True, linestyle=':')
ax1.set_xlim(0, max_strength_tau_plot)
ax1.set_ylim(tau_min_plot, tau_max_plot)
ax1.set_xlabel('e-cloud strength')
ax1.set_ylabel(r'Instability growth rate [s$^{-1}$]')
fig1.subplots_adjust(right=.71, bottom=.12, top=.85)
fig1.savefig(fig_fname + '_glob.png', dpi=200)
plt.show()

