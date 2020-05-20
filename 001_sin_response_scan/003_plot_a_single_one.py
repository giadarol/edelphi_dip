import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from scipy.constants import e as qe

import PyECLOUD.myfilemanager as mfm
import PyECLOUD.mystyle as ms

folder_sims = 'simulations'
response_summary_file = './response_data_processed.mat'
vmax_edens = 2.5e14

i_plot_list = list(range(1,200))
close_figures = True

obsum = mfm.myloadmat_to_obj(response_summary_file)

plt.close('all')
ms.mystyle_arial(fontsz=14, dist_tick_lab=5, traditional_look=False)
for ii in i_plot_list:

    n_osc = obsum.n_osc_list[ii]
    cos_ampl = obsum.cos_ampl_list[ii]
    sin_ampl = obsum.sin_ampl_list[ii]

    current_sim_ident= f'n_{n_osc:.1f}_c{cos_ampl:.2e}_s{sin_ampl:.2e}'
    ob = mfm.myloadmat_to_obj(folder_sims + '/' + current_sim_ident + '/response.mat')

    #fig1 = plt.figure(100+ii)
    #ax1 = fig1.add_subplot(3,1,1)
    #ax2 = fig1.add_subplot(3,1,2, sharex=ax1)
    #ax3 = fig1.add_subplot(3,1,3, sharex=ax1)

    ##ax1.plot(ob.z_slices, ob.int_slices)
    #ax2.plot(ob.z_slices, ob.x_slices)
    #ax3.plot(ob.z_slices, obsum.dpx_mat[ii, :])

    #for ax in [ax1, ax2, ax3]:
    #    ax.grid(True)

    fig2 = plt.figure(200+ii)
    #ax21 = fig2.add_subplot(2,1,1)
    #ax22 = fig2.add_subplot(2,1,2, sharex=ax21)
    pos_col1 = 0.15
    width_col1 = .6
    pos_row1=.55
    height_row1 = .35
    pos_row2 = .12
    height_row2 = .35
    ax21 = fig2.add_axes((pos_col1, pos_row1, width_col1, height_row1))
    axcb = fig2.add_axes((pos_col1+width_col1+0.05, pos_row1,
                    width_col1*0.07, height_row1))
    ax22 = fig2.add_axes((pos_col1, pos_row2, width_col1, height_row2),
            sharex=ax21)

    mpbl = ax21.pcolormesh(1e2*ob.z_slices, 1e3*ob.xg, -(1/qe)*ob.rho_cut.T,
            vmin=0, vmax=vmax_edens)
    plt.colorbar(mpbl, cax=axcb)
    ax21.plot(1e2*ob.z_slices, 1e3*ob.x_slices, 'k', lw=2)
    ax22.plot(1e2*ob.z_slices, 1e6*obsum.dpx_mat[ii, :], lw=2)
    ax21.set_ylim(-2.5, 2.5)
    ax22.set_ylim(1e6*np.nanmax(np.abs(obsum.dpx_mat[ii, :]))*np.array([-1.1, 1.1]))
    ax22.set_ylim(0.2*np.array([-1.1, 1.1]))

    ax21.set_ylabel('x [mm]')
    ax21.set_xlim(-30, 30)
    ax21.set_ylim(-2.5, 2.5)
    ax22.set_ylabel(r'$\Delta$p$_{x}$ [$\mu$rad]')
    ax22.set_xlabel('z [cm]')
    #ax22.grid(linestyle=':', alpha=.9)
    axcb.set_ylabel(r'Charge density [e$^-$/m$^3$]')
    for ax in [ax21, ax22]:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    fig2.suptitle(f'n={n_osc} ' + {True: 'sine', False: 'cosine'}[sin_ampl>cos_ampl], fontsize=14)
    fig2.subplots_adjust(
        top=0.91,
        bottom=0.12,
        left=0.16,
        right=0.915,
        hspace=0.2,
        wspace=0.2)

    fig2.savefig(current_sim_ident + '.png', dpi=200)
    if close_figures:
        plt.close(fig2)

plt.show()

