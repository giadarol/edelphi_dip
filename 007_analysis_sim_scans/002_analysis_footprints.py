import sys
sys.path.append('tools')
sys.path.append("PyHEADTAIL")

import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.api as sm

import PyECLOUD.myfilemanager as mfm
import PyECLOUD.mystyle as ms


from PyPARIS_sim_class import LHC_custom

fname_root = 'full_vs_lin'
folders = [
           '../004a_instab_simulation_6MV_full_mat_nl_map',
           '../004c_instab_simulation_6MV_dipolar_mat_lin_map',
           ]
leg_labels = ['full', 'lin']
labels = ['full', 'lin',]
cmap = plt.cm.rainbow


# fname_root = None
# strength_scan = np.arange(0.1, 2.1, 0.1)
# folders = [f'../006a_footprints_strength_scan/simulations/strength_{ss:.2e}' for ss in strength_scan]
# labels = [f'{ss:.1f}' for ss in strength_scan]
# leg_labels = labels
# cmap = plt.cm.rainbow


def extract_info_from_sim_param(fname):
    with open(fname, 'r') as fid:
        lines = fid.readlines()

    ddd = {}
    # Extract V_RF
    for ll in lines:
        if '=' in ll:
            nn = ll.split('=')[0].replace(' ','')
            try:
                ddd[nn] = eval(ll.split('=')[-1])
            except:
                ddd[nn] = 'Failed!'
    return ddd

plt.close('all')
ms.mystyle_arial(fontsz=14, dist_tick_lab=5,
        traditional_look=False)

figglob = plt.figure(1)
axglob = figglob.add_subplot(111)
axdistrlist = []
figfplist = []
fighlist = []
figvlist = []
fig_comb_h = plt.figure(2)
ax_comb_h = fig_comb_h.add_subplot(111)
figtest = plt.figure(3)
axtest = figtest.add_subplot(111)
for ifol, folder in enumerate(folders):
    try:
        import pickle
        with open(folder+'/sim_param.pkl', 'rb') as fid:
            pars = pickle.load(fid)
    except IOError:
        config_module_file = folder+'/Simulation_parameters.py'
        print('Config pickle not found, loading from module:')
        print(config_module_file)
        pars = mfm.obj_from_dict(
                extract_info_from_sim_param(config_module_file))

    machine = LHC_custom.LHC(
              n_segments=1,
              machine_configuration=pars.machine_configuration,
              beta_x=pars.beta_x, beta_y=pars.beta_y,
              accQ_x=pars.Q_x, accQ_y=pars.Q_y,
              Qp_x=pars.Qp_x, Qp_y=pars.Qp_y,
              octupole_knob=pars.octupole_knob,
              optics_dict=None,
              V_RF=pars.V_RF
              )
    Qs = machine.longitudinal_map.Q_s
    Qx = machine.transverse_map.accQ_x
    Qy = machine.transverse_map.accQ_y
    frac_qx, _ = math.modf(Qx)
    frac_qy, _ = math.modf(Qy)

    filename_footprint = 'footprint.h5'
    ob = mfm.object_with_arrays_and_scalar_from_h5(
            folder + '/' + filename_footprint)

    betax = machine.transverse_map.beta_x[0]
    betay = machine.transverse_map.beta_y[0]
    Jy = (ob.y_init**2 + (ob.yp_init*betay)**2)/(2*betay)
    Jx = (ob.x_init**2 + (ob.xp_init*betax)**2)/(2*betax)

    Qx_min = frac_qx -  0.03
    Qy_min = frac_qy -  0.03
    Qx_max_cut = frac_qx + 0.05
    Qy_max_cut = frac_qy + 0.05

    fig1 = plt.figure(1000+ifol, figsize=(6.4*1.1, 4.8*1.4))
    figfplist.append(fig1)

    ax1 = fig1.add_subplot(111)
    mpbl1 = ax1.scatter(np.abs(ob.qx_i), np.abs(ob.qy_i),
            c =ob.z_init*1e2, marker='.', edgecolors='none', vmin=-32, vmax=32)
    ax1.plot([frac_qx], [frac_qy], '*k', markersize=10)
    ax1.set_xlabel('Q$_x$')
    ax1.set_ylabel('Q$_y$')
    ax1.set_aspect(aspect='equal', adjustable='datalim')
    ax1.set(xlim=(Qx_min, Qx_max_cut), ylim=(Qy_min, Qy_max_cut))
    ax1.grid(True, linestyle='--', alpha=0.5)

    divider = make_axes_locatable(ax1)
    axhistx = divider.append_axes("top", size=1.2, pad=0.25, sharex=ax1)
    axcb = divider.append_axes("right", size=0.3, pad=0.1)
    axhistx.grid(True, linestyle='--', alpha=0.5)
    obstat = sm.nonparametric.KDEUnivariate(ob.qx_i)
    obstat.fit(bw=10e-4)
    q_axis = np.linspace(Qx_min, Qx_max_cut, 1000)
    axhistx.plot(q_axis, obstat.evaluate(q_axis))
    axhistx.fill_between(x=q_axis, y1=0, y2=obstat.evaluate(q_axis), alpha=0.5)
    if leg_labels is None:
        lll ='%.1f'%machine.i_octupole_focusing
    else:
        lll = leg_labels[ifol]
    axglob.plot(q_axis, obstat.evaluate(q_axis),
            label=lll,
            linewidth=2.,
            color=cmap(float(ifol)/float(len(folders))))
    axdistrlist.append(axhistx)
    plt.colorbar(mpbl1, cax=axcb)

    fig2 = plt.figure(2000+ifol)
    ax2 = fig2.add_subplot(111)
    mpbl = ax2.scatter(ob.z_init*1e2,
            np.abs(ob.qx_i)-frac_qx, c =Jx,
            marker='.', edgecolors='none', vmin=0, vmax=8e-9)
    cb = plt.colorbar(mpbl)
    cb.ax.set_ylabel('Transverse action')
    ax2.set_xlim(-30, 30)
    ax2.set_ylim(-0.0, 3e-2)
    ax2.set_xlabel('z [cm]')
    ax2.set_ylabel('$\Delta$Qx', labelpad=5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    fig2.subplots_adjust(
            top=0.88,
            bottom=0.11,
            left=0.155,
            right=0.965,
            hspace=0.2,
            wspace=0.2)
    fighlist.append(fig2)

    fig3 = plt.figure(3000+ifol)
    ax3 = fig3.add_subplot(111)
    mpbl = ax3.scatter(ob.z_init*1e2,
            np.abs(ob.qy_i)-frac_qy, c =Jy,
            marker='.', edgecolors='none', vmin=0, vmax=8e-9)
    cb = plt.colorbar(mpbl)
    cb.ax.set_ylabel('Transverse action')
    ax3.set_xlim(-30, 30)
    ax3.set_ylim(-0.0, 3e-2)
    ax3.set_xlabel('z [cm]')
    ax3.set_ylabel('$\Delta$Qy', labelpad=5)
    ax3.grid(True, linestyle='--', alpha=0.5)
    fig3.subplots_adjust(
            top=0.88,
            bottom=0.11,
            left=0.155,
            right=0.965,
            hspace=0.2,
            wspace=0.2)
    figvlist.append(fig3)

    kwargs = {'markersize': 2}
    if ifol<2:
        kwargs['color'] = ['C0', 'C3'][ifol]
        kwargs['markersize'] = [1, 5][ifol]
    ax_comb_h.plot(ob.z_init*1e2,
            np.abs(ob.qx_i)-frac_qx, '.', **kwargs)
    # sigma_x = np.sqrt(pars['epsn_x']*betax/machine.betagamma)
    # sigma_y = np.sqrt(pars['epsn_y']*betay/machine.betagamma)
    # mask_small_amplitude = np.sqrt(
    #         (ob.x_init/sigma_x)**2 +(ob.x_init/sigma_x)**2) < 0.2
    # z_small = ob.z_init[mask_small_amplitude]
    # qx_small = ob.qx_i[mask_small_amplitude]
    # ax2.plot(z_small*1e2, qx_small - frac_qx, 'k.', markersize=10)

    sigma_z = np.std(ob.z_init)
    z_vector = np.linspace(-3*sigma_z, 3*sigma_z, 100)
    Dz = z_vector[1] - z_vector[0]
    Qx_ave = 0*z_vector
    for i_z, zz in enumerate(z_vector):
        mask_in_slice = np.abs(ob.z_init - zz) <= Dz/2.
        Qx_ave[i_z] = np.mean(np.abs(ob.qx_i)[mask_in_slice]-frac_qx)

    axtest.plot(z_vector, Qx_ave)


    for ff in [fig1, fig2]:
        ff.suptitle(labels[ifol] + ' - I$_{LOF}$=%.1fA'%machine.i_octupole_focusing)

if leg_labels is None:
    legtitle = 'I$_{LOF}$'
else:
    legtitle = None
axglob.legend(loc='best', title=legtitle)
axglob.grid(True, linestyle='--', alpha=0.5)
axglob.set_ylim(bottom=0)
axglob.set_ylabel('Density [a.u.]')
axglob.set_xlabel('Q$_x$')

ax_comb_h.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
ax_comb_h.set_ylim(bottom=0)
ax_comb_h.grid(True, linestyle=':')
ax_comb_h.set_ylabel('Tune deviation')
ax_comb_h.set_xlabel('z [cm]')
fig_comb_h.subplots_adjust(bottom=.12)

for aa in axdistrlist:
    aa.set_ylim(axglob.get_ylim())
    aa.set_ylabel('Density [a.u.]')

if fname_root is not None:
    figglob.savefig(fname_root+'_spreads.png', dpi=200)
    for ff, ll in zip(figfplist, labels):
        ff.savefig(fname_root+'_'+ll.replace(' ', '_').replace('_=', '')+'.png', dpi=200)
    for ff, ll in zip(fighlist, labels):
        ff.savefig(fname_root+'_h_'+ll.replace(' ', '_').replace('_=', '')+'.png', dpi=200)
    for ff, ll in zip(fighlist, labels):
        ff.savefig(fname_root+'_v_'+ll.replace(' ', '_').replace('_=', '')+'.png', dpi=200)
plt.show()
