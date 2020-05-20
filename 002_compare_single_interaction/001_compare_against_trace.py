import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.constants import e as qe

from PyPARIS_sim_class import Simulation as sim_mod
import PyPARIS.util as pu
import PyECLOUD.myfilemanager as mfm
import PyECLOUD.mystyle as ms

# Import response_matrix
import sys
sys.path.append('../')
import response_matrix.response_matrix as rm

test_data_file = './refsim_turn302.mat'
#n_terms_list = range(1, 201, 2)
n_terms_list = [6, 8, 10]
n_tail_cut = 10
response_data_file = '../001_sin_response_scan/response_data_processed.mat'
z_strength_file = '../001a_sin_response_scan_unperturbed/linear_strength.mat'

#test_data_file = './test_pulse.mat'
#n_terms_list = [200]
#n_tail_cut = 10
#response_data_file = '../001_sin_response_scan/response_data.mat'

sim_param_file = '../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = ['../Simulation_parameters_amend.py']

# Instantiate simulation
sim_content = sim_mod.Simulation(param_file=sim_param_file)

# Here sim_content.pp can be edited (directly and through files)
for ff in sim_param_amend_files:
    sim_content.pp.update(param_file=ff)

# Disable real e-clouds
sim_content.pp.enable_arc_dip = False
sim_content.pp.enable_arc_quad = False

# Add ring of CPU information
ring_cpu = pu.get_serial_CPUring(sim_content,
        init_sim_objects_auto=False)
assert(sim_content.ring_of_CPUs.I_am_the_master)

# Initialize machine elements
sim_content.init_all()

# Initialize master to get the beam
if os.path.exists('simulation_status.sta'):
    os.remove('simulation_status.sta')

# Initialize beam
sim_content.init_master()

# Get bunch and slicer
bunch = sim_content.bunch
slicer = sim_content.slicer


# Get simulation data
obsim = mfm.myloadmat_to_obj(test_data_file)
x_test = obsim.x_slices
int_test = obsim.int_slices
x_test[np.isnan(x_test)] = 0.
xg = obsim.xg
yg = obsim.yg


plt.close('all')
ms.mystyle_arial(fontsz=14, dist_tick_lab=5, traditional_look=False)

figglob = plt.figure(1, figsize=(6.4, 4.8*1.15))
#axg1 = figglob.add_subplot(3,1,1)
axg2 = figglob.add_subplot(2,1,1)
axg3 = figglob.add_subplot(2,1,2, sharex=axg2)

#mpbl = axg1.pcolormesh(1e2*obsim.z_slices, 1e3*xg, -(1./qe)*obsim.rho_cut.T)
#plt.colorbar(mpbl,orientation='horizontal')
#axg1.plot(1e2*obsim.z_slices, 1e3*obsim.x_slices, 'k', lw=2)
axg2.plot(1e2*obsim.z_slices, 1e3*x_test, lw=3., color='k')
lines_pic = axg3.plot(1e2*obsim.z_slices, 1e6*obsim.dpx_slices,
        lw=3., color='k', linestyle='-',
        label='PIC')

line_list = []
for ifig, n_terms_to_be_kept in enumerate(n_terms_list):

    # Build matrix
    respmat = rm.ResponseMatrix(
            slicer=slicer,
            response_data_file=response_data_file,
            coord='x',
            kick_factor=1./sim_content.n_segments,
            n_terms_to_be_kept=n_terms_to_be_kept,
            n_tail_cut=n_tail_cut)

    # Recenter all slices
    slices = bunch.get_slices(slicer)
    for ii in range(slices.n_slices):
        ix = slices.particle_indices_of_slice(ii)
        if len(ix) > 0:
            bunch.x[ix] -= np.mean(bunch.x[ix])
            bunch.xp[ix] -= np.mean(bunch.xp[ix])

    # Distort bunch
    bunch.x = bunch.x + np.interp(bunch.z, respmat.z_resp, x_test)

    # Apply matrix
    respmat.track(bunch)

    # Apply detuning
    if z_strength_file is not None:
        obdet = mfm.myloadmat_to_obj(z_strength_file)
        bunch.xp = bunch.xp + bunch.x * np.interp(bunch.z, obdet.z_slices,
                obdet.k_z_integrated) / sim_content.n_segments

    # Measure kicks
    bunch.clean_slices()
    slices_test = bunch.get_slices(slicer, statistics=['mean_x', 'mean_xp'])

    # Get x_reconstr
    a_coeff, x_reconstr = respmat.decompose_trace(x_test)

    # Plots

    z_resp = respmat.z_resp

    n_freq = (n_terms_to_be_kept-1)/2

    fig2 = plt.figure(200 + ifig, figsize=(6.4, 4.8*1.5))
    ax2 = fig2.add_subplot(3,1,2)
    ax2.plot(1e2*z_resp, 1e3*x_test, label='Test trace')
    ax2.plot(1e2*z_resp, 1e3*x_reconstr, label=f'Reconstructed (n={n_freq})')
    ax2.set_ylim(1e3*np.nanmax(np.abs(x_test))*np.array([-1, 1]))
    ax2.set_ylabel('x [mm]')
    ax2.legend(prop={'size':12}, loc='lower right', ncol=2)

    ax3 = fig2.add_subplot(3,1,3, sharex=ax2)
    ax3.plot(1e2*z_resp, 1e6*obsim.dpx_slices, label='Simulation')
    ax3.plot(1e2*slices_test.z_centers, 1e6*slices_test.mean_xp, label=f'Harm. response (n={n_freq})')
    ax3.set_ylim(1e6*np.nanmax(np.abs(obsim.dpx_slices))*np.array([-1.5, 1.1]))
    ax3.set_ylabel('Dpx [urad]')
    ax3.set_xlabel('z [cm]')
    ax3.legend(prop={'size':12}, loc='lower right', ncol=2)

    for aa in [ax2, ax3]:
        aa.grid(linestyle=':', alpha=.9)

    i_yzero = np.argmin(np.abs(yg))

    ax21 = fig2.add_subplot(3,1,1, sharex=ax2)

    ax21.pcolormesh(1e2*obsim.z_slices, 1e3*xg, obsim.rho_cut.T)
    ax21.plot(1e2*obsim.z_slices, 1e3*obsim.x_slices, 'k', lw=2)
    ax21.set_ylim(-2.5, 2.5)
    ax21.set_ylabel('x [mm]')
    ax21.set_title('Electron density', fontsize=14)
    ax21.set_xlim(-30, 30)

    fig2.subplots_adjust(hspace=.22, left=.18,
            bottom=0.09, top=.93)

    fig2.savefig(test_data_file.split('.mat')[0] + f'_n{n_terms_to_be_kept:04d}.png', dpi=200)

    kwargs = {}
    if len(n_terms_list)==3:
        kwargs['color'] = plt.cm.Reds([.3, .5, .8][ifig])
    axg2.plot(1e2*slices_test.z_centers, 1e3*x_reconstr,
            label=f'n={n_terms_to_be_kept}', **kwargs, lw=2)
    ln = axg3.plot(1e2*slices_test.z_centers, 1e6*slices_test.mean_xp,
            label=f'N={n_terms_to_be_kept}', **kwargs, lw=2)
    line_list += ln

#axg1.set_ylim(-2.5, 2.5)
#axg1.set_ylabel('x [mm]')
#axg1.xaxis.set_tick_params(labelbottom=False)

axg2.set_ylim(-.3, .3)
axg2.set_xlim(-30, 30)
#bbox_to_anchor : ?,v-pos,width?
axg3.legend(handles=lines_pic+line_list,
        bbox_to_anchor=(0., -.4, 1, 0), loc='upper left',
           ncol=4, mode="expand", borderaxespad=0., frameon=False)
axg3.set_xlabel('z [cm]')
axg2.set_ylabel('x [mm]')
axg3.set_ylabel(r"$\Delta$x' [$\mu$rad]")
axg3.ticklabel_format(style='sci', scilimits=(0,0),axis='y')
axg2.ticklabel_format(style='sci', scilimits=(0,0),axis='y')
axg2.xaxis.set_tick_params(labelbottom=False)
#legpic = plt.legend(handles=lines_pic)
figglob.subplots_adjust(bottom=.2)
plt.show()
