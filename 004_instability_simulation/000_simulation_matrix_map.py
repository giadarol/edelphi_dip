import os
import time

import numpy as np
from scipy.constants import c as clight

import PyECLOUD.myfilemanager as mfm
from PyPARIS_sim_class import Simulation as sim_mod
import PyPARIS.util as pu

# Import response_matrix
import sys
sys.path.append('../')
import response_matrix.response_matrix as rm
import response_matrix.modulated_quadrupole as mq

# Default value
flag_suppress_alpha_0 = False
alpha_scale = 1.
# start-settings-section
n_terms_to_be_kept = 49
n_tail_cut = 10
recenter_all_slices = True # Cancels initial kick from input

Qp_x = 0.
Qp_y = 0.

plane = 'y'

ecloud_strength_scale = 1.

sim_param_file = '../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = ['../Simulation_parameters_amend.py',
                    'Simulation_parameters_amend_for_matrixsim.py']

include_response_matrix = True
response_data_file = '../001_sin_response_scan/response_data_y_processed.mat'

include_detuning_with_z = True
only_phase_shift = True
add_alpha_0_to_tune = False
factor_alpha_0_to_tune = 0.
z_strength_file = '../000a_sin_response_unperturbed_pinch/linear_strength_y.mat'
detuning_fit_order = 10
N_poly_cut = 1
alpha_N_custom = []

include_non_linear_map = True
flag_wrt_bunch_centroid = True
field_map_file = '../003_generate_field_map/field_map.mat'
# end-settings-section

# Load and fit detuning with z
if include_detuning_with_z:
    if detuning_fit_order > 0:
        obdet = mfm.myloadmat_to_obj(z_strength_file)
        z_slices = obdet.z_slices
        p = np.polyfit(obdet.z_slices, obdet.k_z_integrated, deg=detuning_fit_order)
        alpha_N = p[::-1]*ecloud_strength_scale # Here I fit the strength
        print('Detuning cefficients alpha_N:')
        print(alpha_N)
    else:
        alpha_N = alpha_N_custom

    alpha_N = alpha_N[:N_poly_cut]*alpha_scale

# Instantiate simulation
sim_content = sim_mod.Simulation(param_file=sim_param_file)

# Here sim_content.pp can be edited (directly and through files)
for ff in sim_param_amend_files:
    sim_content.pp.update(param_file=ff)

# Make the slice output file smaller
sim_content.pp.slice_stats_to_store = [f'mean_{plane}', f'mean_{plane}p',
 'n_macroparticles_per_slice']

sim_content.pp.Qp_x = Qp_x
sim_content.pp.Qp_y = Qp_y

if add_alpha_0_to_tune:
    raise ValueError('Needs to be generalized to y plane!')
    assert(only_phase_shift)
    sim_content.pp.Q_x += -(factor_alpha_0_to_tune * alpha_N[0]
                            * sim_content.pp.beta_x) / (4 * np.pi)

# Disable real e-clouds
sim_content.pp.enable_arc_dip = False
sim_content.pp.enable_arc_quad = False

# Add ring of CPU information
ring_cpu = pu.get_serial_CPUring(sim_content,
        init_sim_objects_auto=False)
assert(sim_content.ring_of_CPUs.I_am_the_master)

# Initialize machine elements
sim_content.init_all()

# Initialize beam, slicer, monitors, multijob mode
sim_content.init_master()

# Get bunch, slicer, machine and monitors
bunch = sim_content.bunch
slicer = sim_content.slicer
machine = sim_content.machine
bunch_monitor = sim_content.bunch_monitor
slice_monitor = sim_content.slice_monitor

# Recenter all slices
if recenter_all_slices and sim_content.SimSt.first_run:
    slices = bunch.get_slices(slicer)
    for ii in range(slices.n_slices):
        ix = slices.particle_indices_of_slice(ii)
        if len(ix) > 0:
            bunch.x[ix] -= np.mean(bunch.x[ix])
            bunch.xp[ix] -= np.mean(bunch.xp[ix])
            bunch.y[ix] -= np.mean(bunch.y[ix])
            bunch.yp[ix] -= np.mean(bunch.yp[ix])

# Build matrix
if include_response_matrix:
    respmat = rm.ResponseMatrix(
        slicer=slicer,
        response_data_file=response_data_file,
        coord=plane,
        kick_factor=1./sim_content.n_segments*ecloud_strength_scale,
        n_terms_to_be_kept=n_terms_to_be_kept,
        n_tail_cut=n_tail_cut)
    machine.install_after_each_transverse_segment(respmat)

# Add modulated quadrupole
if include_detuning_with_z:
    if len(alpha_N)>0 or len(beta_N)>0:
        omega_0 = 2 * np.pi * clight / machine.circumference
        v_eta__omegas = (clight *machine.longitudinal_map.eta(dp=0, gamma=machine.gamma)
                / (omega_0 * machine.longitudinal_map.Q_s))
        mquad = mq.ModulatedQuadrupole(coord=plane,
                alpha_N=np.array(alpha_N)/sim_content.pp.n_segments,
                beta_N=[],
                only_phase_shift=only_phase_shift,
                v_eta__omegas=v_eta__omegas)
        machine.install_after_each_transverse_segment(mquad)

# Introduce non-linear field map
if include_non_linear_map:

    obfmap = mfm.myloadmat_to_obj(field_map_file)
    from PyECLOUD.Transverse_Efield_map_for_frozen_cloud import Transverse_Efield_map
    fmap = Transverse_Efield_map(
        xg=obfmap.xg,
        yg=obfmap.yg,
        Ex=obfmap.Ex_L_map,
        Ey=obfmap.Ey_L_map,
        L_interaction=1./sim_content.n_segments*ecloud_strength_scale,
        slicer=slicer,
        flag_clean_slices=False,
        wrt_slice_centroid=flag_wrt_bunch_centroid,
        x_beam_offset=0.,
        y_beam_offset=0.,
        slice_by_slice_mode=False)
    machine.install_after_each_transverse_segment(fmap)


# Simulate
for i_turn in range(sim_content.N_turns):
    print('%s Turn %d' % (time.strftime("%d/%m/%Y %H:%M:%S", time.localtime()), i_turn))

    bunch_monitor.dump(bunch)

    slices = bunch.get_slices(slicer, statistics=['mean_x'])
    slice_monitor.dump(bunch)

    machine.track(bunch)

    if sim_content._check_stop_conditions():
        os.system('touch met_stop_condition')
        break

sim_content._finalize_multijob_mode()
