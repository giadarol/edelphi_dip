import os
import numpy as np

scan_folder_rel = 'simulations_fine'

environment_preparation = f'''
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/venvs/py3/bin/activate
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/setpythopath
PYTHONPATH=$PYTHONPATH:{os.path.abspath('../')}
'''
# Last one is to get response matrix path

strength_scan = np.arange(0., 2., 0.01)

files_to_be_copied = [
        '../008_eigenvalues/002_build_mode_matrix.py',
        '../008_eigenvalues/mode_coupling_matrix.py',
        ]

settings_to_be_replaced_in = '002_build_mode_matrix.py'


scan_folder_abs = os.getcwd() + '/' + scan_folder_rel
os.mkdir(scan_folder_abs)

# prepare scan
for ii in range(len(strength_scan)):

    # Make directory
    current_sim_ident= f'strength_{strength_scan[ii]:.3f}'
    print(current_sim_ident)
    current_sim_abs_path = scan_folder_abs+'/'+current_sim_ident
    os.mkdir(current_sim_abs_path)

    # Copy files
    for ff in files_to_be_copied:
        os.system(f'cp {ff} {current_sim_abs_path}')

    # Replace settings section
    settings_section = f'''# start-settings-section

# Reference
beta_fun_rescale = 93.2  #92.7
l_min = -7
l_max = 7
m_max = 39
n_phi = 3*360
n_r = 3*200
N_max = 49
Qp = 0.
alpha_N_custom = []
n_tail_cut = 0
save_pkl_fname = 'mode_coupling_matrix.pkl'
response_matrix_file = '../../../001_sin_response_scan/response_data_y_processed.mat'
z_strength_file = '../../../000a_sin_response_unperturbed_pinch/linear_strength_y.mat'
detuning_fit_order = 20
include_detuning_with_long_amplitude = False
pool_size = 0
flag_solve_and_plot = False

omega0 = 2*np.pi*clight/27e3 # Revolution angular frquency
omega_s = 4.9e-3*omega0
Q_full = 60.295 #62.27
sigma_b = 0.097057
r_b = 4*sigma_b

a_param = 8./r_b**2
cloud_rescale_by = {strength_scan[ii]:.5e}

# end-settings-section'''

    # sim_param_amend_curr= f''' '''

    # with open(current_sim_abs_path
    #     + '/Simulation_parameters_amend_for_matrixsim.py', 'w') as fid:
    #     fid.write(sim_param_amend_curr)

    with open(current_sim_abs_path+'/'+settings_to_be_replaced_in, 'r') as fid:
        lines = fid.readlines()
    istart = np.where([ss.startswith('# start-settings-section') for ss in lines])[0][0]
    iend = np.where([ss.startswith('# end-settings-section') for ss in lines])[0][0]
    with open(current_sim_abs_path+'/'+settings_to_be_replaced_in, 'w') as fid:
        fid.writelines(lines[:istart])
        fid.write(settings_section + '\n')
        fid.writelines(lines[iend+1:])

    # Prepare job script
    job_content = f'''#!/bin/bash

{environment_preparation}

# Environment preparation
echo PYTHONPATH=$PYTHONPATH
echo which python
which python

cd {current_sim_abs_path}

python 002_build_mode_matrix.py
'''
    with open(current_sim_abs_path + '/job.job', 'w') as fid:
       fid.write(job_content)

# Prepare htcondor cluster of jobs
import PyPARIS_sim_class.htcondor_config as htcc
htcc.htcondor_config(
        scan_folder_abs,
        time_requirement_days=0.1,
        htcondor_files_in=scan_folder_abs)
