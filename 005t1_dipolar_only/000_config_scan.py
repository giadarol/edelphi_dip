import os
import numpy as np

scan_folder_rel = 'simulations_long'

environment_preparation = f'''
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/venvs/py3/bin/activate
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/setpythopath
PYTHONPATH=$PYTHONPATH:{os.path.abspath('../')}
'''
# Last one is to get response matrix path

strength_scan = np.arange(0.02, 2.01, 0.02)

files_to_be_copied = [
        '../004_instability_simulation/000_simulation_matrix_map.py',
        '../004_instability_simulation/run_it_several_times',
        ]

settings_to_be_replaced_in = '000_simulation_matrix_map.py'


scan_folder_abs = os.getcwd() + '/' + scan_folder_rel
os.mkdir(scan_folder_abs)

# prepare scan
for ii in range(len(strength_scan)):

    # Make directory
    current_sim_ident= f'strength_{strength_scan[ii]:.2e}'
    print(current_sim_ident)
    current_sim_abs_path = scan_folder_abs+'/'+current_sim_ident
    os.mkdir(current_sim_abs_path)

    # Copy files
    for ff in files_to_be_copied:
        os.system(f'cp {ff} {current_sim_abs_path}')

    # Replace settings section
    settings_section = f'''# start-settings-section
n_terms_to_be_kept = 200
n_tail_cut = 10
recenter_all_slices = True # Cancels initial kick from input

Qp_x = 0.
Qp_y = 0.
plane = 'y'

ecloud_strength_scale = {strength_scan[ii]:e}

sim_param_file = '../../../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = ['../../../Simulation_parameters_amend.py',
                    'Simulation_parameters_amend_for_matrixsim.py']

include_response_matrix = True
response_data_file = '../../../001_sin_response_scan/response_data_y_processed.mat'

include_detuning_with_z = False
only_phase_shift = False
add_alpha_0_to_tune = False
factor_alpha_0_to_tune = 0.
z_strength_file = '../../../000a_sin_response_unperturbed_pinch/linear_strength_y.mat'
detuning_fit_order = 20
N_poly_cut = detuning_fit_order + 1
alpha_N_custom = []

include_non_linear_map = False
flag_wrt_bunch_centroid = False
field_map_file = '../../../003_generate_field_map/field_map_lin.mat'
# end-settings-section'''

    sim_param_amend_curr= f'''
N_turns = 8000

enable_transverse_damper = False
V_RF = 6e6
longitudinal_mode = 'linear'
'''

    with open(current_sim_abs_path
        + '/Simulation_parameters_amend_for_matrixsim.py', 'w') as fid:
        fid.write(sim_param_amend_curr)

    with open(current_sim_abs_path+'/'+settings_to_be_replaced_in, 'r') as fid:
        lines = fid.readlines()
    istart = np.where([ss.startswith('# start-settings-section') for ss in lines])[0][0]
    iend = np.where([ss.startswith('# end-settings-section') for ss in lines])[0][0]
    with open(current_sim_abs_path+'/'+settings_to_be_replaced_in, 'w') as fid:
        fid.writelines(lines[:istart])
        fid.write(settings_section + '\n')
        fid.writelines(lines[iend+1:])

    # Prepare run script
    run_script_curr= '''#!/bin/bash
for i in {1..1}
do
   echo "Iteration $i"
   if test -f "met_stop_condition"; then
	   echo "Met stop condition!"
	   break
   fi
   python 000_simulation_matrix_map.py
done
    '''
    with open(current_sim_abs_path + '/run_it_several_times', 'w') as fid:
       fid.write(run_script_curr)

    # Prepare job script
    job_content = f'''#!/bin/bash

{environment_preparation}

# Environment preparation
echo PYTHONPATH=$PYTHONPATH
echo which python
which python

cd {current_sim_abs_path}

bash run_it_several_times
'''
    with open(current_sim_abs_path + '/job.job', 'w') as fid:
       fid.write(job_content)

# Prepare htcondor cluster of jobs
import PyPARIS_sim_class.htcondor_config as htcc
htcc.htcondor_config(
        scan_folder_abs,
        time_requirement_days=2., # 120 minutes
        htcondor_files_in=scan_folder_abs)
