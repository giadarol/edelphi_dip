import os
import numpy as np

scan_folder_rel = 'simulations_x'

environment_preparation = '''
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/venvs/py3/bin/activate
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/setpythopath
'''

N_samples = 200
sin_amplitude = 1e-4


files_to_be_copied = [
        '../000_response_to_single_sinusoid/000_response_to_sin.py',
        '../000_response_to_single_sinusoid/Simulation_parameters_amend_for_sin_response.py'
        ]

settings_to_be_replaced_in = '000_response_to_sin.py'

# Generate configurations
assert(N_samples % 2 ==0)
cos_ampl_list = []
sin_ampl_list = []
n_osc_list = []
for ii in range(N_samples//2):
    cos_ampl_list.append(sin_amplitude)
    sin_ampl_list.append(0.)
    n_osc_list.append(ii)

    cos_ampl_list.append(0.)
    sin_ampl_list.append(sin_amplitude)
    n_osc_list.append(ii+1)

scan_folder_abs = os.getcwd() + '/' + scan_folder_rel
os.mkdir(scan_folder_abs)

# prepare scan
for ii in range(len(n_osc_list)):

    # Make directory
    current_sim_ident= f'n_{n_osc_list[ii]:.1f}_c{cos_ampl_list[ii]:.2e}_s{sin_ampl_list[ii]:.2e}'
    print(current_sim_ident)
    current_sim_abs_path = scan_folder_abs+'/'+current_sim_ident
    os.mkdir(current_sim_abs_path)

    # Copy files
    for ff in files_to_be_copied:
        os.system(f'cp {ff} {current_sim_abs_path}')

    # Replace settings section
    settings_section = f'''# start-settings-section
cos_amplitude = {cos_ampl_list[ii]:e}
sin_amplitude = {sin_ampl_list[ii]:e}
N_oscillations = {n_osc_list[ii]:e}

plane = 'x'

flag_no_bunch_charge = False
flag_plots = False

sim_param_file = '../../../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = [
        '../../../Simulation_parameters_amend.py',
        'Simulation_parameters_amend_for_sin_response.py']

# end-settings-section'''

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

python {settings_to_be_replaced_in}

'''
    with open(current_sim_abs_path + '/job.job', 'w') as fid:
       fid.write(job_content)

# Prepare htcondor cluster of jobs
import PyPARIS_sim_class.htcondor_config as htcc
htcc.htcondor_config(
        scan_folder_abs,
        time_requirement_days=120./(24*60.), # 120 minutes
        htcondor_files_in=scan_folder_abs)
