#!/usr/bin/env python

import numpy as np
from pathlib import Path
import shutil # https://stackoverflow.com/a/123238

n_binding_sites = [2,3,4,5] # patches for the patchy colloids

N_molecules = 125 # number of molecules in the box

reactions = ['no-reactions',
            'linear',
            'catalytically-deactivated',
            ]

output_dir = './'
source_dir = './starting-files/'
source_dir_initial_conditions = '{0}/initial-conditions/'.format(source_dir)

source_files_static = [
            'run_simulation.py',
            'run_equilibration.py',
            'run_all.sh',
            'run_initial_condition_and_equilibration.sh',
            'parameters_equilibration.txt',
            'nonbonded_interactions.txt',
              ]
source_files = ['parameters.txt',
                'bonded_interactions.txt',
                'reference_configuration.txt',
                ]
source_files_initial_conditions_static = [
                        'reactions.txt',
                        'parameters.txt',
                                ]


def copy_files(
            source_dir,
            target_dir,
            file_list,
            static=True,
            nonstatic_filename=None,
            ):
    #
    if not static:
        if nonstatic_filename is None:
            raise RuntimeError("Please set variable nonstatic_filename")
    #
    #
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    #
    for file in file_list:
        #
        target_file = '{0}/{1}'.format(target_dir,file)
        #
        if static:
            #
            source_file = '{0}/{1}'.format(source_dir,file)
            #
        else:
            #
            subdir = file.split('.')[0]
            source_file = '{0}/{1}/{2}'.format(source_dir,
                                        subdir,
                                        nonstatic_filename)
            #
        shutil.copy2(source_file,target_file)
    #



###########################################################
# For creating the initial conditions for each simulation #
###########################################################
def create_initial_conditions_script(target_dir,
                        N_molecules,
                        n_binding_sites,
                        filename='create_initial_conditions.py'):
    #
    s = ',1'*n_binding_sites # binding sites are particle type 1 by default
    string = """#!/usr/bin/env python

import pyrdsim

N_particles = {{0:{N_molecules}}}

particle_types = {{0:[0{s}]}}

reference_configurations = {{0:'../reference_configuration.txt'}}

parameters = {{
	'parameters_filename':'parameters.txt',
	'trajectory_filename':'trajectory.h5',
}}


simulation = pyrdsim.system(parameters=parameters)

simulation.create_initial_conditions(N_particles=N_particles,
							particle_types=particle_types,
                            reference_configurations=reference_configurations)

initial_conditions_filename ='initial_conditions.txt'
simulation.write_current_state_to_text_file(filename=initial_conditions_filename)
""".format(N_molecules=N_molecules,s=s)
    #
    with open(target_dir + '/' + filename,'w') as f:
        f.write(string)
    #



def set_up_single_simulation(N_molecules,
                            n_binding_sites,
                            target_dir,
                            ):
    global source_dir
    global source_dir_initial_conditions
    global source_files_static
    global source_files
    global source_files_initial_conditions
    #
    #d0 = 3.
    #d1 = 1.5*n_binding_sites
    #
    target_dir_initial_conditions = target_dir + '/initial-conditions/'
    #
    # copy main simulation files that are independent
    # of the number of binding sites
    copy_files(source_dir=source_dir,
                target_dir=target_dir,
                file_list=source_files_static)
    #
    # copy main simulation files that depend on the number of binding sites
    copy_files(source_dir=source_dir,
                target_dir=target_dir,
                file_list=source_files,
                static=False,
                nonstatic_filename='{0:d}.txt'.format(n_binding_sites),
                )
    #
    # copy initial conditions files
    copy_files(source_dir=source_dir_initial_conditions,
                target_dir=target_dir_initial_conditions,
                file_list=source_files_initial_conditions_static)
    #
    # copy reaction definitions
    source_file = '{0}/reactions/{1}.txt'.format(source_dir,reaction)
    target_file = '{0}/reactions.txt'.format(target_dir)
    shutil.copy2(source_file,target_file)
    #
    #
    create_initial_conditions_script(target_dir=target_dir_initial_conditions,
                                    N_molecules=N_molecules,
                                    n_binding_sites=n_binding_sites)
    #


for n_bs in n_binding_sites:
    for reaction in reactions:
        #
        target_dir = '{0}/{1}-binding-sites/{2}/'.format(output_dir,
                                                        n_bs,
                                                        reaction)
        target_dir_initial_conditions = '{0}/initial-conditions/'.format(
                                                        target_dir)
        #
        set_up_single_simulation(n_binding_sites=n_bs,
                                N_molecules=N_molecules,
                                target_dir=target_dir,
                                )
                #


