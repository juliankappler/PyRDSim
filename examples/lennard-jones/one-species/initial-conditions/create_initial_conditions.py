#!/usr/bin/env python

import pyrdsim

N_particles = {0:200}

particle_types = {0:[0]}

parameters = {
	'parameters_filename':'parameters.txt',
	'trajectory_filename':'trajectory.h5',
}

simulation = pyrdsim.system(parameters=parameters)


simulation.create_initial_conditions(N_particles=N_particles,
									particle_types=particle_types)


initial_conditions_filename ='initial_conditions.txt'
simulation.write_current_state_to_text_file(
							filename=initial_conditions_filename)
