#!/usr/bin/env python

import pyrdsim


parameters = {
	'parameters_filename':'parameters_equilibration.txt',
}


simulation = pyrdsim.system(parameters=parameters)
result = simulation.simulate(verbose=False)

initial_conditions_filename = 'initial_conditions.txt'
simulation.write_current_state_to_text_file(
						filename=initial_conditions_filename)
