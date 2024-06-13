# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
# cython: boundscheck=False

import numpy as np
cimport numpy as np
import functools # 
import os
cimport cpython
DTYPE   = np.float_
ctypedef np.float_t DTYPE_t
import h5py
import copy
from scipy.spatial import distance
from libc.math cimport isnan
from numpy.math cimport INFINITY
import warnings
from libc.time cimport time,time_t # see https://stackoverflow.com/a/39043290
from libc.math cimport round
cdef extern from "math.h":
	double sqrt(double x) nogil
cdef extern from "math.h":
	double exp(double x) nogil
from libc.math cimport acos
from libc.math cimport tanh
import scipy.linalg
import cma
from libcpp.vector cimport vector


cdef extern from "<random>" namespace "std":
	# Wrapper for C++11 pseudo-random number generator
	# This implementation is similar to the one used in PyRoss (where I 
	#	contributed the stochastic simulations), see
	#		https://github.com/rajeshrinet/pyross/
	cdef cppclass mt19937:
		mt19937()
		mt19937(unsigned long seed)

	cdef cppclass normal_distribution[T]:
		normal_distribution()
		normal_distribution(T a, T b)
		T operator()(mt19937 gen)

	cdef cppclass uniform_real_distribution[T]:
		uniform_real_distribution()
		uniform_real_distribution(T a, T b)
		T operator()(mt19937 gen)


cdef class random_numbers:
	"""
	Random number generator
	
	This implementation is similar to the one used in PyRoss (where I 
	contributed the stochastic simulations), see
		https://github.com/rajeshrinet/pyross/ 


	Methods
	-------
	random_standard_normal
	uniform_dist
	initialize_random_number_generator
	""";
	cdef:
		mt19937 gen
		long seed


	cdef random_standard_normal(self):
		'''
		Generates and returns standard normal random number.
		'''
		cdef:
			normal_distribution[double] dist = normal_distribution[double](0.0,1.0)
		return dist(self.gen)

	cdef uniform_dist(self):
		'''
		Draws random sample X from uniform distribution on (0,1)
		'''
		cdef:
			uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0,1.0)
		return dist(self.gen)

	cpdef initialize_random_number_generator(self,long supplied_seed=-1):
		'''
		Sets seed for random number generator.
		If negative seed is supplied, a seed will be generated based
		on process ID and current time.

		Parameters
		----------
		supplied_seed: long
			Seed for random number generator

		Returns
		-------
		None
		'''
		cdef:
			long max_long = 9223372036854775807
		if supplied_seed < 0:
			self.seed = (abs(os.getpid()) + long(time(NULL)*1000)) % max_long
		else:
			self.seed = supplied_seed % max_long
		self.gen = mt19937(self.seed)




cdef class system(random_numbers):
	'''
	Main simulation class.

	Derives its random number generator from parent class random_numbers

	Methods
	-------
	1. initialization + processing of parameters:
		__init__
		set_default_parameters
		set_system_parameters
		set_parameters
		preprocess_line
		load_parameters_file
		load_bonded_interactions
		create_nonbonded_interaction_functions
		add_interaction_partners_for_nonbonded_interactions
		add_interaction_partners_for_reactions
		add_pair_to_interaction_partner_list
		create_bond_and_angle_force_functions
	2. neighbor lists & particle distance calculation:
		set_up_neighbor_list_cells
		create_neighbor_list
		calculate_distances
	3. IO and storage for trajectories:
		create_arrays
		load_trajectory_from_file
		append_trajectory_to_file
		load_initial_conditions_from_file
		write_current_state_to_text_file
	4. simulation (evaluate drift, execute reactions, etc):
		update_drifts
		reaction_step
		update_positions
		apply_boundary_conditions
		integration_step
		simulate
	5. creation of initial conditions:
		create_reference_molecule
		place_configuration_randomly
		create_initial_conditions

	''';

	cdef:
		#
		# spatial simulation parameters
		int N_dim # spatial dimensions for the simulation
		np.ndarray L # length scale
		np.ndarray periodic
		#
		double max_rc
		#
		# temporal simulation parameters
		double dt
		int N_steps
		int N_out
		int N_steps_completed
		int stride
		int N_traj # = N_out//stride + 1
		int neighbor_list_update_frequency
		double rs
		double min_dist
		#
		# model parameters
		int N_species # number of different atom species
		int N_particles # number of particles, 1D int array with
							   # len = N_species
		int N_molecules
		np.ndarray x # current positions
		np.ndarray a # current positions
		np.ndarray trajectory # current positions
		np.ndarray trajectory_types
		#
		np.ndarray particle_types
		dict bonded_neighbor_list
		dict neighbor_list
		list neighbor_list_keys
		np.ndarray molecule_ids
		np.ndarray first_index_of_molecule
		#
		dict molecule_parameters
		dict molecule_interactions
		dict N_particles_per_molecule
		#
		int neighbor_list_rc_to_cell_length_ratio
		np.ndarray neighbor_list_cell_id
		np.ndarray d_cell_id
		np.ndarray neighbor_list_cell_length
		np.ndarray neighbor_list_number_of_cells_along_axis
		np.ndarray neighbor_list_pbc_jump_cutoff_along_axis
		#
		np.ndarray xx, xx_hat, d_xx
		#
		np.ndarray tau_by_tauD
		#
		np.ndarray noise_prefacs, force_prefacs
		#
		double angle_pi_approximation_threshold
		#
		list bond_interaction_functions
		list angle_interaction_functions
		#
		dict interaction_partners # dictionary such that 
		# interaction_partners[i] = [list of those molecule types with which 
		# molecule type i interacts]
		#
		np.ndarray reaction_rates
		np.ndarray total_reaction_rates
		int N_reactions
		#
		#############
		# Filenames #
		#############
		str parameters_filename
		str bint_filename
		str nbint_filename
		str reactions_filename
		str initial_conditions_filename
		str trajectory_filename
		#
		set comment_chars
		list interactions, reactions
		#
		################################################
		# Definitions used for loading parameter files #
		################################################
		# general system and model parameters
		str parameters_section_keyword
		dict parameters_datatypes
		list parameters_int_arrays
		list parameters_float_arrays
		#
		# nbint = nonbonded interactions
		str nbint_section_keyword
		dict nbint_datatypes
		list nbint_int_arrays
		list nbint_float_arrays
		# 
		# reactions
		str reactions_section_keyword
		dict reactions_datatypes
		list reactions_int_arrays
		list reactions_float_arrays
		#


	################################################
	# 1. initialization + processing of parameters #
	################################################

	def __init__(self, parameters={}):
		cdef:
			int i, e

		# set the default simulation/model parameters
		self.set_default_parameters()

		# set the system parameters (which cannot be modified)
		self.set_system_parameters()
		
		# process simulation/model parameters that have either been 
		# passed when creating the instance, or are stored in a text
		# file
		try:
			parameters_filename = parameters['parameters_filename']
			self.parameters_filename = parameters_filename
		except KeyError:
			pass # use default value 'parameters.txt'
		try:
			# try to load the file with the parameters
			parameters_loaded = self.load_parameters_file(
							filename=self.parameters_filename,
							section_keyword=self.parameters_section_keyword,
							datatypes=self.parameters_datatypes,
							int_arrays=self.parameters_int_arrays,
							float_arrays=self.parameters_float_arrays)[0]
			# note that we choose the element [0], because self.load_parameters
			# returns a list with a single element, namely the dictionary
			# with the loaded parameters
		except FileNotFoundError:
			# if the file we are trying to load does not exist, we have not
			# loaded any parameters
			parameters_loaded = {}
		
		# any parameters passed to self.__init__() override the corresponding
		# parameters from the parameters file
		for key, value in parameters.items():
			parameters_loaded[key.lower()] = value

		self.set_parameters(parameters=parameters_loaded)
		

	def set_default_parameters(self):
		'''
		Set default parameters (for parameters than can be changed by the user)

		This method is called when the class is instantiated, i.e. in __init__
		''';
		#
		self.parameters_filename = 'parameters.txt'
		self.bint_filename = './bonded_interactions.txt'
		self.nbint_filename = './nonbonded_interactions.txt'
		self.reactions_filename = './reactions.txt'
		self.initial_conditions_filename = './initial_conditions.txt'
		self.trajectory_filename = './trajectory.h5'
		#
		self.max_rc = 0.
		self.interaction_partners = {}
		self.N_species = 0
		self.stride  = 1
		self.neighbor_list_update_frequency = 10
		self.comment_chars = set(['#',';'])
		self.neighbor_list_rc_to_cell_length_ratio = 2
		#
		# by default, there are no reactions
		self.N_reactions = 0
		self.reaction_rates = np.array([[]],
								dtype=DTYPE)
		self.total_reaction_rates = np.array([],
								dtype=DTYPE)
		
		

	def set_system_parameters(self):
		'''
		Set fixed class parameters (for parameters that should not be changed 
		by the user)

		This method is called when the class is instantiated, i.e. in __init__

		The dictionaries and lists instantiated below define how the various
		text files that are used for model definition are processed. So if you
		want to add a feature that requires adding information to the 
		parameters file, the nonbonded interactions definitions, or the
		reactions definitions, then you probability want to modify something
		here.
		''';
		#

		self.angle_pi_approximation_threshold = 0.1
		#
		#
		################################################
		# Definitions used for loading parameter files #
		################################################
		# 
		# general system and model parameters
		#
		self.parameters_section_keyword = ''
		self.parameters_datatypes = {'N_dim':int,'dt':float,'N_steps':int,
					'stride':int,
					'N_out':int,
					'bonded_interactions_filename':str,
					'nonbonded_interactions_filename':str,
					'reactions_filename':str,
					'initial_conditions_filename':str,
					'trajectory_filename':str,
					'neighbor_list_update_frequency':int,
					'rs':float,
					'cell_length_ratio':int}
		self.parameters_int_arrays = []
		self.parameters_float_arrays = ['L','periodic','tau_by_tauD']
		#
		#
		# nbint = nonbonded interactions
		self.nbint_section_keyword='[interaction]'
		self.nbint_datatypes =  {'type':str,
				'u0':float,'r0':float,'rc':float,
				'dr':float,
				'a':float,
				'n':int,
				'particle_type':int,
				'direction':int,
				'position':float,
				}
		self.nbint_int_arrays = ['pair']
		self.nbint_float_arrays = []
		# 
		# reactions
		self.reactions_section_keyword='[reaction]'
		self.reactions_datatypes =  {'type':str,'rate':float,'reactant':int,
						'product':int,'r0':float,'catalyst':int,
						'rate_decrease':float,'inhibitor':int,
						'inhibitor2':int}
		self.reactions_int_arrays = []
		self.reactions_float_arrays = []
		#
		#


	
	def set_parameters(self,parameters):
		'''
		Set the parameters given by a dictionary "parameters"

		This function is called from self.__init__(), using parameters that are
		loaded from a text file. There is usually no need for the user to call
		this function explicitly.

		Arguments
		---------
		parameters: dictionary
			This dictionary has as keys strings with the names of parameters,
			and as values the corresponding values. Most keys are mandatory.
			

		''';

		try:
			self.comment_chars = parameters['comment_chars']
		except KeyError:
			pass

		# Set seed for pseudo-random number generator (if provided)
		try:
			self.initialize_random_number_generator(
								  supplied_seed=parameters['seed'])
		except KeyError:
			self.initialize_random_number_generator()

		######################
		# spatial parameters #
		######################
		self.N_dim = parameters['n_dim'] # dimension of space
		self.L = parameters['l'] # array with the box dimensions
		# check if a length is given for each dimension
		if len(self.L) != self.N_dim:
			raise RuntimeError(
				"Number of box edge lenghts {0}".format(len(self.L))\
				+ " is inconsistent with N_dim = {0}.".format(self.N_dim)\
				+ " Please provide one length per dimension."
								)
		try:
			self.periodic = parameters['periodic']
		except KeyError:
			# if the periodicity is not defined by the input dictionary, we 
			# set the box periodic in all dimensions
			self.periodic = np.ones(self.N_dim,dtype=DTYPE)
		# check if the periodicity definition is compatible with the number
		# of dimensions given
		if len(self.periodic) != self.N_dim:
			raise RuntimeError(
				"Periodicity definition {0}".format(self.periodic)\
				+" is inconsistent with n_dim = {0}. ".format(self.N_dim)\
				+"Please provide one periodicity value per spatial dimension.")


		
		#######################
		# temporal parameters #
		#######################
		self.dt         = parameters['dt'] # in units of tau
		self.N_steps    = parameters['n_steps']
		self.N_out    = parameters['n_out']
		try:
			self.stride    = parameters['stride']
		except KeyError:
			pass
		self.N_traj = self.N_out//self.stride # +1 for initial condition
		

		#############
		# frictions #
		#############
		try:
			self.tau_by_tauD = np.array(parameters['tau_by_taud'],dtype=DTYPE)
		except KeyError:
			if self.N_species == 0: # if no particle species have been defined,
				# we define a single species with unit friction
				self.tau_by_tauD = np.array([1.],dtype=DTYPE)
		self.N_species = len(self.tau_by_tauD)
		#self.tauDs_by_tau = 1./np.array(self.tauD_by_tau,dtype=DTYPE)
		#
		mask = (self.tau_by_tauD < 0)
		self.noise_prefacs = np.ones_like(self.tau_by_tauD,dtype=DTYPE)
		self.force_prefacs = np.ones_like(self.tau_by_tauD,dtype=DTYPE)
		#
		self.noise_prefacs[mask] = 0.
		self.noise_prefacs[~mask] = np.sqrt( 2.*self.dt*self.tau_by_tauD[~mask] )
		#
		self.force_prefacs[mask] = 0.
		self.force_prefacs[~mask] = self.dt*self.tau_by_tauD[~mask]
		

		#############
		# filenames #
		#############
		# Note: default filenames are defined in self.set_default_parameters()
		try: 
			self.bint_filename = parameters['bonded_interactions_filename']
		except KeyError:
			pass

		try:
			self.nbint_filename = parameters['nonbonded_interactions_filename']
		except KeyError:
			pass
		
		try:	
			self.reactions_filename = parameters['reactions_filename']
		except KeyError:
			pass
		
		try:
			self.initial_conditions_filename = \
								parameters['initial_conditions_filename']
		except KeyError:
			pass

		try:
			self.trajectory_filename = parameters['trajectory_filename']
		except KeyError:
			pass


		####################################################
		# load and set interaction and reaction parameters #
		####################################################

		# load and set nonbonded interactions
		try:
			# if a dictionary with nonbonded interactions is provided in the
			# parameters dictionary, we use this dictionary
			self.interactions = parameters['interactions'] 
		except KeyError:
			# otherwise we load the nonbonded interactions from the file
			# self.nbint_filename
			self.interactions = self.load_parameters_file(
					filename=self.nbint_filename,
					section_keyword=self.nbint_section_keyword,
					datatypes=self.nbint_datatypes,
					int_arrays=self.nbint_int_arrays,
					float_arrays=self.nbint_float_arrays)
		self.create_nonbonded_interaction_functions()
		self.add_interaction_partners_for_nonbonded_interactions()
		
		# load and set reactions
		try:
			# if a dictionary with reactions is provided in the
			# parameters dictionary, we use this dictionary
			self.reactions = parameters['reactions']
		except KeyError:
			# otherwise we load the reactions from the file
			# self.reactions_file
			self.reactions = self.load_parameters_file(
						filename=self.reactions_filename,
						section_keyword=self.reactions_section_keyword,
						datatypes=self.reactions_datatypes,
						int_arrays=self.reactions_int_arrays,
						float_arrays=self.reactions_float_arrays)
		self.N_reactions = len(self.reactions)
		self.add_interaction_partners_for_reactions()

		# load bonded interactions
		self.load_bonded_interactions()

		# cell length ratio used for the neighbor list algorithm
		try:
			self.neighbor_list_rc_to_cell_length_ratio = \
							parameters['cell_length_ratio']
		except KeyError:
			pass
		
		try:
			self.rs =  parameters['rs'] # cutoff for distance calculation
		except KeyError:
			self.rs =  2*self.max_rc
		#
		try:
			self.neighbor_list_update_frequency = \
						parameters['neighbor_list_update_frequency']
		except KeyError:
			pass
			

	def preprocess_line(self,line):
		'''
		Preprocess a line that is being loaded from a text file

		This is used for parsing parameter files in the methods
		self.load_parameters_file
		self.load_bonded_interactions
		''';
		#
		line = line.lower() # turn into lower case
		#
		# remove comments
		for c in self.comment_chars:
			# if a comment char is found in the line, everything after
			# the first occurence of the comment char is removed
			if c in line:
				line = line.split(c)[0]
		#
		# replace tabs with a space, and remove the newline character
		line = line.replace('\t',' ') 
		line = line.replace('\n','') 
		# make a copy of the line that does not contain any spaces
		line_nospace = line.replace(' ','')
		#
		return line, line_nospace

	cdef load_parameters_file(self,
						filename,
						datatypes,
						int_arrays=[],
						float_arrays=[],
						section_keyword='',
						):
		'''
		Load parameters from a text file

		Any parameter can be either
		- an arrays of integers, 
		- an array of floats, or
		- a single value of any dataype (e.g. float, int, string)

		parameters are set by
		name_of_parameter = value_of_parameter
		in the text file. There is an option to have sections.
		
		Examples:
		1. Parameter file without sections:
		Input text file:
			N_dim = 3 # dimension
			N_steps = 100 # number of timesteps
			periodic = 1 -1 1 # periodicity for each dimension
		To load this file, one would pass the arguments:
			datatypes = {'n_dim':int,'n_steps':int}
			int_arrays = ['periodic']
			float_arrays = []
		Then the function returns a list with one dictionary:
			output_list_of_dictionaries = [
					{'n_dim':3,
					'n_steps':100,
					'periodic':np.array([1,-1,1],dtype=int)
					}]

		2. Parameter file with sections:
		Input text file:
			[section]
			pair = 1 3
			U0 = 100
			[section]
			pair = 2 2
			U0 = 25.
		To load this file, one would pass the arguments:
			section_keyword='[section]'
			datatypes = {'u0':float}
			int_arrays = ['pair']
			float_arrays = []
		Then the function returns a list with two dictionaries:
			output_list_of_dictionaries = [
					{'pair':np.array([1,3],dtype=int),'u0':100.},
					{'pair':np.array([2,2],dtype=int),'u0':25.} ]


		Arguments
		---------
		filename: string
			Filename of the text file containing the parameters
		datatypes: dictionary
			Dictionary that defines the data type of every parameter that
			is not an array. E.g.
				datatypes['N_out'] = int
			means that the parameters file should contain a line like e.g.
				N_out = 10
			where the value 10 is chosen as an example (could be any int).
			Note that the names of the data types are not case sensitive, i.e.
			writing n_out = 10 or N_OUT = 10 in the parameters file leads to
			the same results.
		int_arrays: list of strings
			List that defines the parameters that should be loaded as integer
			arrays. E.g. int_arrays = ['pair'] means that in the
			parameters file there should be a line
				pair = 1 5
			with several integers, separated by spaces.
		float_arrays: list of strings
			List that defines the parameters that should be loaded as floats
			arrays. E.g. float_arrays = ['l','periodic'] means that in the
			parameters file there should be lines
				l = 3.5 2. 10
				periodic = 1 -1 1
			with entries that can be casted to floats, separated by spaces.
		section_keyword: string
			String that defines the beginning of a new section in the 
			parameter file. This is relevant for loading parameter files that
			contain interactions or reactions, and separates the various 
			interactions/reactions that can be contained in a single file.
			Default values is section_keyword = '' (i.e. empty string), which
			we interpret as "no section keyword provided"

		Return:
		-------
		output_list_of_dictionaries: list of dictionaries
			len(output_list_of_dictionaries) = number of sections found
		
		''';
		#
		# transform the names of all provided datatypes, integer arrays, and
		# float arrays to lower case
		datatypes_lower = {k.lower():v for k,v in datatypes.items()}
		int_arrays_lower = [e.lower() for e in int_arrays]
		float_arrays_lower = [e.lower() for e in float_arrays]
		#
		output_list_of_dictionaries = []
		current_parameter_dictionary = {}
		#
		if len(section_keyword) > 0:
			# if a section keyword has been provided, we will only start
			# processing parameters once the keyword has been found for the
			# first time
			process_line = False
		else:
			# if no section keyword has been provided, we start processing
			# the parameters in the parameters file from the first line 
			# onwards
			process_line = True
		
		#
		with open(filename,'r') as f:
			# open file and go through it line by line
			for line in f:
				#
				line, line_nospace = self.preprocess_line(line=line)
				#
				# if a section keyword has been provided, we check if the
				# current line contains the keyword
				if len(section_keyword) > 0:
					if section_keyword in line_nospace:
						# if the keyword appears in the current line, we start
						# processing the data in the parameter file from that
						# line onwards
						process_input = True
						#
						# if the dictionary current_parameter_dictionary has 
						# entries (which means that the section that just 
						# started is not the first section, and we have 
						# processed parameters from the previous section),
						# then we append the already processed parameters
						# from the previous section to the output list 
						if len(current_parameter_dictionary) > 0:
							output_list_of_dictionaries.append(\
										current_parameter_dictionary\
															)
							# we empty the dictionary again, to store the
							# parameters from the current section
							current_parameter_dictionary = {}
				#
				# only if there is a '=' symbol in the current line, we
				# potentially process it
				if '=' in line_nospace:
					#
					# there should only be a single '=' symbol in the line.
					line_nospace = line_nospace.split('=')
					if len(line_nospace) != 2:
						raise RuntimeError(\
							"Current line is not of the form"\
							+" <name of parameters> = <value of parameter>:"\
							+"\n{0}".format(line)\
									)
					line = line.split('=')
					#
					if process_input:
						# if we process the input, we have to check whether
						# the parameter stores an integer array, a float array,
						# or a single value
						#
						if line_nospace[0] in int_arrays_lower:
							# integer array
							value = np.array(
										[int(i) for i in line[1].split()],
											dtype=int)
							#
						elif line_nospace[0] in float_arrays_lower:
							# float array
							value = np.array(
										[float(i) for i in line[1].split()],
											dtype=float)
							#
						else:
							# individual value, cast to provided datatype
							try:
								value = \
									datatypes_lower[line_nospace[0]](line_nospace[1])
							except KeyError:
								raise RuntimeError("Unrecognized entry in"\
									+" parameters file: {0}".format(
										line_nospace[0]
									))
						#
						# append new array/value to dictionary
						current_parameter_dictionary[line_nospace[0]] = value
					else:
						# If a section keyword has been provided, and there is
						# a parameter defined before the keyword appears for
						# the first time, we throw an error
						raise RuntimeError("Parameter outside of section.")
					#
				#
			#
		#
		if len(current_parameter_dictionary) > 0:
			# if current_parameter_dictionary has any elements, append it to 
			# the output_list_of_dictionaries
			output_list_of_dictionaries.append(current_parameter_dictionary)
		#
		return output_list_of_dictionaries

	def load_bonded_interactions(self):
		'''
		Load bonded interactions from a text file

		The name of the text file is defined by the class variable
			self.bint_filename

		The text file contains the properties of each molecule. Here is an 
		example to illustrate the format:

		[ molecule ] # every molecule definition starts with this keyword 
		molecule_id = 0 # each defined molecule needs to have a unique ID
		N_atoms = 3 # number of atoms that the current molecule contains

		[ bonds ] # bonds between the N_atoms atoms 
		; ai  aj    r0   U0 
		0   1     1.0    400.0  # atoms 0 and 1 have a bond with equilibrium
						# length r0 = 1*L and spring constant U0 = 400 kJ/L**2
		0   2     1.0    400.0 # atoms 0 and 2 have a bond with equilibrium
						# length r0 = 1*L and spring constant U0 = 400 kJ/L**2

		[ angles ]
		; ai  aj    theta0  k
		1   0   2  180.    50.0 # the angle between the atoms (1,0,2) has
			# an angle potential with equilibrium angle 180 degrees and a 
			# spring constant k = 50 kJ/degree**2

		[ molecule ] # second molecule, consisting of 4 atoms
		molecule_id = 1
		N_atoms = 4
		[ bonds ]
		; ai  aj    r0   U0
		0   1     1.0    400.0
		0   2     1.0    400.0
		0   3     1.0    400.0
		[ angles ]
		; ai  aj    theta0  k
		1   0   2  120.    100.0
		2   0   3  120.    100.0
		3   0   1  120.    100.0
		
		''';
		#
		# parameters that need to be specified for every molecule, along with
		# their datatypes
		molecule_parameters = {'molecule_id':int,
								'n_atoms':int,
									}
		self.molecule_parameters = {}
		self.molecule_interactions = {}
		#
		cur_params = {} # parameters for current molecule
		cur_interactions = [] # interactions for current molecule
		#
		process_input = False
			
		with open(self.bint_filename,'r') as f:
			#
			for line in f: # iterate through loaded text file line by line
				#
				line, line_nospace = self.preprocess_line(line=line)
				#
				# first check if a new section starts in the current line
				if '[molecule]' in line_nospace:
					# means with the current line, the definition of a new
					# molecule starts
					#
					process_input = True
					#
					if len(cur_params) > 0:
						# if the current molecule is not the first molecule
						# that is being defined, store the previous molecule
						# definition in the output dictionaries
						self.molecule_parameters[cur_params['molecule_id']] \
										= cur_params
						self.molecule_interactions[cur_params['molecule_id']] \
										= cur_interactions
					#
					# reset parameter dictionary and interaction list
					cur_params = {}
					cur_interactions = []
					#
					# by default we are in the "main header" of the molecule,
					# and neither in the bond or angle section
					collecting_interactions_bond = False
					collecting_interactions_angle = False
					#
				elif '[bonds]' in line_nospace:
					# we are entering a section that specifies the bonds
					collecting_interactions_bond = True
					collecting_interactions_angle = False
					#
				elif '[angles]' in line_nospace:
					# we are entering a section that specifies the angles
					collecting_interactions_bond = False
					collecting_interactions_angle = True
					#
				else: 
					pass # no new section starts in the current line
				#
				#
				if not process_input:
					# if we have not yet seen a line that contains the string
					# '[molecule]', we continue with the next line
					continue 
				#
				#
				if collecting_interactions_bond:
					# every line that defines a bond contains four numbers
					# specifying
					# - bonding partner 1,
					# - bonding partner 2,
					# - the equilibrium distance r0 between the partners,
					# - the strength U0 of the bonding potential
					# the numbers are separated by a space, and are of type
					# int, int, float, float
					# 
					# The interaction potential is
					#     U(r) = U0 * (r - r0)**2,
					# where r is the Euclidean distance between the two
					# bonding partners
					try:
						#
						line_split = line.split() 
						#
						if len(line_split) == 4:
							# we only try to process the current line if it
							# contains four values
							#
							# we define a dictionary that contains the 
							# bond defined in the current line
							cur_interaction = {
								'type':'bond',
								'participants':[int(line_split[0]),
												int(line_split[1])],
								'r0':float(line_split[2]),
								'u0':float(line_split[3]),
											}
							#
							# append the current bond to the list of
							# interactions
							cur_interactions.append(cur_interaction)
					except:
						# this error will be thrown for example if the four
						# entries in the current line cannot be cast to 
						# int, int, float, float
						raise RuntimeError("Could not process "\
											+"line: {0}".format(line))
				elif collecting_interactions_angle:
					# every line defines an interaction angle
					#
					try:
						#
						line_split = line.split()
						#
						if len(line_split) == 5:
							# we only try to process the current line if it
							# contains five values
							#
							cur_interaction = {
									'type':'angle',
									'participants':[int(line_split[0]),
													int(line_split[1]),
													int(line_split[2])],
									'theta0':np.pi*float(line_split[3])/180.,
									'u0':float(line_split[4])*(180./np.pi)**2,
										}
							# note that we convert the equilibrium angle and
							# the interaction potential to radian
							#
							# append the current angle interaction to the list of
							# interactions
							cur_interactions.append(cur_interaction)
							#
					except:
						# this error will be thrown for example if the five
						# entries in the current line cannot be cast to 
						# int, int, int, float, float
						raise RuntimeError("Could not process "\
											+"line: {0}".format(line))
				else:
					# if we are neither collecting bond parameters nor angle
					# parameters, we look for other molecule parameters
					#
					for cur_name, dtype in molecule_parameters.items():
						#
						# check if the string cur_name is in the current line
						if cur_name in line_nospace:
							#
							# if it is, we make sure that for the current
							# molecule we have not yet collected a value for 
							# this parameter
							try:
								cur_params[cur_name];
								raise RuntimeError(
									"Repeated molecule parameter {0}".format(
												cur_name) \
									+ " in file {0}".format(
											self.bint_filename
													)
												)
							except KeyError:
								# add parameter to parameters dictionary
								cur_params[cur_name] = \
											dtype(line_nospace.split('=')[1])
						#
					#
				#
		if len(cur_params) > 0:
			self.molecule_parameters[cur_params['molecule_id']] = cur_params
			self.molecule_interactions[cur_params['molecule_id']] = cur_interactions
		#


	cdef create_nonbonded_interaction_functions(self):
		'''
		For every nonbonded interaction, creates a function to evaluate the
		corresponding force strength

		''';
		cdef:
			int i, i0, i1
			dict interaction
		#
		#
		for i, interaction in enumerate(self.interactions):
			#
			interaction = {k.lower():v for k, v in interaction.items()}
			#
			#print(interaction)
			#
			interaction['type'] = interaction['type'].lower()
			current_type = interaction['type']
			#
			if current_type == 'piecewise_linear':
				# for the piecewise_linear force, the cutoff radius
				# is called r0
				interaction['rc'] = interaction['r0']
			#
			# adjust maximal cutoff radius if needed
			if interaction['rc'] > self.max_rc:
				self.max_rc = interaction['rc']
			#	
			#
			# Define interaction function
			if current_type == 'lj': 
				# Lennard-Jones potential
				#
				def func(r,u0,r0,n):
					r6 = (r0/r)**n
					return -2.*n*u0/r*r6*(r6-1.)
				#
				func_partial = functools.partial(func,
									u0=interaction['u0'],
									r0=interaction['r0'],
									n=interaction['n'],
												)
				#
			elif current_type == 'tanh': 
				# tanh interaction potential
				#
				def func(r,u0,r0,a_inv):
					return u0*(1. - tanh((r-r0)*a_inv)**2)*a_inv/2.
				#
				func_partial = functools.partial(func,
									u0=interaction['u0'],
									r0=interaction['r0'],
									a_inv=1./interaction['a'],
												)
				#
			elif current_type == 'yukawa': 
				# yukawa potential
				#
				def func(r,u0,r0):
					return -u0*(1.+r0/r)*exp(-r/r0)/(r*r0)
				#
				func_partial = functools.partial(func,
									u0=interaction['u0'],
									r0=interaction['r0'],
												)
				#
			elif current_type == 'harmonic': 
				# harmonic potential
				#
				def func(r,u0):
					return u0*r
				#
				func_partial = functools.partial(func,
									u0=interaction['u0'],
												)
				#
			elif current_type == 'piecewise_linear':
				# piecewise linear potential
				#
				#
				def func(r,r_inner,U0_by_dr):
					#
					if r < r_inner:
						return 0
					else:
						return U0_by_dr
				#
				func_partial = functools.partial(func,
					# inner cutoff radius for force:
					r_inner=interaction['r0'] - interaction['dr'], 
					# slope of force:
					U0_by_dr=interaction['u0']/interaction['dr'],
												)
				#
			elif current_type == 'wall':
				#
				def func(x,u0,r0,a_inv):
					#
					return u0*(1. - tanh((x-r0)*a_inv)**2)*a_inv/2.
				#
				func_partial = functools.partial(func,
							a_inv = 1./interaction['a'],
							u0 = interaction['u0'],
							r0 = interaction['r0'],
										)
				#
				interaction['pair'] = [interaction['particle_type'],
										interaction['direction']]
			else:
				raise RuntimeError("Interaction {0} not recognized".format(
							current_type
								))
			#
			interaction['func'] = func_partial
			#
			self.interactions[i] = interaction
			#


	cdef add_interaction_partners_for_nonbonded_interactions(self):
		'''
		Add all pairs (i0, i1) of molecule types that have a nonbonded 
		interaction to self.interaction_partners.

		This is important because only pairs that are contained in 
			self.interaction_partners 
		are considered for the creation of self.neighbor_list, see the method
			self.create_neighbor_list.
		The neighbor list self.neighbor_list in turn determines which 
		pairs of particles are considered when calculating nonbonded 
		interactions as well as reactions (that involve more than a single 
		particle)
		''';
		#
		cdef:
			int i
			dict interaction
		
		for i, interaction in enumerate(self.interactions):
			# if the potential is not a wall potential, there is a pair
			# interacting
			if interaction['type'] != 'wall':
				#
				self.add_pair_to_interaction_partner_list(\
										*interaction['pair']
												)
			#

	cdef add_interaction_partners_for_reactions(self):
		'''
		Add all pairs (i0, i1) of molecule types that are involved in locally
		catalyzed/inhibited reactions.

		This is important because only pairs that are contained in 
			self.interaction_partners 
		are considered for the creation of self.neighbor_list, see the method
			self.create_neighbor_list.
		The neighbor list self.neighbor_list in turn determines which 
		pairs of particles are considered when calculating nonbonded 
		interactions as well as reactions (that involve more than a single 
		particle)
		''';
		
		cdef:
			int i
			dict reaction

		for i,reaction in enumerate(self.reactions):
			#
			if reaction['type'] == 'locally_catalyzed':
				self.add_pair_to_interaction_partner_list(
						int(reaction['reactant']),
						int(reaction['catalyst'])
								)
			elif reaction['type'] == 'locally_inhibited':
				#
				self.add_pair_to_interaction_partner_list(
						int(reaction['reactant']),
						int(reaction['inhibitor'])
							)
				#
				try:
					self.add_pair_to_interaction_partner_list(
						int(reaction['reactant']),
						int(reaction['inhibitor2'])
							)
					#
				except KeyError:
					pass
			else:
				pass # linear reactions only involve a single particle


	def add_pair_to_interaction_partner_list(self,i0,i1):
		'''
		Add a pair to the list of interaction partners

		Upon istantiation of the system class, the method 
		self.set_default_parameters() creates an empty class dictionary 
			self.interaction_partners = {}
		
		The purpose of this dictionary is to, for every molecule type i,
		have a list of all the molecule types it interacts with. Interaction
		can here mean either a nonbonded interaction, or a reaction in which
		both molecule types appear.

		So after specifying all interactions and reactions, 
			self.interaction_partners[i] = [list of integers of all the 
											molecule types that molecule 
											type i interacts with]

		''';
		#
		try: # check if a list of interaction partners exists for i0
			self.interaction_partners[i0]
		except KeyError:
			# if not, create an empty set
			self.interaction_partners[i0] = set()
		self.interaction_partners[i0].add(i1)
		#
		# same procedure with i0, i1 interchanged:
		try:
			self.interaction_partners[i1]
		except KeyError:
			self.interaction_partners[i1] = set()
		self.interaction_partners[i1].add(i0)




	cdef create_bond_and_angle_force_functions(self):
		'''
		create lists 
			self.bond_interaction_functions
			self.angle_interaction_functions
		and dictionary
			self.bonded_neighbor_list

		The two lists contain functions that evaluate all the intermolecular
		forces (i.e. bonds and angles). The dictionary is such that 
			self.bonded_neighbor_list[i] 
		''';
		cdef:
			int N_molecules = self.N_molecules
			int N_particles = self.N_particles
			int i, j, k
			#
			int [:,:] molecule_ids = self.molecule_ids
			int [:] first_index_of_molecule = self.first_index_of_molecule
			int  N_dim = self.N_dim
			int angle_is_pie

		#
		self.bond_interaction_functions = []
		self.angle_interaction_functions = []
		self.bonded_neighbor_list = {i:set() for i in range(N_particles)}
		#
		for i in range(N_molecules):
			#
			j = first_index_of_molecule[i]
			# get molecule id of current molecule
			mol_id = molecule_ids[j,2]
			#
			for interaction in self.molecule_interactions[mol_id]:
				#
				if interaction['type'] == 'bond':
					#
					i1 = interaction['participants'][0] + j
					i2 = interaction['participants'][1] + j
					#
					def func(i1,i2,bond_strength,r0):
						return i1, i2, bond_strength * (self.d_xx[i1,i2] - r0)
					#
					func_partial = functools.partial(func,
						i1=i1,
						i2=i2,
						bond_strength=interaction['u0'],
						r0=interaction['r0'],
													)
					#
					self.bond_interaction_functions.append( func_partial )
					#
					self.bonded_neighbor_list[min(i1,i2)].add(max(i1,i2))
					#
				elif interaction['type'] == 'angle':
					#
					i1=interaction['participants'][0] + j
					i2=interaction['participants'][1] + j
					i3=interaction['participants'][2] + j
					#
					# angle interactions
					# the angle interactions we use are analogous to the 
					# GROMACS definition, see 
					# https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html#harmonic-angle-potential
					#
					# check if the current equilibrium angle is equal to pi
					if np.isclose(interaction['theta0'],np.pi,
										atol=1e-5):
						angle_is_pi = 1.
					else:
						angle_is_pi = -1.
					#
					# define function for calculating angle force prefactor
					def func(i1,i2,i3,
							k0, # angle-interaction spring constant
							theta0, # equilibrium angle
							theta0_minus_approximation_threshold,
							angle_is_pie, # +1 if theta0 = pi, -1 otherwise
							):
						#
						# calculate the angles. 
						# For this, calculate
						# v1 = x[i1] - x[i2] = xx[i2,i1]
						# v2 = x[i3] - x[i2] = xx[i2,i3]
						# norm(v1) = d_xx[i2,i1]
						# norm(v2) = d_xx[i2,i3]
						# h = v1 * v2 / (norm(v1)*norm(v2)) = hat(v1)*hat(v2)
						# theta = angle between x[i1]-x[i2] and x[i3]-x[i2] 
						#       = arccos(h)
						#
						# calculate h
						h = 0.
						for k in range(N_dim):
							#
							h += self.xx_hat[i2,i1,k] * self.xx_hat[i2,i3,k]
						#
						# calculate theta
						if h < -1.:
							if h < -1.0000001:
								raise RuntimeError(
									"Inner product of units vectors is "\
									+"smaller than -1. Value = {0}".format(h))
							else: # if -1.0000001 < inner product < -1,
								#   we assume its just a rounding error
								h = -1.
								theta = np.pi
						else:
							theta = acos(h)
						# angle theta is in rad
						#
						if angle_is_pi > 0.:
							# this here is to increase the numerical stability
							# for an angle 180 degrees, where h \approx 1.
							if theta < theta0_minus_approximation_threshold:
								prefac = k0*(theta-theta0)/(sqrt(1.-h**2))
							else:
								# use taylor expanded approximate
								prefac = -k0/(1.-(theta-theta0)**2/6. \
											+ (theta-theta0)**4/(120.) )
						else:
							prefac = k0 *(theta-theta0) /(sqrt(1.-h**2))
						#
						return i1, i2, i3, prefac, h
					#
					func_partial = functools.partial(func,
							i1=i1,
							i2=i2,
							i3=i3,
							k0=interaction['u0'],
							theta0=interaction['theta0'],
							theta0_minus_approximation_threshold=\
								interaction['theta0']\
								-self.angle_pi_approximation_threshold,
							angle_is_pie=angle_is_pie,
										)
					#
					self.angle_interaction_functions.append( func_partial )
					#
					self.bonded_neighbor_list[min(i1,i2)].add(max(i1,i2))
					self.bonded_neighbor_list[min(i2,i3)].add(max(i2,i3))
					#
	
	
	#####################################################
	# 2. neighbor lists & particle distance calculation #
	#####################################################

	cdef set_up_neighbor_list_cells(self):
		'''
		Set up some arrays that are relevant for updating neighbor lists
		using cell lists

		(see e.g. https://en.wikipedia.org/wiki/Cell_lists to get an idea
		of what cell lists are)
		''';
		#
		cdef:
			int k, l
			int rc_cl = self.neighbor_list_rc_to_cell_length_ratio
			double r = float(self.neighbor_list_rc_to_cell_length_ratio)

		# to do:
		# - create array with cell id
		# - determine cell lengths and number of cells along each axis
		self.neighbor_list_cell_length = np.zeros(self.N_dim,dtype=DTYPE)
		self.neighbor_list_number_of_cells_along_axis = np.zeros(self.N_dim,
												dtype=np.intc)
		self.neighbor_list_pbc_jump_cutoff_along_axis = np.zeros(self.N_dim,
												dtype=np.intc)
		for k in range(self.N_dim):
			#
			l = 1
			while ( self.L[k]/float(l+1) >= self.rs/r ):
				l += 1
			#
			self.neighbor_list_cell_length[k] = self.L[k]/float(l)
			self.neighbor_list_number_of_cells_along_axis[k] = l
			self.neighbor_list_pbc_jump_cutoff_along_axis[k] = l - rc_cl


	cdef create_neighbor_list(self):
		'''
		Create a neighbor list for each particle

		Particles that are within a molecule and which interact with each other
		(either via a bond or an angle interactions) are always on the neighbor
		list.
		Particles that are not within a molecule, but couple to each other
		(either via a nonbonded interaction, or via a reaction), are added to
		the neighbor list if their distance is smaller than the threshold 
		distance self.rs

		We proceed as follows to calculate the neighbor list:
		- we start from the list of bonded neighbors
		- we partition the box into (hyper)cubic cells, and for each particle
		  determine the cell it is located in
		- for each pair of particles (that do not belong to the same molecule),
		  we look how many cells they are apart in each dimension. If they 
	      are close enough, we add the pair to the tentative neighbor list
		- for all particle pairs that are on the tentative neighbor list, we 
		  evaluate the distances. We remove all pairs whose distance is larger
		  than the threshold distance
		'''
		cdef:
			int N_particles = self.N_particles
			int N_dim = self.N_dim
			int i, j, k, l, neighbor_count
			int particle_type_i, particle_type_j
			double are_neighbors
			int bnc
			#
			int [:,:] molecule_ids = self.molecule_ids
			double [:,:] d_xx = self.d_xx
			#
			double [:,:] x = self.x
			#
			double [:] periodic = self.periodic
			#
			int [:] particle_types = self.particle_types
			#
			#
			int rc_cl = self.neighbor_list_rc_to_cell_length_ratio
			int [:] cell_id = self.neighbor_list_cell_id
			#int [:] ref_cell_id = np.zeros(self.N_dim,dtype=np.intc)
			int [:,:] d_cell_id = self.d_cell_id
			double [:] cl = self.neighbor_list_cell_length
			int [:] nc = self.neighbor_list_number_of_cells_along_axis
			int [:] pbc_nc = self.neighbor_list_pbc_jump_cutoff_along_axis
			int d_cell
			
			#
			double rs = self.rs #

		###########################################
		# find out in which cell each particle is #
		###########################################
		# neighbor_list_cell[i*N_dim + k] = integer that says in which cell 
		# particle i is with respect to the dimension k
		for i in range(N_particles):
			for k in range(N_dim):
				cell_id[i*N_dim + k] = int( x[i,k] // cl[k] )

		#
		# particles that are bonded to each other are always on the neighbor
		# list
		self.neighbor_list = self.bonded_neighbor_list.copy()
		#
		##################################################
		# create a neighbor list such that all particles #
		# which are in close by cells are neighbors      #
		##################################################
		for i, particle_type_i in enumerate(particle_types):
			#
			for j in range(i+1,N_particles):
				#
				if molecule_ids[i,0] == molecule_ids[j,0]:
					continue
				#
				particle_type_j = self.particle_types[j]
				#
				# If the particles don't interact, we don't need their distance
				try:
					if particle_type_i not in \
						self.interaction_partners[particle_type_j]:
						continue
				except KeyError:
					continue
				#
				# by default, we set particles i, j as neighbors
				are_neighbors = 1.
				#
				# then we check if they are too far apart
				for k in range(N_dim):
					# get difference of cells where the particles are located
					# (with respect to dimension k)
					d_cell = cell_id[j*N_dim + k] - cell_id[i*N_dim + k]
					#
					if d_cell < 0: # take absolute value
						d_cell = -d_cell
					#
					# d_cell = absolute value of difference of cell indices 
					#          of particles i & j along axis k
					#
					if d_cell <= rc_cl: # if d_cell <= rc_cl, then the 
						# particles are close and we continue with the next
						# dimension
						continue
						#
					elif periodic[k] > 0: # if d_cell > nl_n, then it might 
					# still be that one particle is at the very left of the 
					# box, and the other at the very right..
						if d_cell < pbc_nc[k]:
							are_neighbors = -1.
							break
					else:
						are_neighbors = -1.
						break
				#
				if are_neighbors > 0:
					self.neighbor_list[min(i,j)].add( max(i,j) )
		#
		##################################################################
		# calculate the distance matrix using the current neighbor lists #
		##################################################################
		self.calculate_distances()
		#
		########################################
		# with the distances available, remove #
		# those pairs that are too far apart   #
		########################################
		for i, neighbors in self.neighbor_list.items():
			#
			remove = set([])
			#
			for j in neighbors:
				if d_xx[i,j] > rs:
					remove.add(j)
			#
			self.neighbor_list[i] = neighbors - remove
		#


	cdef calculate_distances(self):
		'''
		Calculate distances between all particle pairs on the neighbor list
		''';
		cdef:
			int N_dim = self.N_dim
			int N_particles = self.N_particles
			int k
			int i1, i2
			int l
			#
			double [:] L = self.L
			double [:,:] x = self.x
			#
			double [:,:,:] xx = self.xx
			double [:,:,:] xx_hat = self.xx_hat
			double [:,:] d_xx = self.d_xx
			#
			double d_xx_local, xx_local_k
			#
			int [:,:] molecule_ids = self.molecule_ids
			double [:] periodic = self.periodic
			#
			double min_dist = np.inf
			#
			
		#
		for i1, neighbors in self.neighbor_list.items():
			#
			for i2 in neighbors:
				#
				# note that, to obtain each pair (i1,i2) exactly once,
				# we sum over the indices as i1 < i2
				#
				d_xx_local = 0
				for k in range(N_dim): # iterate over all dimensions
					#
					xx_local_k = x[i2,k] - x[i1,k]
					#
					if periodic[k] > 0:
						xx_local_k = xx_local_k - L[k]*round(xx_local_k/L[k])
					#
					d_xx_local += xx_local_k**2
					#
					#xx_local[tid,k] = xx_local_k
					xx[i1,i2,k] = xx_local_k #xx_local[tid,k]
					xx[i2,i1,k] = -xx_local_k #xx_local[tid,k]
				#
				d_xx_local = sqrt(d_xx_local)
				#
				if d_xx_local < min_dist:
					if molecule_ids[i1,0] != molecule_ids[i2,0]:
						min_dist = d_xx_local
				#
				d_xx[i1,i2] = d_xx_local
				d_xx[i2,i1] = d_xx_local
				#
				for k in range(N_dim): # iterate over all dimensions
					#
					xx_hat[i1,i2,k] = xx[i1,i2,k]/d_xx_local
					xx_hat[i2,i1,k] = -xx_hat[i1,i2,k]
				#
		self.min_dist = min_dist

	######################################
	# 3. IO and storage for trajectories #
	######################################

	cdef create_arrays(self):
		'''
		Create the arrays used to store data during a simulation
		'''
		cdef:
			int N_particles = self.N_particles
			int N_dim = self.N_dim
			int N_reactions = self.N_reactions
			int N_traj = self.N_traj

		# drift
		self.a    = np.zeros( [N_particles,N_dim],
									dtype=DTYPE)
		#
		# connecting vectors between particles
		self.xx    = np.zeros( [N_particles,N_particles,N_dim],
									dtype=DTYPE)
		# normalized connecting vectors between particles
		self.xx_hat    = np.zeros( [N_particles,N_particles,N_dim],
									dtype=DTYPE)
		# distance between particles
		self.d_xx    = np.zeros( [N_particles,N_particles],
									dtype=DTYPE)
		#
		# trajectory locations of particles
		self.trajectory  = np.zeros( [N_traj,
									N_particles,N_dim], dtype=DTYPE)
		# trajectory particle types (nontrivial, since particles can react!)
		self.trajectory_types  = np.zeros( [N_traj,
									N_particles], dtype=np.intc)
		#
		# self.reaction_rates[i,j] = rate of reaction i for particle j
		self.reaction_rates = np.zeros([N_reactions,N_particles],
								dtype=DTYPE)
		# self.reaction_rates[j] = total rate for reaction of particle j
		self.total_reaction_rates = np.zeros(N_particles,
								dtype=DTYPE)
		#
		# for saving to which cell which particle belongs (for neighbor list)
		self.neighbor_list_cell_id = np.zeros(N_particles*N_dim,
												dtype=np.intc)
		self.d_cell_id = np.zeros([N_particles*N_dim,
										N_particles*N_dim],
												dtype=np.intc)

	cdef load_trajectory_from_file(self):
		'''
		Load a previously saved trajectory

		This is used for appending to an existing trajectory
		''';

		try:
			with h5py.File(self.trajectory_filename, 'r') as hf:
				self.molecule_ids = hf['molecule_ids'][()]
				self.first_index_of_molecule =  hf['first_index_of_molecule'][()]
				self.particle_types = hf['trajectory_types'][-1]
				self.x = hf['trajectory'][-1]
				#
				self.N_steps_completed = hf['N_steps_completed'][()]
				self.N_particles = hf['N_particles'][()]
				self.N_molecules = hf['N_molecules'][()]
				#
		except FileNotFoundError:
			self.N_steps_completed = 0
		#

	cdef save_trajectory_to_file(self):
		'''
		Save the current trajectory to a file

		We use the hdf5 file format for saving trajectories
		''';

		with h5py.File(self.trajectory_filename, 'w') as hf:
			hf.create_dataset("trajectory", 
						data=(self.x).reshape([1,self.N_particles,self.N_dim]), 
						maxshape=(None,self.N_particles,self.N_dim),
						chunks=True)
			hf.create_dataset("trajectory_types", 
						data=(self.particle_types).reshape([1,self.N_particles]), 
						maxshape=(None,self.N_particles),
						chunks=True)
			hf.create_dataset("molecule_ids", 
						data=self.molecule_ids,
						chunks=True)
			hf.create_dataset("first_index_of_molecule", 
						data=self.first_index_of_molecule,
						chunks=True)
			hf.create_dataset("N_steps_completed", 
						data=self.N_steps_completed,
						dtype=int)
			hf.create_dataset("N_particles", 
						data=self.N_particles,
						dtype=int)
			hf.create_dataset("N_molecules", 
						data=self.N_molecules,
						dtype=int)
			hf.create_dataset("dt_trajectory", 
						data=self.dt*self.stride,
						dtype=float)
			hf.create_dataset("N_species", 
						data=self.N_species,
						dtype=int)
			hf.create_dataset("stride", 
						data=self.stride,
						dtype=int)
			hf.create_dataset("L", 
						data=self.L)



	cdef append_trajectory_to_file(self):
		'''
		Append the current trajectory to an existing hdf5 file
		''';

		with h5py.File(self.trajectory_filename, 'a') as hf:
			hf["trajectory"].resize(
							(hf["trajectory"].shape[0] + self.N_traj), 
									axis = 0)
			hf["trajectory"][-self.N_traj:] = self.trajectory
			#
			hf["trajectory_types"].resize(
							(hf["trajectory_types"].shape[0] + self.N_traj), 
									axis = 0)
			hf["trajectory_types"][-self.N_traj:] = self.trajectory_types
			#
			hf['N_steps_completed'][...] = self.N_steps_completed



	cdef load_initial_conditions_from_file(self):
		'''
		Load an initial condition from a text file

		We assume that the text file has (N_dim + 4) columns. Each line in the
		text file contains information about a single particle, and uses the
		following order:

		column  content
		---------------
		0       molecule number
				This number enumerates the molecules present in the system. 
				For example a molecule number "5" means that this is the fifth
				molecule saved in the text file
		1       particle number within molecule
				This number enumerates the particles within the current 
				molecule.
				For, "2" means that this is the second particle belonging
				to the current molecule
		2       molecule id 
				This number defines what molecule the current molecule is.
				The properties of each molecule are defined in the text file
				for the bonded interactions (default: bonded_interactionts.txt)
		3       particle type
				This number defines the particle type of the current particle.
				The interaction- and reaction properties of each particle are
				defined in the text files for nonbonded interactions and 
				reactions (default: nonbonded_interactionts.txt and 
				reactions.txt)
		4+      location of the current particle
				In the columns 4 .. 4 + N_dim - 1, the location of the current
				particle is stored
		
		''';
		cdef:
			double [:] L = self.L
			set comment_chars = self.comment_chars
		#

		molecule_ids = [] # column 2
		particle_types = [] # column 3
		x = [] # column 4 onwards
		first_index_of_molecule = []
		#
		N_particles = 0
		N_molecules = 0
		N_species = 0
		#
		with open(self.initial_conditions_filename,'r') as f:
			for line in f:
				# remove comments
				for char in comment_chars:
					if char in line:
						line = line.split(char)[0]
				#
				# if current line is empty, continue
				if len(line) == 0:
					continue
				#
				# remove tabs and split line
				line = line.replace('\t',' ')
				line_split = line.split()
				#
				# for the first line, we check whether the number of columns
				# in the text file matches our expectations
				if N_particles == 0:
					N_dim = len(line_split) - 4
					if N_dim != self.N_dim:
						raise RuntimeError("Dimension of initial conditions "\
									+"({0}) inconsistent".format(N_dim)\
									+" with N_dim = {0}".format(self.N_dim))
				elif len(line_split) !=  N_dim + 4:
					# if the current line does not have the expected amount
					# of columns, we skip it
					continue
				#
				#
				molecule_ids.append([
						int(line_split[0]), # molecule number
						int(line_split[1]), # particle number within molecule
						int(line_split[2]), # molecule id
									])
				#
				p_type = int(line_split[3]) # particle type
				particle_types.append( p_type )
				#
				x.append([ float(line_split[j+4])  for j in range(N_dim)])
				#
				# if the current particle is the first of a new molecule, we
				# store its index separately
				if int(line_split[1]) == 0:
					first_index_of_molecule.append(N_particles)
					N_molecules += 1
				#
				# if the current particle type is larger than the largest one
				# observed so far, we increase the latter number
				if p_type > N_species:
					N_species = p_type
				#
				# after processing the line, the number of particles has 
				# increased by one
				N_particles += 1

		if N_particles == 0:
			raise RuntimeError("Initial condition file contains no particles")
		
		self.N_particles = N_particles
		self.N_molecules = N_molecules
		if self.N_species < N_species + 1:
			raise RuntimeError(
					"Initial conditions contains {0}".format(N_species+1) \
					+ " particle types, but only {0}".format(self.N_species)\
					+ " friction coefficients have been defined. Please set"\
					+ " particle_friction parameters for all particles.")
		#	
		# turn the data we have loaded into instance variables
		self.x = np.array(x,dtype=DTYPE)
		self.molecule_ids = np.array(molecule_ids,
										dtype=np.intc)
		self.particle_types = np.array(particle_types,
										dtype=np.intc)
		self.first_index_of_molecule = np.array(first_index_of_molecule,
													dtype=np.intc)
		#



	def write_current_state_to_text_file(self,
								filename='state.txt'):
		#
		cdef:
			int N_particles = self.N_particles
		#
		with open(filename,'w') as f:
			#
			for i in range(N_particles):
				#
				for j in range(self.N_dim+4):
					if j < 3:
						f.write('{0:d}   '.format(self.molecule_ids[i,j]))
					elif j == 3:
						f.write('{0:d}   '.format(self.particle_types[i]))
					else:
						f.write('{0:3.5f}   '.format(self.x[i,j-4]))
				f.write('\n')
		#

	###########################################################
	# 4. simulation (evaluate drift, execute reactions, etc): #
	###########################################################

	cdef update_drifts(self):
		'''
		Update the drift for all particles
		''';
		cdef:
			double [:,:] a = self.a
			double [:,:] x = self.x
			#
			double [:,:] d_xx = self.d_xx
			double [:,:,:] xx_hat = self.xx_hat
			#
			int [:] particle_types = self.particle_types
			int [:,:] molecule_ids = self.molecule_ids
			#
			int N_particles = self.N_particles
			int N_dim = self.N_dim
			#
			int p1,p2, p1_,p2_ # particle types
			int i, j, k, i1, i2, i3 # indices for iterations
			str int_type # interaction type
			double f, prefac, h
			#
			list bond_interaction_functions = self.bond_interaction_functions
			list angle_interaction_functions = self.angle_interaction_functions
		
		# reset drift array
		a[:,:] = 0

		###############################################
		# evaluate drift from non-bonded interactions #
		###############################################
		for j,interaction in enumerate(self.interactions):
			#
			int_type = interaction['type']
			p1_, p2_ = interaction['pair']
			#
			for i1 in range(N_particles):
				#
				if int_type == 'wall': # wall interaction
					# for wall interaction, we have
					# p1_ = particle type that interacts with the wall
					# p2_ = dimension perpendicular to the wall
					if particle_types[i1] == p1_: 
						#
						f = interaction['func'](x[i1,p2_])
						a[i1,p2_] += -f
						#
				# for all other interaction types, we need to consider pairs
				else:
					#
					for i2 in self.neighbor_list[i1]:
						#
						# here we only consider intermolecular interactions,
						# so if particles i1, i2 are in the same molecule,
						# we skip i2:
						if molecule_ids[i1][0] == molecule_ids[i2][0]:
							continue
						#
						if particle_types[i1] < particle_types[i2]:
							p1 = particle_types[i1]
							p2 = particle_types[i2]
						else:
							p1 = particle_types[i2]
							p2 = particle_types[i1]
						#
						#
						if ((p1 == p1_) and (p2 == p2_)):
							# this means the current pair of particles interacts
							# via the current interaction
							#
							r = d_xx[i1,i2] # current scalar distance
							#
							# every implemented force has a cutoff distance:
							if r > interaction['rc']:
								continue
							#
							# if we are below the cutoff distance, we evaluate
							# the force:
							f = interaction['func'](r)
							#if int_type == 'tanh':
							#	print(i1,i2,f)
							#elif int_type =='lj':
							#	print('LJ!')
							#
							for k in range(N_dim): # iterate over all dimensions
								# for small r, it holds that U_LJ < 0,
								# and the force should be repulsive
								# note that xx[i1,i2,:]
								# is the vector that points from i1 to i2
								a[i1,k] +=   f*xx_hat[i1,i2,k]
								a[i2,k] +=  -f*xx_hat[i1,i2,k]
		#
		#
		###########################################
		# evaluate drift from bonded interactions #
		###########################################
		for i, func in enumerate(bond_interaction_functions):
			#
			i1, i2, f = func()
			#
			for k in range(N_dim): # iterate over all dimensions
				# for small r, it holds that U_LJ < 0,
				# and the force should be repulsive
				# note that xx[i1,i2,:]
				# is the vector that points from i1 to i2
				a[i1,k] +=   f*xx_hat[i1,i2,k]
				a[i2,k] +=  -f*xx_hat[i1,i2,k]

		##########################################
		# evaluate drift from angle interactions #
		##########################################
		for i, func in enumerate(angle_interaction_functions):
			#
			i1, i2, i3, prefac, h = func()
			#
			for k in range(N_dim):
				#
				f = prefac / d_xx[i2,i1] * ( xx_hat[i2,i3,k] - h*xx_hat[i2,i1,k] )
				a[i1,k] += f
				a[i2,k] -= f
				#
				#
				f = prefac / d_xx[i2,i3] * ( xx_hat[i2,i1,k] - h*xx_hat[i2,i3,k] )
				a[i3,k] += f
				a[i2,k] -= f
				#
		#


	cdef reaction_step(self):
		'''
		Calculate reaction rates and execute reactions
		''';
		#
		cdef:
			int N_particles = self.N_particles
			#
			double [:,:] reaction_rates = self.reaction_rates
			double [:] total_reaction_rates = self.total_reaction_rates
			#
			int i, j, k, l
			double s, u
			#
			int [:] particle_types = self.particle_types
			#
			double [:,:] d_xx = self.d_xx
			#
			double dt = self.dt
			#
			int particle_type_j
			int reactant, product, catalyze, inhibtor, inhibitor2


		########################
		# reset reaction rates #
		########################
		reaction_rates[:,:] = 0.
		total_reaction_rates[:] = 0.
		#
		################################
		# calculate new reaction rates #
		################################
		for i, reaction in enumerate(self.reactions):
			#
			rate = reaction['rate']
			reactant = reaction['reactant']
			product = reaction['product']
			#
			if reaction['type'] == 'linear':
				# linear reaction
				#
				for j, particle_type_j in enumerate(particle_types):
					# check if particle is a reactant for current reaction
					if particle_type_j == reactant: # if so, add rate
						reaction_rates[i,j] = rate
						total_reaction_rates[j] += rate
				#
			elif reaction['type'] == 'locally_catalyzed':
				# locally catalyzed reaction
				#
				catalyst = reaction['catalyst']
				threshold_distance = reaction['r0']
				#
				for j, particle_type_j in enumerate(particle_types): 
					#
					# check if particle is reactant for current reaction
					if particle_type_j == reactant:
						#print('reactant identified')
						# if so, check if any neighbor is a catalyst
						for k in self.neighbor_list[j]:
							if particle_types[k] == catalyst:
								#print('catalyst identified')
								# check if they are so close to each other
								# that the catalyzation happens
								if d_xx[j,k] < threshold_distance:
									reaction_rates[i,j] += rate
									total_reaction_rates[j] += rate
									#print('deactivation rate')
					#
					# check if particle is catalyst for current reaction
					if particle_type_j == catalyst:
						# if so, check if any neighbor is a reactant
						for k in self.neighbor_list[j]:
							if particle_types[k] == reactant:
								# check if they are so close to each other
								# that the catalyzation happens
								if d_xx[j,k] < threshold_distance:
									reaction_rates[i,k] += rate
									total_reaction_rates[k] += rate
									#print('deactivation rate')
					#
			elif reaction['type'] == 'locally_inhibited':
				# locally inhibited reaction
				#
				inhibitor = reaction['inhibitor']
				threshold_distance = reaction['r0']
				rate_decrease = reaction['rate_decrease']
				try:
					inhibitor2 = reaction['inhibitor2']
				except KeyError:
					# if there is no second inhibitor specified, we set a 
					# value that for inhibitor2 that will never evaluate to
					# "true" in any of the checks further below
					inhibitor2 = -1
				#
				# set all rates equal to the non-inhibited value
				for j, particle_type in enumerate(particle_types):
					if particle_type == reactant:
						reaction_rates[i,j] = rate
					else:
						reaction_rates[i,j] = 0.
				#
				# decrease the rates of those particles that have inhibitors
				# close by
				for j, particle_type in enumerate(particle_types):
					#
					# option 1: particle j is reactant
					if particle_type == reactant:
						for k in self.neighbor_list[j]:
							# check if neighboring particle is inhibitor
							if (particle_types[k] == inhibitor) or\
								(particle_types[k] == inhibitor2):
								#
								if d_xx[j,k] < threshold_distance:
									reaction_rates[i,j] -= rate_decrease
							#
							#
					#
					# option 2: particle j is an inhibitor
					if (particle_type == inhibitor) or \
						(particle_type == inhibitor2):
						for k in self.neighbor_list[j]:
							if particle_types[k] == reactant:
								#
								if d_xx[j,k] < threshold_distance:
									reaction_rates[i,k] -= rate_decrease
				#
				for j in range(N_particles):
					# check if there are any negative rates (could happen if there
					# are many inhibitors around a reactant), and set those equal
					# to zero
					if reaction_rates[i,j] < 0:
						reaction_rates[i,j] = 0.
					#
					# add current reaction rate to the total for particle j
					total_reaction_rates[j] += reaction_rates[i,j]

		#####################
		# execute reactions #
		#####################
		for i, total_reaction_rate in enumerate(total_reaction_rates):
			# first we determine whether or not particle i undergoes a
			# reaction
			if ( (dt*total_reaction_rate) > self.uniform_dist() ):
				# if a reaction takes place, we determine which one
				u = self.uniform_dist()
				s = 0.
				for j in range(self.N_reactions):
					s += reaction_rates[j,i]/total_reaction_rate
					if s >= u:
						particle_types[i] = self.reactions[j]['product']
						break
				#

	cdef update_positions(self,include_random_force=1.):
		'''
		Update particle positions using deterministic drift and random noise
		''';
		cdef:
			int N_dim = self.N_dim
			int N_particles = self.N_particles
			#
			double [:] noise_prefacs = self.noise_prefacs
			double [:] force_prefacs = self.force_prefacs
			int [:] particle_types = self.particle_types
			#
			double [:,:] x = self.x
			double [:,:] a = self.a
			#
			int i, j, k
			#
		#
		for j in range(N_particles):
			for k in range(N_dim):
				# deterministic force
				x[j,k] += force_prefacs[particle_types[j]]*a[j,k]
				# random force
				x[j,k] += include_random_force \
							* noise_prefacs[particle_types[j]] \
							* self.random_standard_normal()


	cdef apply_boundary_conditions(self):
		'''
		Apply boundary conditions to current configuration

		- For periodic boundary conditions, particles that exit
		  on one side of the box re-enter at the opposite side.
		- For non-periodic boundary conditions, particles are 
		  reflected at the box boundary

		''';
		cdef:
			double [:,:] x = self.x
			double [:] L = self.L
			double [:] periodic = self.periodic
			#
			int N_particles = self.N_particles
			int N_dim = self.N_dim
			#
			int j, k

		for j in range(N_particles): # iterate over particles
			for k in range(N_dim):  # iterate over dimensions
				if periodic[k] < 0:
					# for non-periodic boundaries, we reflect at the boundary
					if x[j,k] > L[k]: # "right" boundary
						x[j,k] -= 2*(x[j,k] - L[k])
					if x[j,k] < 0.: # "left" boundary
						x[j,k] -= 2*x[j,k]
				else:
					# for periodic boundaries, particles re-enter on the 
					# opposite side
					x[j,k] = x[j,k] % L[k]
				#
				# check if everything worked
				if (x[j,k] < 0) or (x[j,k] > L[k]):
					raise RuntimeError(
						"x[{0},{1}] = {2} ".format(j,k,x[j,k]) \
						+ "out of bounds. Maybe the timestep is too"\
						+" large for the given the interaction parameters?"
										)



	cdef integration_step(self,include_random_force=1.,
						step=0):
		'''
		Run a single integration step

		an integration step consists of:
		1. updating positions
		2. applying boundary conditions
		3. updating neighbor lists (if desired) 
		4. updating particle distance matrix
		5. performing reactions (this already uses the new positions)
		6. updating the drift using the new positions
		''';

		self.update_positions(include_random_force=include_random_force)
		#
		self.apply_boundary_conditions()
		#
		if step % self.neighbor_list_update_frequency == 0:
			self.create_neighbor_list()
		else:
			self.calculate_distances()
		#
		self.reaction_step()
		#
		self.update_drifts()
		#






	cpdef simulate(self,
				verbose=True,
				random_force=True,
				append=False,
				):
		"""
		Run overdamped Langevin dynamics coupled with reactions

		Arguments
		---------
		verbose (bool, default: True):
			output progress of simulation to terminal
		random_force (bool, default: True):
			include random force in time integration. If this is set to False,
			and reactions are turned off, then the code is basically minimizing
			the potential energy
		append (bool, default: False):
			If False: overwrite existing trajectory file
			If True: append to existing trajectory file

		Returns
		-------
		dictionary with simulation results

		"""

		cdef:
			#
			int stride = self.stride
			int N_traj = self.N_traj
			double [:] L = self.L
			#
			int i, j, k, l, m
			int status_update_frequency
			time_t start_time, elapsed_time
			int h_elapsed, m_elapsed, s_elapsed
			int h_remaining, m_remaining, s_remaining
			float include_random_force

		# check if random force should be included in the simulations
		if random_force:
			include_random_force = 1.
		else:
			include_random_force = 0.
		#
		if verbose:
			print("Loading and processing initial conditions..",end='\r')
		
		if append:
			# if we append, we load the existing trajectory file
			self.load_trajectory_from_file()
		else:
			# otherwise we load the initial conditions
			self.load_initial_conditions_from_file()
			self.N_steps_completed = 0
		#
		self.create_arrays() # create arrays for trajectories etc
		#
		self.set_up_neighbor_list_cells() # initalize cells for neighbor lists
		#
		self.create_bond_and_angle_force_functions() # initalize bond and angle forces
		#
		if verbose:
			print("Loading and processing initial conditions..done",end='\n')

		# before the first integration step, we need to create a neighbor
		# list and evaluate the drift using the initial configuration
		if verbose:
			print("Creating neighbor list and evaluating drift..",end='\r')
		self.create_neighbor_list() # create initial neighbor list
		
		#
		self.update_drifts()
		if verbose:
			print("Creating neighbor list and evaluating drift..done",
							end='\n')

		start_time = time(NULL) # for tracking the time
		status_update_frequency = max(1,self.N_steps//1000)

		# 
		for l in range(self.N_steps_completed,self.N_steps):
			#
			if (l+1) % status_update_frequency == 0 and l > 0:
				if verbose:
					elapsed_time = time(NULL) - start_time
					m_elapsed, s_elapsed = divmod(elapsed_time, 60)
					h_elapsed, m_elapsed = divmod(m_elapsed, 60)
					remaining_time = (self.N_steps/l -1)*elapsed_time
					m_remaining, s_remaining = divmod(remaining_time, 60)
					h_remaining, m_remaining = divmod(m_remaining, 60)
					print("Running simulation.. Progress: {0}%, elapsed time: {1:d}:{2:02d}:{3:02d}, remaining time: {4:d}:{5:02d}:{6:02d}\t\t\t".format(int(l/self.N_steps*100.),
																																	int(np.round(h_elapsed)),int(np.round(m_elapsed)),int(np.round(s_elapsed)),
																																	int(np.round(h_remaining)),int(np.round(m_remaining)),int(np.round(s_remaining))),
						  end='\r')
			#
			# perform integration step
			self.integration_step(include_random_force=include_random_force,
								step=l+1)
			#
			# store current configuration at every stride-th step
			if (l+1) % stride == 0:
				#
				m = (l//stride) % N_traj
				for i in range(self.N_particles):
					self.trajectory_types[m,i] = self.particle_types[i]
					for k in range(self.N_dim):
						self.trajectory[m,i,k] = self.x[i,k]
				#
				# at the final step, we save the trajectory
				if m == (N_traj - 1):
					self.N_steps_completed += N_traj*stride
					if self.N_steps_completed == N_traj*stride:
						self.save_trajectory_to_file()
					else:
						self.append_trajectory_to_file()

		dict_out = {
		'x':self.x, # final positions
		'particle_types':self.particle_types, # final particle types
		'trajectory':np.array(self.trajectory), # trajectory of positions
		'trajectory_types':np.array(self.trajectory_types,
							dtype=int), # trajectory of particle types
		'molecule_ids':self.molecule_ids, 
		'first_index_of_molecule':self.first_index_of_molecule, 
			}

		return dict_out


	#####################################
	# 5. creation of initial conditions #
	#####################################

	cdef create_reference_molecule(self,
								interactions, # bonded interactions
								N_atoms,
								N_tries=10):
		'''
		For a given dictionary with bonded interactions, create a reference
		molecule

		We create N_tries reference molecules, each consisting of N_atoms
		particles which try to minimize the potential energy that corresponds
		to the given bonded interactions.

		Arguments
		---------
		interactions (dictionary):
			contains the bonded interactions
		N_atoms (int):
			number of particles to be placed (should be consistent with the
			interactions)
		N_tries (int):
			number of tries for finding the energetically most favorable
			configuration

		Returns
		-------
		a numpy array of size (N_atoms, N_dim) with the positions of the
		found energetically minimal configuration

		''';
		cdef:
			int N_dim = self.N_dim

		#
		def energy(x_flattened): 
			'''
			calculate energy of given configuration using the given 
			interactions
			''';
			#
			# create array with positions
			x = x_flattened.reshape([N_atoms-1,N_dim])
			x = np.concatenate((np.zeros([1,N_dim],dtype=float),
										x))
			# create array with difference vectors
			xij = x[:, np.newaxis] - x # xij[i,j] = x[i] - x[j]
			d_xij = np.linalg.norm(xij,axis=-1) # d_xij[i,j] = || x[i] - x[j] ||
			#
			V = 0. # energy
			for i,interaction in enumerate(interactions):
				#
				if interaction['type'] == 'bond':
					i1, i2 = interaction['participants']
					V += interaction['u0'] \
							* ( d_xij[i1,i2] - interaction['r0']) ** 2
				#
				elif interaction['type'] == 'angle':
					i1, i2, i3 = interaction['participants']
					h = np.sum(xij[i2,i1]*xij[i2,i3]) \
							/ (d_xij[i2,i1]*d_xij[i2,i3])
					theta = acos(np.round(h,7))
					V += interaction['u0'] \
							* ( theta - interaction['theta0'] )**2
			return V
		#
		lowest_energy = np.inf # lowest energy found so far
		for i in range(N_tries):
			# create random initial condition
			x = np.random.rand(N_atoms*N_dim).reshape([N_atoms,N_dim])
			x[0] = 0. # we always have the first particle at
			#           the origin as a reference point
			x0 = x[1:].flatten() # flatten the rest of the coordinates
			#
			if len(x0) > 0:
				#
				# find minimal configuration
				xopt, es = cma.fmin2(energy, x0, 0.5)
				#
				# reshape current solution
				x = xopt.reshape([N_atoms-1,N_dim])
				x = np.concatenate((np.zeros([1,N_dim],dtype=float),
												x))
				#
				# evaluate energy of current solution
				current_energy = energy(x[1:].flatten())
				#
				# see if the current energy is lower than the lowest
				# energy we have found so far
				if current_energy < lowest_energy:
					lowest_energy = current_energy
					x_out = x.copy()
			else:
				return x
		return x_out


	cdef place_configuration_randomly(self,configuration):
		'''
		Rotate and translate a given configuration randomly inside the box

		Arguments
		---------
		configuration (np.array):
			array of shape (N_particles, N_dim) with positions of particles

		Returns
		-------
		array of shape (N_particles, N_dim) with rotated and translated
		positions of the particles

		''';
		cdef:
			double [:] L = self.L
			int N_dim = self.N_dim
		#
		rotated_configuration = np.zeros_like(configuration)
		#
		rng = np.random.default_rng()
		#
		# rotate
		M = self.create_random_rotation()
		for j,l in enumerate(configuration):
			rotated_configuration[j] = M.dot(l)
		#
		# move into box
		for j in range(N_dim):
			cur_min = np.min(rotated_configuration[:,j])
			if cur_min < 0:
				rotated_configuration[:,j] -= cur_min
		#
		# translate by random amount
		for j in range(N_dim):
			cur_max = np.max(rotated_configuration[:,j])
			rotated_configuration[:,j] += rng.random()*(L[j]-cur_max)
		#
		return rotated_configuration

	cdef get_minimal_distance(self,particles_0,particles_1):
		#
		return np.min(distance.cdist(particles_0, particles_1, 'euclidean'))

	cdef create_random_rotation(self):
		cdef:
			int N_dim = self.N_dim
		#
		rng = np.random.default_rng()
		#
		N_rotations = rng.choice(np.arange(1,100))
		#
		N_generators = int(N_dim*(N_dim-1)/2)
		generators = np.zeros([N_generators,N_dim,N_dim],dtype=float)
		transformations = np.zeros([N_generators,N_dim,N_dim],dtype=float)
		exp_angle = rng.random()*10*np.pi/180.
		#
		count = 0
		for i in range(N_dim):
			for j in range(i+1,N_dim):
				#
				generators[count,i,j] = 1.
				generators[count,j,i] = -1.
				#
				transformations[count] = scipy.linalg.expm(generators[count]*exp_angle)
				count += 1
		#
		rotation_sequence = rng.choice(N_generators,size=N_rotations)
		output_rotation = np.eye(N_dim,dtype=float)
		for i,e in enumerate(rotation_sequence):
			output_rotation = output_rotation.dot(transformations[e])
		#
		return output_rotation


	def create_initial_conditions(self,
									N_particles,
									particle_types,
									reference_configurations=None,
									min_distance_threshold=None,
									min_distance_threshold_sim=None,
									):
		'''
		Create an initial condition for a given number of molecules

		Arguments
		---------
		N_particles (dictionary):
			dictionary that defines how many molecules of which type are
			created.
			Example:
				N_particles = {0:125, 1:50}
			creates 125 molecules with molecule_id = 0 and 50 molecules with
			molecule_id = 1. The molecule defined by each molecule_id is given
			in the text file with the bonded interactions
		particle_types (dictionary):
			dictionary that defines the particle type for each particle in 
			each molecule
			Example:
				particle_types = {0:[0,1,1,1],
									1:[0,2]}
			means that the molecule with molecule_id = 0should have 4 particles
			(this should be consistent with the file that contains the bonded
			interactions), and that the first particle will be of particle
			type 0, whereas the remaining particules will be of particle type
			1. Similarly, the molecule with molecule_id = 1 consists of two
			particles, one with particle type 0 and one with particule type 2.
		reference_configurations (optional np.array, default None):
			array that contains reference configurations for each molecule.
			If no reference_configurations are provided, the program attempts
			to create reference configurations based on the molecule topologies
			defined in the bonded interactions definition.

		Returns
		-------
		an array with the created initial condition


		''';
		cdef:
			double [:] L = self.L
			int N_dim = self.N_dim
		#
		N_tries = 1000
		N_configurations_per_try = 1000
		if min_distance_threshold is None:
			min_distance_threshold = self.rs/4.
		if min_distance_threshold_sim is None:
			min_distance_threshold_sim = self.rs/10.
		#
		# N_particles[i] = number of particles of molecule type i
		# L[i] = box dimension in the i-th direction
		#
		# generate reference configuration for each molecule type
		#
		N_atoms = {}
		for i,parameters in (self.molecule_interactions).items():
				print('self.molecule_parameters[{0}] = {1}'.format(i,
								self.molecule_parameters[i])
								)
				N_atoms[i] = self.molecule_parameters[i]['n_atoms']
		#
		if not reference_configurations:
			reference_configurations = {}
			for i,parameters in (self.molecule_interactions).items():
				reference_configurations[i] =  \
							self.create_reference_molecule(self.molecule_interactions[i],
										N_atoms=N_atoms[i])
		else:
			reference_configurations_ = reference_configurations.copy()
			reference_configurations = {}
			for i,filename in reference_configurations_.items():
				reference_configurations[i] = np.genfromtxt(filename)
				if len(reference_configurations[i]) != N_atoms[i]:
					error_msg = ("Reference configuration for molecule {0}"
						" has {1} atoms, but interaction parameters for that"
						" molecule demand {2} atoms")
					raise RuntimeError(error_msg.format(i,
						len(reference_configurations[i]),
						N_atoms[i]))
		#
		# create random order for insertion of molecules
		molecule_order = np.zeros(0,dtype=int)
		N_particles_total = 0
		for i,N in N_particles.items():
			#
			molecule_order = np.concatenate((molecule_order,np.ones(N)*i))
			#
			N_particles_total += N_atoms[i]*N
		rng = np.random.default_rng()
		rng.shuffle(molecule_order)
		cumsum_particles = np.zeros_like(molecule_order,dtype=int)
		for i,e in enumerate(molecule_order):
			cumsum_particles[i] = N_atoms[e]
		cumsum_particles = np.cumsum(cumsum_particles)
		#
		# mol number, running number in mol, mol id, particle id, friction, positions
		output_array = np.zeros([N_particles_total,N_dim+4],dtype=float)
		count = 0
		for i,e in enumerate(molecule_order):
			print("Adding molecule {0} of {1}".format(i+1,len(molecule_order)))
			#
			reference_configuration = reference_configurations[e]
			count_prev = count
			for k in range(N_tries):
				if i > 0:
					min_distance = -np.inf
					for l in range(N_configurations_per_try):
						cur_rotated_configuration = self.place_configuration_randomly(
									configuration=reference_configuration
													)
						#
						cur_rotated_configuration_center = \
								(cur_rotated_configuration[0]).reshape([1,N_dim])
						#
						cur_remaining_particles_centers = output_array[:count_prev,4:].copy()
						cur_mask = (output_array[:count_prev,1] == 0)
						cur_remaining_particles_centers = cur_remaining_particles_centers[cur_mask]
						#
						current_min_distance = self.get_minimal_distance(\
											cur_rotated_configuration_center,
											cur_remaining_particles_centers)
						if current_min_distance > min_distance:
							rotated_configuration = cur_rotated_configuration
							min_distance = current_min_distance
				else:
					rotated_configuration = self.place_configuration_randomly(
									configuration=reference_configuration
													)
				#
				for j,f in enumerate(rotated_configuration):
					output_array[count,0] = i
					output_array[count,1] = j
					output_array[count,2] = e
					output_array[count,3] = particle_types[e][j]
					#
					output_array[count,4:] = f.copy()
					count += 1
				#
				if i > 2:
					self.molecule_ids = np.array(output_array[:count,:3].copy(),
												dtype=np.intc)
					self.particle_types = np.array(output_array[:count,3].copy(),
												dtype=np.intc)
					self.x = output_array[:count,4:].copy()
					self.N_particles = count
					self.write_current_state_to_text_file(
									filename=self.initial_conditions_filename)
					#
					if min_distance < min_distance_threshold:
						print("min_distance =",min_distance,
										' (below threshold)')
						count = count_prev
						continue
					#
					try:
						result = self.simulate(verbose=False,
												random_force=True)
					except RuntimeError:
						print("RuntimeError at molecule {0}".format(i+1))
						count = count_prev
						continue
					except ZeroDivisionError:
						print("ZeroDivisionError error at molecule {0}".format(i+1))
						count = count_prev
						continue
					#
					if self.min_dist < min_distance_threshold_sim:
						print("try {0}. Min distance below threshold".format(k))
						count = count_prev
						continue
					#
					relaxed_conf = result['x']
					if np.sum(np.isnan(relaxed_conf)) > 0: 
						count = count_prev
						continue
					output_array[:count,4:] = relaxed_conf.copy()
				#
				break
			if k == N_tries-1:
				raise RuntimeError("Could not place molecule {0}".format(i+1))
			self.molecule_ids = np.array(output_array[:count,:3].copy(),
												dtype=np.intc)
			self.particle_types = np.array(output_array[:count,3].copy(),
										dtype=np.intc)
			self.x = output_array[:count,4:].copy()
			self.N_particles = count
			self.write_current_state_to_text_file(
									filename=self.initial_conditions_filename)
		return output_array
				
				