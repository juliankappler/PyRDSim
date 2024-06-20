# PyRDSim: Python module for particle-based reaction-diffusion simulations

## About

This module allows to simulate many-particle overdamped Langevin dynamics, including reactions that change the interaction properties of individual particles. 

Some features (which are illustrated in the examples below):
* **Reactions that depend on the local neighborhood of a particle.** Any reaction can either have a constant rate, or a rate that depends on the neighborhood of the particle (to model e.g. catalyzed deactivation, or inhibited deactivation, of a molecule). Particles that undergo reactions can change their chemical properties, i.e. the way they interact.
* **Molecules** that consist of several particles can be defined straightforwardly. We illustrate this in the examples below with a patchy colloid model.
* **Attractive walls** can be defined at box boundaries (to model e.g. binding to a surface)

## Examples

Each example comes with code to generate a random initial condition, equilibrate the initial condition, run the simulation, and generate a video of it. You can do all of this by running the respective script "run_all.sh".

* [patchy-colloids](https://github.com/juliankappler/PyRDSim/blob/main/examples/patchy-colloids/): Run overdamped Langevin simulations of patchy colloids. We include the starting files for molecules with 2, 3, 4, and 5 patches. Each patch can exist in two forms, activated or deactivated; while activated patches attract each other, deactivated patches do not interact with each other. For each number of patches (2,3,4,5), we include code to simulate three reaction variants: i) no reactions (patches always activated), ii) constant rates of activation/deactivation, and iii) constant rate of activation, catalyzed rate of deactivation. Here is a video of patchy colloids with 3 patches, and reaction scenario iii):

https://github.com/juliankappler/PyRDSim/assets/37583039/5cc4dbe4-9faf-4d5b-ac2e-a3d5d96eb316

* [lennard-jones](https://github.com/juliankappler/PyRDSim/blob/main/examples/lennard-jones/): Run overdamped Langevin simulations of Lennard-Jones particles (without any reactions). There are two subfolders, one for running a single particle species, and one for running two particle species. Here is an example video for the latter:

https://github.com/juliankappler/PyRDSim/assets/37583039/367e7e1e-2707-49b0-952a-75c5003e307f

* [lennard-jones-wall](https://github.com/juliankappler/PyRDSim/blob/main/examples/lennard-jones-wall/): Run overdamped Langevin simulations of Lennard-Jones particles, with periodic boundary conditions in the $x$-direction and reflecting boundary conditions in the $y$-direction. The particles can be in a deactivated and an activated state. The activation rate is constant. The deactivation rate is only nonzero if another activated particle is close by. Activated particles feel an attraction to each other, and to the lower boundary of the domain ($y=0$). Here is a video:

https://github.com/juliankappler/PyRDSim/assets/37583039/5645923c-aefa-4553-acc2-9be553b976fb


## Usage

To run a simulation, one needs to specify the following (we use the standard filenames here; other filenames can be set via the parameters.txt file):

* parameters.txt: Contains simulation parameters (e.g. timestep, total time of simulation, friction coefficients for particles, ..)
* bonded_interactions.txt: Defines intramolecular interactions, i.e. the topology of the molecules to be simulated
* nonbonded_interactions.txt: Defines intermolecular interactions (e.g. Lennard-Jones, Yukawa, tanh-potential)
* reactions.txt: Defines reactions (e.g. constant rate, catalytically activated, catalytically inhibited)


## Installation

To install PyRDSim, you can run the following commands.

```bash
>> git clone https://github.com/juliankappler/PyRDSim.git .
>> cd PyRDSim
>> pip install .
```

## Acknowledgements

We acknowledge funding from the European Union’s Horizon 2020 research and innovation pro- gramme under the Marie Skłodowska-Curie grant agreement No 101068745.
