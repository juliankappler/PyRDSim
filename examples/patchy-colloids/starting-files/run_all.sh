#!/bin/bash

cd initial_conditions
python create_initial_conditions.py &> create_initial_conditions.log
cd -

python run_equilibration.py &> run_equilibration.log

python run_simulation.py &> run_simulation.log

python ../../create_video.py