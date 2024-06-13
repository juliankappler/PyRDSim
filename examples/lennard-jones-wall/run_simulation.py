#!/usr/bin/env python

import pyrdsim
import time

start_time = time.time()

simulation = pyrdsim.system()

simulation.simulate(verbose=True)


runtime = (time.time() - start_time)
with open('runtime.txt','w') as f:
    f.write(str(runtime))

print('\nRuntime in seconds: {0:3.1f}'.format(runtime))