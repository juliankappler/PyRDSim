#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['animation.embed_limit'] = 10000
import h5py
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

input_file = 'trajectory.h5'
plot_stride = 200 # only plot every second frame

# Load trajectory
with h5py.File(input_file, 'r') as hf:
    trajectory = hf['trajectory'][()]
    trajectory_types = hf['trajectory_types'][()]
    L = hf['L'][()]
    N_species = hf['N_species'][()]
    stride = hf['stride'][()]
    dt = hf['dt_trajectory'][()]



def round_relative(x,decimals=1):
    # For rounding the simulation time, which 
    # is printed at the top of the plot
    #
    if x == 0:
        return x
    #
    order = int(np.ceil(np.log10(x)))
    #
    if order <= 0:
        return np.round(x,decimals=decimals-order)
    else:
        return np.round(x,decimals=decimals)




# instantiate figure
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
fig.canvas.draw()
ax.set_aspect(1)
ax.set_xlim(( 0, L[0]))
ax.set_ylim(( 0, L[1]))
fontsize=20
ax.set_xlabel(r'$x/l$',fontsize=fontsize)
ax.set_ylabel(r'$y/l$',fontsize=fontsize)
ax.set_title(r'$t/\tau = 0$',fontsize=fontsize)
fig.tight_layout()


# set markersizes
r0 = 1.3
# see 
# https://stackoverflow.com/questions/65174418/how-to-adjust-the-marker-size-of-a-scatter-plot-so-that-it-matches-a-given-radi
markersize = ax.transData.transform([r0,0])[0] \
                - ax.transData.transform([0,0])[0] # points
markersizes = [markersize for i in range(N_species)]
for i in range(1,N_species):
    markersizes[i] /= 2.

# set colors
colors= ['dodgerblue','crimson','limegreen']
if N_species > len(colors):
    raise RuntimeError("Please provide a color for each species (by "\
            "extending the list 'colors')")

# initialization function
def init():
    global lines
    global line
    lines = []
    for j in range(N_species):
        line, = ax.plot([], [], marker='o',ls='',
            color=colors[j],
           markersize=markersizes[j],
              )
        lines.append(line)
    return (line,)

# animation function
def animate(i):
    #
    ax.set_title(r'$t/\tau = {0}$'.format(round_relative(
                                            x=i*plot_stride*dt)
                                                ),
                fontsize=20)
    #print(i)
    for j in range(N_species):
        j = N_species - j - 1
        mask = (trajectory_types[i*plot_stride] == j)
        lines[j].set_data(trajectory[i*plot_stride,:,0][mask], 
                            trajectory[i*plot_stride,:,1][mask])
    return (line,)

# call the animator
anim = animation.FuncAnimation(fig, 
                            animate,
                            init_func=init,
                            frames=len(trajectory)//plot_stride, 
                            interval=50, 
                            blit=True)

# save the video
f = "video.mp4"
writervideo = animation.FFMpegWriter(fps=30) 
anim.save(f, writer=writervideo)