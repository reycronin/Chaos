"""
===========
MovieWriter
===========

This example uses a MovieWriter directly to grab individual frames and write
them to a file. This avoids any event loop integration, but has the advantage
of working with even the Agg backend. This is not recommended for use in an
interactive setting.

"""
# -*- noplot -*-
from matplotlib.ticker import FuncFormatter, MultipleLocator

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)



dx = .05
y = np.arange(0,2+dx,dx)
x = np.arange(0,1+dx,dx)
OMEGA, K = np.meshgrid(x,y)
W = K*1.0

minW = np.min(W)
maxW = np.max(W)

fig, ax = plt.subplots(figsize=(5,8))
cc = ax.pcolormesh(OMEGA,K,W, cmap ='gnuplot')

ax.yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
ax.yaxis.set_major_locator(MultipleLocator(base=4*np.pi))
ax.xaxis.set_major_locator(MultipleLocator(base=1))

cbar = fig.colorbar(cc, ax=ax, shrink = .5)
plt.xlabel('$\Omega$')
plt.ylabel('K')
cbar.set_label('number of iterations where |$\Theta - \Theta_o$| $< \epsilon$')
cbar.set_ticks([minW, int((maxW-minW)/2) ,int(maxW)])
cbar.set_ticklabels([minW, int((maxW-minW)/2), int(maxW)])
plt.savefig('Arnold.png')

with writer.saving(fig, "writer_test.mp4", 100):
    for i in range(100):
        W += .5 * np.random.randn()
        cc.set_data(W)
        writer.grab_frame()
