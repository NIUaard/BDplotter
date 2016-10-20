#from pylab import *
# Last modified PP, 03-23-2011

import sys
import os
os.putenv ("PYTHONPATH","/opt/nicadd/contrib/piot/Local_Install/lib/python2.4/site-packages/")
from pylab import *

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy


def BD_Init():
   params = {'backend': 'ps',
	  'axes.labelsize': 18,
	  'text.fontsize': 18,
	  'legend.fontsize': 18,
	  'xtick.labelsize': 18,
	  'ytick.labelsize': 18,
	  'subplot.wspace': 0.8,
	  'subplot.hspace': 0.8,
	  'text.usetex': False}
   plt.rcParams.update(params)



def BD_Xhistogram (X,Y,BinLength):
# bunchCharge is in Coulombs.  
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   NumBins=np.ceil(dx/BinLength)
   hx,edgesx = np.histogram(x, normed=0, bins=NumBins)
   Hx=hx/hx.max()*dy/3.0+(min_y+0.005*abs(min_y))
   hy,edgesy = np.histogram(y, normed=0, bins=NumBins)
   Hy=hy/hy.max()*dx/3.0+(min_x+0.005*abs(min_x))
#   plt.figure(PlotNumber)
   print Hx.max() , Hy.max(), dx, dy
   pl.step(edgesx[1:],hx) # plot as step lines instead of bars
   plt.legend()
