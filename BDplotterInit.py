import numpy as np
import pylab as pyl
import scipy as sci
import scipy.special as spe
import scipy.optimize as opt
import os 
from matplotlib import rc
import matplotlib.pyplot as plt
import math
import random 
from matplotlib.colors import LogNorm

from matplotlib import ticker

import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# beam density friendly color map


cdict = cm.get_cmap('spectral')._segmentdata

print cdict['red'][:] 

print cdict['red'][0]
print cdict['blue'][0]
print cdict['green'][0]
print cdict['red'][1]
print cdict['blue'][1]
print cdict['green'][1]

cdict['red'][0]   = (0, 0.0, 1)  
cdict['blue'][0]  = (0, 0.0, 1)  
cdict['green'][0] = (0, 0.0, 1)  

del cdict['red'][17:19]
del cdict['blue'][17:19]
del cdict['green'][17:19]

cdict['red'][-1]   = (1, 1.00, 1)  
cdict['blue'][-1]  = (1, 0.0, 0)  
cdict['green'][-1] = (1, 0.0, 0)  


#for i in range(17):
#  cdict['red'][i][0]   = cdict['red'][i][0]*1.0/0.8
#  cdict['red'][i][1]   = cdict['red'][i][0]*1.0/0.8 
#  cdict['blue'][i]  = (0, 0.0, 1)  
#  cdict['green'][i] = (0, 0.0, 1)  


#print cdict['red'][:]
#print cdict['blue'][:]
#print cdict['green'][:]

beamcmap = LinearSegmentedColormap('name', cdict)

