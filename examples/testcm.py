#-------------------------
#-------------------------default configurations
import numpy as np 
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
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 


# plotting stuff
params = {'axes.labelsize': 18,
          'text.fontsize': 18,
          'legend.fontsize': 18,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'text.usetex':True}
          
rc('text',fontsize=18)
rc('legend',fontsize=18)
rc('xtick',labelsize=18)
rc('ytick',labelsize=18)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=False)

default_params = dict(nbins = 10,
                      steps = None,
                      trim = True,
                      integer = False,
                      symmetric = False,
                      prune = None)
ticker.MaxNLocator.default_params['nbins']=3

#-------------------------
#------------------------- start plotting here
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


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


print cdict['red'][:]
print cdict['blue'][:]
print cdict['green'][:]
my_cmap = LinearSegmentedColormap('name', cdict)


x=np.linspace(0,1.0,100)
x=np.vstack((x,x))
pattern=np.loadtxt('ex_frame.txt')
#pattern=pattern-10.0
plt.figure()
plt.imshow(x,aspect='auto', cmap=my_cmap)
#plt.imshow(pattern, aspect='auto',origin='lower', cmap=my_cmap)
plt.colorbar()
plt.show()
