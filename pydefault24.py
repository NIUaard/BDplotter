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

# on MAC OS
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

''' 
set defaults for nicely-formatted plots
define handy functions
'''
# plotting stuff
params = {'axes.labelsize': 24,
          'text.fontsize': 24,
          'legend.fontsize': 24,
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'text.usetex':True}
          
rc('text',fontsize=24)
rc('legend',fontsize=24)
rc('xtick',labelsize=24)
rc('ytick',labelsize=24)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

default_params = dict(nbins = 10,
                      steps = None,
                      trim = True,
                      integer = False,
                      symmetric = False,
                      prune = None)
ticker.MaxNLocator.default_params['nbins']=3


def PrettyPlot():
   plt.ticklabel_format(axis='y',style='sci',scilimits=(1,4))
   plt.ticklabel_format(axis='x',style='sci',scilimits=(1,4))
   plt.gca().xaxis.set_major_locator( ticker.MaxNLocator(nbins = 5) )
   plt.gca().yaxis.set_major_locator( ticker.MaxNLocator(nbins = 5) )
