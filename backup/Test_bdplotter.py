#!/usr/bin/env python

# read and plot a set of line output from Astra (Xemit.*, Yemit.* and Zemit.*)

# 
import sys
import os
os.putenv ("PYTHONPATH","/opt/nicadd/contrib/piot/Local_Install/lib/python2.4/site-packages/")
sys.path.append('/pdata/prokop/bdplotterDir/')
sys.path.append('/pdata/prokop/BasicScripts/')
from pylab import *

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from bdplotter_v1 import *

##from ElegantReadingFunctions import *
###from ImpactZReadingFunctions import *



path='/pdata/prokop/GlueTrack/September2012_Upgrade/June2013_ImpactZ/081513_ScanVersion_BasicEEX_Acc_Shaped_1nC_CSR_PartTwo.11111'


dDist=np.dtype({'names':['x','px','y','py','z','pz'],'formats':[np.double,np.double,np.double,np.double,np.double,np.double]})
particleData=np.loadtxt(open(path), dtype=dDist)




figure("Testbed",figsize=[12,12])

subplot(331)
Advanced_DensityPlot_HistogramSettings(particleData['x']*1000,particleData['y']*1000,51,'r',-1.3,0.6)

subplot(332)
BD_CurrentProfile_Steps_Color_Labeled (particleData['z']*1000,(particleData['pz']/mean(particleData['pz'])-1.0)*100.0,1.0e-9,100,'b','0pC')

show()
