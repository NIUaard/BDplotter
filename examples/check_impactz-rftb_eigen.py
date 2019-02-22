import numpy as np
import sys
import math
import os
import matplotlib.pyplot as plt
import pydefault
from BDplotterInit import *
import BDplotter as BD
import matplotlib.pyplot as plt
from cosmetics import * 

dir_impact = '/bdata/piot/FAST-IOTA/magnetized_flat/Impact-Z/'
filenamei  =  dir_impact+'fort'


Si,eexi, eeyi, =BD.LoadImpactSigma(filenamei)
Xi,Yi,Zi =BD.LoadImpactZSig(filenamei, freq=1.3e9)

#f, (ax1, ax2) = plt.subplots(2)
#, sharex=True)
f, ax1=plt.subplots()
ax1.plot (Xi['z'],Xi['rms'],'--', color='C0', linewidth=1.0)
ax1.plot (Yi['z'],Yi['rms'],'--', color='C2',  linewidth=1.0, label=r'$\gamma \epsilon_{y}$')
ax1.plot (Zi['z'],Zi['rms'],'--', color='C1',  linewidth=1.0)

# this plot the concentional and eigenemittance from ImpactZ
f, ax2=plt.subplots()
ax2.semilogy(Xi['z'],Xi['emit'],'--', color='C0', linewidth=1.0,label=r'$\varepsilon_x$')
ax2.semilogy(Yi['z'],Yi['emit'],'--', color='C1', linewidth=1.0,label=r'$\varepsilon_y$')
ax2.semilogy(Si['z'],1e6*eexi,'--',   color='C2', linewidth=1.0,label=r'$\varepsilon_+$')
ax2.semilogy(Si['z'],1e6*eeyi,'--',   color='C3', linewidth=1.0,label=r'$\varepsilon_-$')
ax2.semilogy(Si['z'],1e6*np.sqrt(eexi*eeyi),'--',   color='C4', linewidth=1.0,label=r'$\varepsilon_{4d}$')
ax2.legend(loc='upper right', borderpad=0.2, labelspacing=0.1)
#ax2.plot(Si['z'],np.sqrt(Si['x2']),'--',   color='C3', linewidth=1.0)
#ax2.plot(Si['z'],np.sqrt(Si['y2']),'--',   color='C4', linewidth=1.0)
#ax2.plot(Si['z'],1e6*np.sqrt((Si['x2']*Si['px2']-Si['xpx']**2)),'--',   color='C3', linewidth=1.0)
#ax2.plot(Si['z'],(Si['ypy']),'--',   color='C4', linewidth=1.0)
plt.xlabel(r'distance along RFTB (m)')
plt.ylabel(r'normalized emittance ($\mu$m)')

plt.show()
