#!/usr/bin/env python

# read and plot a set of line output from Astra (Xemit.*, Yemit.* and Zemit.*)

# 
import sys
import os
from pylab import *
from bdplotter_pro import *


BD_Init()

# argument on the filename and rtun number
arg = sys.argv 
file = arg[1]
run  = arg[2]

fileX= file + '.Xemit.'+run
fileY= file + '.Yemit.'+run
fileZ= file + '.Zemit.'+run

print fileX
print fileY
print fileZ

#da=io.array_import.read_array('tgdata.dat')
dt=np.dtype({'names':['z','t','avg','rms','rmsprime','emit','corr'],'formats':[np.double,np.double,np.double,np.double,np.double,np.double,np.double]})
X=np.loadtxt(open(fileX), dtype=dt)
Y=np.loadtxt(open(fileY), dtype=dt)
Z=np.loadtxt(open(fileZ), dtype=dt)


# figure 1
#ax = subplot(111)
figure()
plt.plot (X['z'],X['emit'])
#,label=r'$\gamma \epsilon_{x}$')
plt.plot (Y['z'],Y['emit'])
#,'g--',label='$\gamma \epsilon_{y}$')
plt.plot (Y['z'],0*Y['rms'],'w--')
xlabel('distance (m)', fontsize=18)
ylabel(r'transverse emittance ($\mu$m)', fontsize=18)
#legend()

# figure 2
figure()
plt.plot (X['z'],X['rms'],label=r'$\sigma_{x}$')
plt.plot (Y['z'],Y['rms'],'g--',label=r'$\sigma_{y}$')
plt.plot (Z['z'],Z['rms'],'r-',label=r'$\sigma_{y}$')
xlabel(r'distance (m)', fontsize=18)
ylabel(r'rms beam sizes (mm)', fontsize=18)
legend()

# figure 3 energy and energy spread
figure()
subplot (2,1,1)
plt.plot (Z['z'],Z['avg'],label=r'$\epsilon_{x}$')
#-----------------
ylabel(r'$E_k$ (MeV)}', fontsize=18)
subplot (2,1,2)
plt.plot (Z['z'],Z['rmsprime'],label='total')
plt.plot (Z['z'],Z['corr'],label='correlated')
xlabel(r'distance (m)', fontsize=18)
ylabel(r'$\Delta E$ (keV)}', fontsize=18)
legend()


# do this last to show all pics
show()
