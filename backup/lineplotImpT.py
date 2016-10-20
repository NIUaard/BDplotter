#!/usr/bin/env python

# read and plot a set of line output from Astra (Xemit.*, Yemit.* and Zemit.*)

# 
import sys
import os
os.putenv ("PYTHONPATH","/opt/nicadd/contrib/piot/Local_Install/lib/python2.4/site-packages/")
from pylab import *


# argument on the filename and rtun number
arg = sys.argv 
file = arg[1]

file24= file + '.24'
file25= file + '.25'
file26= file + '.26'
file18= file + '.18'

print file18
print file24
print file25
print file26

#da=io.array_import.read_array('tgdata.dat')
dt=np.dtype({'names':['t','z','avg','rms','bgavg','bgrms','CS','emit'],'formats':[np.double,np.double,np.double,np.double,np.double,np.double,np.double,np.double]})
dz=np.dtype({'names':['t','z','rms','bgavg','bgrms','emit','corr'],'formats':[np.double,np.double,np.double,np.double,np.double,np.double,np.double]})
dr=np.dtype({'names':['t','z','gamma','K','beta','Rmax','bgrms'],'formats':[np.double,np.double,np.double,np.double,np.double,np.double,np.double,np.double]})
X=np.loadtxt(open(file24), dtype=dt)
Y=np.loadtxt(open(file25), dtype=dt)
Z=np.loadtxt(open(file26), dtype=dz)
R=np.loadtxt(open(file18), dtype=dr)

# plotting stuff
params = {'backend': 'ps',
	  'axes.labelsize': 18,
	  'text.fontsize': 18,
	  'legend.fontsize': 18,
	  'xtick.labelsize': 18,
	  'ytick.labelsize': 18,
	  'text.usetex': True}
rcParams.update(params)
#rc('text', usetex=True, size=10)
#xlabel(r"$\rho_{8.0}", fontsize="large")
ax = subplot(111)
plot (X['z'],X['rms'],label='\sigma_{x}')
plot (Y['z'],Y['rms'],'g--',label='\sigma_{y}')
plot (Z['z'],Z['rms'],'r-',label='\sigma_{y}')
xlabel('distance (m)', fontsize=18)
#ylabel('\sigma_{x} \mbox{~(mm)}', fontsize=18)
ylabel('rms beam size (mm)', fontsize=18)
legend()
# figure 2
figure()
plot (X['z'],X['emit'],label=' \gamma \epsilon_{x}')
plot (Y['z'],Y['emit'],'g--',label='\gamma \epsilon_{y}')
xlabel('distance (m)', fontsize=18)
#ylabel('\sigma_{x} \mbox{~(mm)}', fontsize=18)
ylabel('transverse emittance (\mu{\mbox{m)}', fontsize=18)
legend()
# figure 3 energy and energy spread
figure()
subplot (2,1,1)
plot (R['z'],R['K'])
xlabel('distance (m)', fontsize=18)
ylabel('kinetic energy (MeV)}', fontsize=18)
subplot (2,1,2)
plot (Z['z'],Z['bgrms'],label='total')
plot (R['z'],R['bgrms'],label='correlated')
xlabel('distance (m)', fontsize=18)
ylabel('energy spread (keV)}', fontsize=18)
legend()


# do this last to show all pics
show()
