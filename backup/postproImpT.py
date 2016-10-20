#!/usr/bin/env python

# read and plot a set of line output from Astra (Xemit.*, Yemit.* and Zemit.*)
#os.putenv ("PYTHONPATH","/home/piot/Local_Install/lib/python2.4/site-packages/")

# 
import sys
import os
from pylab import *
from bdplotter import *


# argument on the filename and rtun number
arg = sys.argv 
file = arg[1]

#fileX= file + '.Xemit.'+run
#fileY= file + '.Yemit.'+run
#fileZ= file + '.Zemit.'+run

print file
#print fileY
#print fileZ

#da=io.array_import.read_array('tgdata.dat')
dt=np.dtype({'names':['x','px','y','py','z','pz'],'formats':[np.double,np.double,np.double,np.double,np.double,np.double]})
PS=np.loadtxt(open(file))
#, dtype=dt)
N=len(PS)
print N
mc2=0.511e6
# create coordinate adding the mean value
x =(PS[:,0])*1e3
y =(PS[:,2])*1e3
z =(PS[:,4])*1e3
px=(PS[:,1])
py=(PS[:,3])
pz=(PS[:,5])

# plotting stuff
BD_Init()
# figure 1 is xy density and histogram
BD_DensityPlot_w_projec (x,y)
#BD_DensityPlot_w_projec_sub (x,y)
#subplot (2,2,1)
xlabel('x (mm)', fontsize=18)
ylabel('y (mm)', fontsize=18)
#subplot (2,2,2)
#xlabel('y (mm)', fontsize=18)
#ylabel('population', fontsize=18)
#subplot (2,2,3)
#xlabel('x (mm)', fontsize=18)
#ylabel('population', fontsize=18)
#show()
# do this last to show all pics
figure()
BD_DensityPlot_w_Hprojec (z,pz)
#BD_DensityPlot_w_projec (z,pz)
#BD_DensityPlot_w_projec_sub (z,pz)
xlabel('z (mm)', fontsize=18)
ylabel('P_z ()', fontsize=18)
show()
