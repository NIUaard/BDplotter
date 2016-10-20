#!/usr/bin/env python

# read and plot a set of line output from Astra (Xemit.*, Yemit.* and Zemit.*)

# 
import sys
import os
from bdplotterV042014 import *

BD_Init()

# argument on the filename and rtun number
arg = sys.argv 
file = arg[1]

#fileX= file + '.Xemit.'+run
#fileY= file + '.Yemit.'+run
#fileZ= file + '.Zemit.'+run

print file
#print fileY
#print fileZ
Nbins=21 
print "Nbins=", Nbins
#da=io.array_import.read_array('tgdata.dat')
dt=np.dtype({'names':['x','y','z','px','py','pz','t','q','ref','tag'],'formats':[np.double,np.double,np.double,np.double,np.double,np.double,np.double,np.double,np.double,np.double]})
PS=np.loadtxt(open(file))
#, dtype=dt)
N=len(PS)
print N
mc2=0.511e6
# create coordinate adding the mean value
x =(PS[1:,0]+PS[0,0])*1e3
y =(PS[1:,1]+PS[0,1])*1e3
z =(PS[1:,2]+PS[0,2])*1e3
px=(PS[1:,3]+PS[0,3])/mc2
py=(PS[1:,4]+PS[0,4])/mc2
pz=(PS[1:,5]+PS[0,5])/mc2


# figure 1 is xy density and histogram

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
zmin = z.min()
zmax = z.max()
pxmin = px.min()
pxmax = px.max()
pymin = py.min()
pymax = py.max()
pzmin = pz.min()
pzmax = pz.max()

BD_DensityPlot_Pro(x,y, numHexes=21, numXHistBins=21, numYHistBins=51, 
                   show_density=1, show_xcurrent=0, show_xhist=0, show_yhist=0, 
		   xhist_scale=0.3,   yhist_scale=0.3, histogram_linecode='r-', 
		   bunch_charge=0, xoffset=None, yoffset=None, density_type='log', 
		   threshold_population=0, color_scale=pl.cm.Greys)


plt.figure()
BD_CurrentProfile_Steps_Color_Labeled (x,y,1e-9,51,'red','current')

plt.show()
