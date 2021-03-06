#from pylab import *
# Last modified PP, 03-23-2011

import sys
import os
#os.putenv ("PYTHONPATH","/opt/nicadd/contrib/piot/Local_Install/lib/python2.4/site-packages/")
from pylab import *


sys.path.append('/pdata/prokop/bdplotterDir/')
sys.path.append('/pdata/prokop/BasicScripts/')



import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import ImpactZReadingFunctions as IZ
#import scipy
from bdplotter_pro import *



#da=io.array_import.read_array('tgdata.dat')
dt=np.dtype({'names':['z','avg','rms','bgavg','bgrms','CS','emit'],'formats':[np.double,np.double,np.double,np.double,np.double,np.double,np.double,np.double]})
#dz=np.dtype({'names':['z','avg','rms','bgavg','bgrms','emit','corr'],'formats':[np.double,np.double,np.double,np.double,np.double,np.double,np.double]})
dr=np.dtype({'names':['z','phase','gamma','K','beta','Rmax'],'formats':[np.double,np.double,np.double,np.double,np.double,np.double]})

dS=np.dtype({'names':['s','Sx','Sy','Ss', 'enx', 'eny', 'enz' ],'formats':[np.double,np.double,np.double,np.double,np.double,np.double,np.double]})



dDist=np.dtype({'names':['x','px','y','py','z','pz'],'formats':[np.double,np.double,np.double,np.double,np.double,np.double]})





distPath1='/pdata/prokop/GlueTrack/September2012_Upgrade/ImpactZ/ImpactZ_Dogleg_EEX_0p5_0p8.11111'

SampleData=IZ.ImpactZ_ReadDistribution35(distPath1)

distPath2='/pdata/prokop/GlueTrack/August2011/ImpactZ/101711_BC1_420.11111'

CurvedData=IZ.ImpactZ_ReadDistribution35(distPath2)

#file1 = '/pdata/prokop/GlueTrack/September2012_Upgrade/ImpactZ/ImpactZ_Dogleg_EEX_0p5_0p8'






BD_Init()







##BD_DensityPlot_ElaborateOptions(X,Y, numHexes=151, numHistBins=151,c show_density=1, show_xhist=0, show_yhist=0, histogram_linecode='r-', xcurrent=0, xoffset=0, yoffset=0, density_type='log',color_scale=pl.cm.Greys)

#figure()
figure(figsize=[15,20])
subplot(321)
BD_DensityPlot_Pro(CurvedData['x']*1000,CurvedData['px']*1000, numHexes=101, numXHistBins=75, numYHistBins=75, show_density=1, show_xcurrent=0, show_xhist=0, show_yhist=0, histogram_linecode='r-', bunch_charge=3.2e-9, density_type= 'log', color_scale=pl.cm.jet, threshold_population=0)


ylabel(r'p$_y$ ($\beta \gamma$)')
xlabel('x (mm)')
#plt.tight_layout()

#figure()
subplot(322)
BD_DensityPlot_Pro(CurvedData['x']*1000,CurvedData['px']*1000, numHexes=101, numXHistBins=151, numYHistBins=151, show_density=1, show_xcurrent=0, show_xhist=1, show_yhist=1, histogram_linecode='r-', bunch_charge=3.2e-9,  density_type=30, color_scale=pl.cm.Greys, threshold_population=0)
ylabel(r'p$_y$ ($\beta \gamma$)')
xlabel('x (mm)')
#plt.tight_layout()


#figure()
subplot(323)
BD_DensityPlot_Pro(CurvedData['x']*1000,CurvedData['y']*1000, numHexes=151, numXHistBins=151, numYHistBins=151, show_density=1, show_xcurrent=0, show_xhist=0, show_yhist=0, histogram_linecode='r-', bunch_charge=3.2e-9, density_type= 'log', color_scale=pl.cm.jet, threshold_population=0)


ylabel('y (mm)')
xlabel('x (mm)')
#plt.tight_layout()

#figure()
subplot(324)
BD_DensityPlot_Pro(CurvedData['x']*1000,CurvedData['y']*1000, numHexes=151, numXHistBins=151, numYHistBins=151, show_density=1, show_xcurrent=0, show_xhist=1, show_yhist=1, histogram_linecode='r-', bunch_charge=3.2e-9,  density_type=10,color_scale=pl.cm.Greys, threshold_population=0)
ylabel('y (mm)')
xlabel('x (mm)')
#plt.tight_layout()

#figure()
subplot(325)
BD_DensityPlot_Pro(CurvedData['z']*1.0e6*(0.23061)/360,(CurvedData['pz']/mean(CurvedData['pz'])-1)*100, numHexes=151, numXHistBins=151, numYHistBins=151, show_density=1, show_xcurrent=0, show_xhist=0, show_yhist=0, histogram_linecode='r--', bunch_charge=3.2e-9, density_type= 'log', color_scale=pl.cm.jet, threshold_population=0)


xlabel(r'z ($\mu$m)')
ylabel(r'$\delta$ (%)')
#plt.tight_layout()

#figure()
subplot(326)
BD_DensityPlot_Pro(CurvedData['z']*1.0e6*(0.23061)/360,(CurvedData['pz']/mean(CurvedData['pz'])-1)*100, numHexes=151, numXHistBins=151, numYHistBins=151, show_density=1, show_xcurrent=0, show_xhist=1, show_yhist=1, histogram_linecode='r--', bunch_charge=3.2e-9,  density_type=10,color_scale=pl.cm.Greys, threshold_population=0)
xlabel(r'z ($\mu$m)')
ylabel(r'$\delta$ (%)')
plt.tight_layout()





figure(figsize=[10,10])



subplot(221)
BD_DensityPlot_Pro(CurvedData['x']*1000,CurvedData['px'], numHexes=101, numXHistBins=151, numYHistBins=151, show_density=1, show_xcurrent=0, show_xhist=1, show_yhist=1, histogram_linecode='r-', bunch_charge=3.2e-9,  density_type=10, color_scale=pl.cm.Greys, threshold_population=0)
ylabel(r'p$_x$ ($\beta \gamma$)')
xlabel('x (mm)')
xticks([-1.0,0.0,1.0])
plt.tight_layout()

subplot(222)
BD_DensityPlot_Pro(CurvedData['y']*1000,CurvedData['py'], numHexes=151, numXHistBins=151, numYHistBins=151, show_density=1, show_xcurrent=0, show_xhist=1, show_yhist=1, histogram_linecode='r-', bunch_charge=3.2e-9,  density_type=10,color_scale=pl.cm.Greys, threshold_population=0)
ylabel(r'p$_y$ ($\beta \gamma$)')
xlabel('y (mm)')
plt.tight_layout()


subplot(223)
BD_DensityPlot_Pro(CurvedData['z']*1.0e6*(0.23061)/360,(CurvedData['pz']/mean(CurvedData['pz'])-1)*100, numHexes=151, numXHistBins=151, numYHistBins=151, show_density=1, show_xcurrent=0, show_xhist=1, show_yhist=1, histogram_linecode='r--', bunch_charge=3.2e-9,  density_type=10,color_scale=pl.cm.Greys, threshold_population=0)
xlabel(r'z ($\mu$m)')
ylabel(r'$\delta$ (%)')
plt.tight_layout()





#figure()
#BD_DensityPlot_ElaborateOptions(SampleData['x'],SampleData['y'], numHexes=51, numXHistBins=51, numYHistBins=51, show_density=0, show_xcurrent=0, show_xhist=1, show_yhist=0, histogram_linecode='r--', bunch_charge=3.2e-9, xoffset=0, yoffset=0, density_type='log',color_scale=pl.cm.Greys)

#BD_DensityPlot_ElaborateOptions(SampleData['x'],SampleData['y'], show_density=0, show_xhist=1, show_yhist=0, histogram_linecode='r--', bunch_charge=3.2e-9, density_type='log',color_scale=pl.cm.Greys)

#BD_DensityPlot_ElaborateOptions(SampleData['x'],SampleData['y'], show_density=0, show_xhist=0, show_yhist=1, histogram_linecode='r--', bunch_charge=3.2e-9, density_type='log',color_scale=pl.cm.Greys)
###plt.tight_layout()

#figure()
#BD_DensityPlot_ElaborateOptions(SampleData['x'],SampleData['y'], numHexes=51, numXHistBins=51, numYHistBins=51, show_density=0, show_xcurrent=0, show_xhist=0, show_yhist=1, histogram_linecode='r--', bunch_charge=3.2e-9, xoffset=0, yoffset=0, density_type='log',color_scale=pl.cm.Greys)


show()



