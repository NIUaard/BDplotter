'''
sdsa
'''

# pretty plot function

import numpy as np
import scipy as sci
import pylab as pyl
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

# SYSTEM DEPENDENT :: MAC OS (path of dvipng)
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

''' 
########################################################################
set defaults for nicely-formatted plots
define handy functions
########################################################################
'''
# enable tex-formatting (requires dvipng)

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
rc('text', usetex=True)

default_params = dict(nbins = 10,
                      steps = None,
                      trim = True,
                      integer = False,
                      symmetric = False,
                      prune = None)
ticker.MaxNLocator.default_params['nbins']=3

# select a number of major tick/label per axis (defaults is set to 5 here)
def FormatLabelSci():
   plt.ticklabel_format(axis='y',style='sci',scilimits=(1,4))
   plt.ticklabel_format(axis='x',style='sci',scilimits=(1,4))
   plt.gca().xaxis.set_major_locator( ticker.MaxNLocator(nbins = 5) )
   plt.gca().yaxis.set_major_locator( ticker.MaxNLocator(nbins = 5) )


''' 
########################################################################
Functions to LOAD output file and return normalized 
########################################################################
'''
#---------------------------------------------------- ASTRA
# load *emit file for the rootname and run number
def LoadAstraEmit(rootname, run):
# conversion of long. emittance to um
   keVmm2um=1.0
   fileX= rootname + '.Xemit.'+run
   fileY= rootname + '.Yemit.'+run
   fileZ= rootname + '.Zemit.'+run
   fileC= rootname + '.Cemit.'+run
   print fileX
   print fileY
   print fileZ

# load x,y,z emit
   uemit=np.dtype({'names':['z','t','avg','rms','rmsprime','emit','corr'],
               'formats':[np.double,np.double,np.double,np.double,
	                  np.double,np.double,np.double]})
   cemit=np.dtype({'names':['emit100x','emit95x','emit90x','emit80x',
                            'emit100y','emit95y','emit90y','emit80y',
                            'emit100z','emit95z','emit90z','emit80z'],
                'formats':[np.double,np.double,np.double,np.double,
	                   np.double,np.double,np.double,np.double,
	                   np.double,np.double,np.double,np.double]})
			   
   X=np.loadtxt(open(fileX), dtype=uemit)
   Y=np.loadtxt(open(fileY), dtype=uemit)
   Z=np.loadtxt(open(fileZ), dtype=uemit)
# check if Cemit file ie there
   C=[]
   if (os.path.isfile(fileC)):
       C=np.loadtxt(open(fileC), dtype=cemit)

   return (X,Y,X,C)


def LoadAstraSigma(rootname, run):
# conversion of long. emittance to um
   keVmm2um=1.0
   fileS= rootname + '.Sigma.'+run
   print fileS

# load x,y,z emit
   sigma=np.dtype({'names':['z','gamma',
                            'x2','xpx', 'xy',  'xpy', 'xz',  'xpz',
			         'px2', 'pxy', 'pxpy','pxz', 'pxpz',
			                'y2',  'ypy', 'yz',  'ypz',
					       'py2', 'pyz', 'pypz',
					              'z2' ,  'zpz',
					                      'pz2'],
                 'formats':[np.double,np.double,
		            np.double,np.double,np.double,np.double,np.double,np.double,
		                      np.double,np.double,np.double,np.double,np.double,
		                                np.double,np.double,np.double,np.double,
		                                          np.double,np.double,np.double,
		                                                    np.double,np.double,
		                                                              np.double]})
   S=np.loadtxt(open(fileS), dtype=sigma)

   return (S)


# load *emit file for the rootname and run number
def LoadAstraPhaseSpace(filename):

   phase=np.dtype({'names':['x','y','z','px','py','pz','clock','charge','index','status'],
               'formats':[np.double, np.double, np.double, np.double, np.double, np.double,
	                  np.double, np.double, np.integer, np.integer]})

   Ptmp=np.loadtxt(open(filename), dtype=phase)
   N=len(Ptmp['pz'])
   Ptmp['pz'][1:N]=Ptmp['pz'][0]+Ptmp['pz'][1:N]
   Ptmp['z'][1:N]=Ptmp['z'][0]+Ptmp['z'][1:N]
#   print np.shape(Ptmp)
   keep_us=np.where(Ptmp['status']>0)
   print keep_us
   PhSp=Ptmp[:][keep_us]
   return (PhSp)

#---------------------------------------------------- ELEGANT
def LoadElegantPhaseSpace(filename):
   conversion_t_to_z=299492458.0
   Mycommand = 'sddsprintout -noLabel -noTitle -col=x -col=y -col=t -col=xp -col=yp -col=p '+ filename + ' tmpelegantphasespace'
   os.system (Mycommand)
   phase=np.dtype({'names':['x','y','z','px','py','pz'],
               'formats':[np.double, np.double, np.double, np.double, np.double, np.double]})

   			   
   Ptmp=np.loadtxt(open("tmpelegantphasespace"), dtype=phase)
   PhSp=Ptmp
   PhSp['z']=Ptmp['z']*conversion_t_to_z
   PhSp['px']=Ptmp['px']*Ptmp['pz']
   PhSp['py']=Ptmp['py']*Ptmp['pz']
   return (PhSp)

def LoadElegantTwiss(filename):
   conversion_t_to_z=299492458.0
   Mycommand = 'sddsprintout -noLabel -noTitle -col=s -col=betax -col=alphax -col=betay -col=alphay -col=etax -col=etay ' + filename + ' tmpeleganttwiss'
   os.system (Mycommand)
   twiss=np.dtype({'names':['s','betax','alphax','betay','alphay','etax','etay'],
               'formats':[np.double, np.double, np.double, np.double, np.double, np.double, np.double]})

   			   
   TwissParam=np.loadtxt(open("tmpeleganttwiss"), dtype=twiss)
   return (TwissParam)

def LoadElegantSig(filename):
   conversion_t_to_z=299492458.0
   Mycommand = 'sddsprintout -noLabel -noTitle -col=s -col=betax -col=alphax -col=betay -col=alphay -col=etax -col=etay ' + filename + ' tmpeleganttwiss'
   os.system (Mycommand)
   twiss=np.dtype({'names':['s','betax','alphax','betay','alphay','etax','etay'],
               'formats':[np.double, np.double, np.double, np.double, np.double, np.double, np.double]})

   			   
   TwissParam=np.loadtxt(open("tmpeleganttwiss"), dtype=twiss)
   return (TwissParam)

''' 
########################################################################
Functions to plot beam parameters X, Y, Z all follow standard ASTRA
 units.
########################################################################
'''

def PlotEmit1plt(X,Y,Z):

   fig, ax1=plt.subplots()
   ax1.plot (X['z'],X['emit'],'-', color='blue', linewidth=2.0, label=r'$\gamma \epsilon_{x}$')
   ax1.plot (Y['z'],Y['emit'],'--', color='red', linewidth=2.0, label=r'$\gamma \epsilon_{y}$')
   ax1.legend(loc='lower right')
   ax1.set_ylabel(r'transverse emittance ($\mu$m)', fontsize=22)
   ax1.set_xlabel(r'distance (m)', fontsize=22)
   FormatLabelSci()
   ax2 = ax1.twinx()
   ax2.plot (Z['z'],Z['emit'],color='green', linewidth=2.0, label=r'$\gamma \epsilon_{z}$')
   plt.xlabel('distance (m)', fontsize=22)
   ax2.set_ylabel(r'longitudinal emittance ($\mu$m)', fontsize=22, color="green")
   FormatLabelSci()
   for label in ax2.get_yticklabels():
       label.set_color("green")
   ax2.legend(loc='upper left')
   plt.tight_layout()

def PlotEigenEmits(S):
# S is the output of LoadAAstraSigma
   fig, ax1=plt.subplots()
   ax1.plot (X['z'],X['emit'],'-', color='blue', linewidth=2.0, label=r'$\gamma \epsilon_{x}$')
   ax1.plot (Y['z'],Y['emit'],'--', color='red', linewidth=2.0, label=r'$\gamma \epsilon_{y}$')
   ax1.legend(loc='lower right')
   ax1.set_ylabel(r'transverse emittance ($\mu$m)', fontsize=22)
   ax1.set_xlabel(r'distance (m)', fontsize=22)
   FormatLabelSci()
   ax2 = ax1.twinx()
   ax2.plot (Z['z'],Z['emit'],color='green', linewidth=2.0, label=r'$\gamma \epsilon_{z}$')
   plt.xlabel('distance (m)', fontsize=22)
   ax2.set_ylabel(r'longitudinal emittance ($\mu$m)', fontsize=22, color="green")
   FormatLabelSci()
   for label in ax2.get_yticklabels():
       label.set_color("green")
   ax2.legend(loc='upper left')
   plt.tight_layout()



def PlotSize1plt(X,Y,Z):

   fig, ax1=plt.subplots()
   ax1.plot (X['z'],X['rms'],color='blue', linewidth=2.5, label=r'$\sigma_{x}$')
   ax1.plot (Y['z'],Y['rms'],'--', color='red', linewidth=2.0, label=r'$\sigma_{y}$')
   ax1.legend()
   ax1.set_ylabel(r'transverse rms beam size (mm)', fontsize=22)
   ax1.set_xlabel(r'distance (m)', fontsize=22)
   FormatLabelSci()
   ax2 = ax1.twinx()
   ax2.plot (Z['z'],Z['rms'],color='green', linewidth=2.0, label=r'$\sigma_{z}$')
   plt.xlabel('distance (m)', fontsize=22)
   ax2.set_ylabel(r'longitudinal rms beam size (mm)', fontsize=22, color="green")
   for label in ax2.get_yticklabels():
          label.set_color("green")
   ax2.legend()
   FormatLabelSci()
   plt.tight_layout()



def PlotEnergy1plt(X,Y,Z):

   fig, ax1=plt.subplots()
   ax1.plot (Z['z'],Z['rmsprime'],color='blue', linewidth=2.0, label=r'$\sigma_{x}$')
   ax1.plot (Z['z'],Z['corr'],color='green', linewidth=2.0, label=r'$\sigma_{x}$')
   ax1.legend()
   ax1.set_ylabel(r'energy spread (keV)', fontsize=22)
   ax1.set_xlabel(r'distance (m)', fontsize=22)
   FormatLabelSci()
   ax2 = ax1.twinx()
   ax2.plot (Z['z'],Z['avg'],color='red', linewidth=2.0, label=r'$\sigma_{z}$')
   plt.xlabel('distance (m)', fontsize=22)
   ax2.set_ylabel(r'kinetic energy (MeV)', fontsize=22, color="red")
   for label in ax2.get_yticklabels():
       label.set_color("red")
   ax2.legend()
   FormatLabelSci()
   plt.tight_layout()


def DensityPlot(X,Y,Nbin, axis=None):
# axis is a 4-tuple (xmin, xmax, ymin, ymax)
#   plt.hexbin(x,y, cmap=plt.cm.hot, bins='log',gridsize=Nbin)
#   plt.axis([xminplot, xmaxplot, yminplot, ymaxplot])
   plt.hist2d(X,Y, bins=Nbin)
   if axis!=None:
      plt.axis([axis[0],axis[1],axis[2],axis[3]])
#   , norm=LogNorm())
#   plt.colorbar()
   FormatLabelSci()
   plt.tight_layout()


def CenterDistribution (X,Y):
   x=X-X.mean()
   y=Y-Y.mean()


# create an histogram with zeros at the end so that the histogram fall back
# to a zero population
def histogram0(x,Nbins):
   yhist,xhist = np.histogram(x,normed=0, bins=Nbins)
#   print xhist
#   print yhist
   yhist0=np.zeros(Nbins+2)
   xhist0=np.zeros(Nbins+2)
#   print xhist0
   xhist0[1:Nbins+1]=xhist[0:Nbins]
   yhist0[1:Nbins+1]=yhist[0:Nbins]
   dx=xhist[1]-xhist[0]
   xhist0[0]=xhist[0]-dx
   xhist0[Nbins+1]=xhist[Nbins-1]+dx
   yhist0[0]=0.0
   yhist0[Nbins+1]=0.0
   return(yhist0,xhist0)

def DensityplotwProjec2x2(X,Y,Nbins,axis=None):   
   plt.subplot(221)
   
   if axis!=None:
      MyAxis=axis
      print MyAxis
      DensityPlot(X,Y,Nbins, axis=MyAxis)
   else: 
      DensityPlot(X,Y,Nbins)
       
   FormatLabelSci()

   plt.subplot(222)
   yhist0, xhist0 = histogram0(Y,Nbins)
   plt.step(yhist0, xhist0, linewidth=2.5, color='red') # plot as step lines instead of bars
   if axis!=None:
      plt.ylim([MyAxis[2],MyAxis[3]])
   FormatLabelSci()

   plt.subplot(223)
   yhist0, xhist0 = histogram0(X,Nbins)
   plt.step(xhist0, yhist0, linewidth=2.5, color='red') # plot as step lines instead of bars
   if axis!=None:
      plt.xlim([MyAxis[0],MyAxis[1]])
   FormatLabelSci()
   plt.tight_layout()
