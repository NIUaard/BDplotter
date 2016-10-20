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
Functions to computed derived quantities
########################################################################
'''

def BunchFormFactor (z,kmin,kmax,nk,IsLog):
# compute the bunch form factor of the distribution in z coordinate
# return the k and the associated BFF between kmin and kmax with nk points 
# if IsLog set to true the spacing between kmin and kmax follows a log progression
   k=np.linspace(kmin,kmax,nk)
   if IsLog=='true':
      klmin=np.log(kmin)
      klmax=np.log(kmax)
      kl=np.linspace(klmin,klmax,nk)
      k=np.exp(kl)
      
   bff=k*0.
   for j in range(len(k)):
      sterm=sum(np.sin(2*np.pi*k[j]*z[:]));
      cterm=sum(np.cos(2*np.pi*k[j]*z[:]));
      bff[j]=cterm**2+sterm**2
   bff=bff/(1.0*len(z))**2
   return (k,bff)
   

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
   x=X-X.np.mean()
   y=Y-Y.np.mean()


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
   
   
# slice analysius from Chris Prokop (GlueTrack)
   
def UniformSliceAnalysis (xpxd,numbins,bunch_charge):
### bunchCharge is in Coulombs.  
        xpx=np.zeros((len(xpxd),6))
	print len(xpx)
	
        xpx[:,0]=xpxd['x']
        xpx[:,2]=xpxd['y']
        xpx[:,4]=xpxd['z']
        xpx[:,1]=xpxd['px']
        xpx[:,3]=xpxd['py']
        xpx[:,5]=xpxd['pz']


        xpx[:,4]=xpx[:,4]-np.mean(xpx[:,4])
	max_z=xpx[:,4].max()
	min_z=xpx[:,4].min()
	dz=max_z-min_z
	numparts=len(xpx)
	binwidth=dz/(1.0*(numbins-1.))
	index=0.0
	print max_z, min_z, numparts, numbins, binwidth

  #section to initialize the matrix that contains all of the values.  Data set, or really just a matrix?
	sliceMatrix=np.zeros((numbins,14))

	while index<numbins:
		slice_floor=min_z + (index-0.5)*binwidth
		slice_ceiling=min_z + (index+0.5)*binwidth
#		this_slice=where(logical_and(xxs[:,4]<slice_ceiling,xxs[:,4]>slice_floor),1,0)  # This is currently broken.
		this_slice=xpx[np.logical_and((xpx[:,4]<slice_ceiling),(xpx[:,4]>slice_floor))]
#		print "slice info:", index , slice_floor, slice_ceiling, len(this_slice)
		D=SingleSliceAnalysis(this_slice,xpx,slice_floor,slice_ceiling,bunch_charge)
		print D
		sliceMatrix[index][:]=D
		index=index+1.0

	return sliceMatrix


def SingleSliceAnalysis(cut_xpx,full_xpx,zmin,zmax,bunch_charge):
	cms=2.998e8
	width=(zmax-zmin)

	zcenter=zmin+width/2.0


	slice_particles=len(cut_xpx)

	slice_charge=bunch_charge*slice_particles/len(full_xpx)
# 0 or 1 particles cause lots of issues.  1 Particle has 0 emittance.  Make sure dimensions match!
#	if slice_particles<10:
#		return [zcenter, width, slice_particles, slice_charge, slice_charge*cms/width, np.mean((cut_xpx[:,5])/511000.0), np.mean(cut_xpx[:,0]), np.mean(cut_xpx[:,2]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]



	slice_current=slice_charge*cms/width

	cut_momentum=(cut_xpx[:,5])
	
	slice_pav=np.mean(cut_momentum)
	slice_beta=np.sqrt(1.0-1.0/slice_pav**2.0)


	slice_xcen=np.mean(cut_xpx[:,0])
	slice_ycen=np.mean(cut_xpx[:,2])

	slice_energyspread=np.std(cut_xpx[:,5]/np.mean(cut_xpx[:,5]) - 1.0)	

	x_centered=cut_xpx[:,0] - slice_xcen
	xprime_centered=cut_xpx[:,1]/np.mean(cut_xpx[:,5])-np.mean(cut_xpx[:,1]/np.mean(cut_xpx[:,5]))

	x=cut_xpx[:,0]
	xprime=cut_xpx[:,1]

	y=cut_xpx[:,2]
	yprime=cut_xpx[:,3]

	y_centered=cut_xpx[:,2] - slice_ycen
	yprime_centered=cut_xpx[:,3]/np.mean(cut_xpx[:,5])-np.mean(cut_xpx[:,3]/np.mean(cut_xpx[:,5]))

	z=cut_xpx[:,4]

#	energy=cut_xpx[:,5]
#	cut_momentum

	z_centered=cut_xpx[:,4] - zcenter
	slice_delta = cut_momentum/np.mean(cut_momentum)-1.0



#	zdata=particleData['z']-np.mean(particleData['z'])
#	zprimedata=particleData['pz']/np.mean(particleData['pz'])-1


	emitnx=slice_beta*slice_pav*np.sqrt( np.mean(x_centered**2.0)*np.mean(xprime_centered**2.0)-np.mean(x_centered*xprime_centered)**2.0)
	emitny=slice_beta*slice_pav*np.sqrt( np.mean(y_centered**2.0)*np.mean(yprime_centered**2.0)-np.mean(y_centered*yprime_centered)**2.0)
	

	emitnz=slice_beta*slice_pav*np.sqrt( np.mean(z_centered**2.0)*np.mean(slice_delta**2.0)-np.mean(z_centered*slice_delta)**2.0)

#	emitnx=slice_pav*sqrt( np.mean(x**2.0)*np.mean(xprime**2.0)-np.mean(x*xprime)**2.0)
#	emitny=slice_pav*sqrt( np.mean(y**2.0)*np.mean(yprime**2.0)-np.mean(y*yprime)**2.0)
	TBrightness=slice_current/(4.0*np.pi*np.pi*emitnx*emitny)
	FullBrightness=slice_charge/(emitnx*emitny*emitnz)
#	print np.mean(x_centered), np.mean(xprime_centered), np.mean(y_centered), np.mean(yprime_centered), np.mean(cut_xpx[:,5])
#	print std(x_centered), std(xprime_centered), std(y_centered), std(yprime_centered), std(cut_xpx[:,5])
#	enx=
#	eny=
### This now needs to include some measure for the TOTAL brightness?  Philippe suggests emittance, but slice longitudinal emittance is kind of iffy?  Very few particles.
#	return [zcenter, width, slice_particles, slice_charge, slice_current, slice_pav, slice_xcen, slice_ycen, emitnx, emitny, slice_energyspread]	
	return [zcenter, width, slice_particles, slice_charge, slice_current, slice_pav, slice_xcen, slice_ycen, emitnx, emitny, slice_energyspread, TBrightness, emitnz, FullBrightness]


