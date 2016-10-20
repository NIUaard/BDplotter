#from pylab import *
# Last modified PP, 03-23-2011

import sys
import os
os.putenv ("PYTHONPATH","/opt/nicadd/contrib/piot/Local_Install/lib/python2.4/site-packages/")
from pylab import *

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy


def BD_Init():
   params = {'backend': 'ps',
	  'axes.labelsize': 18,
	  'text.fontsize': 18,
	  'legend.fontsize': 18,
	  'xtick.labelsize': 18,
	  'ytick.labelsize': 18,
	  'subplot.wspace': 0.8,
	  'subplot.hspace': 0.8,
	  'text.usetex': False}
   plt.rcParams.update(params)

def BD_DensityPlot(X,Y):
#read astra distribution
   x=X-X.mean()
   y=Y-Y.mean()
   xmin = x.min()
   xmax = x.max()
   ymin = y.min()
   ymax = y.max()
   xinter=(xmax-xmin)
   xminplot=xmin-.05*xinter
   xmaxplot=xmax+.05*xinter
   yinter=(ymax-ymin)
   yminplot=ymin-.05*yinter
   ymaxplot=ymax+.05*yinter
   pl.hexbin(x,y, cmap=pl.cm.Greys, bins='log',gridsize=151)
   pl.axis([xminplot, xmaxplot, yminplot, ymaxplot])

def BD_DensityPlot_wbin(X,Y,numBins):
#read astra distribution
   x=X-X.mean()
   y=Y-Y.mean()
   xmin = x.min()
   xmax = x.max()
   ymin = y.min()
   ymax = y.max()
   xinter=(xmax-xmin)
   xminplot=xmin-.05*xinter
   xmaxplot=xmax+.05*xinter
   yinter=(ymax-ymin)
   yminplot=ymin-.05*yinter
   ymaxplot=ymax+.05*yinter
   pl.hexbin(x,y, cmap=pl.cm.Greys, bins='log',gridsize=numBins)
#   hexbin(x,y, cmap=pl.cm.copper, bins='log',gridsize=151)
   pl.axis([xminplot, xmaxplot, yminplot, ymaxplot])

def BD_DensityPlot_w_projec_sub (X,Y):
#read astra distribution
   x=X-X.mean()
   y=Y-Y.mean()
   xmin = x.min()
   xmax = x.max()
   ymin = y.min()
   ymax = y.max()
   xinter=(xmax-xmin)
   xminplot=xmin-.05*xinter
   xmaxplot=xmax+.05*xinter
   yinter=(ymax-ymin)
   yminplot=ymin-.05*yinter
   ymaxplot=ymax+.05*yinter
   pl.subplot (2,2,1)
   BD_DensityPlot(X,Y)
   pl.subplot (2,2,2)
   pl.subplot (2,2,2)
   ny, binsy, patchy = pl.hist(y, bins=256, normed=1, histtype='step', orientation='horizontal')
   pl.axis([0, ny.max()*1.1, yminplot, ymaxplot])
   pl.subplot (2,2,3)
   nx, binsx, patchx = pl. hist(x, bins=256 , normed=1, histtype='step')
   pl.axis([xminplot, xmaxplot, 0, nx.max()*1.1 ])
   
def BD_DensityPlot_w_projec (X,Y):
#read astra distribution
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y






   
########### old method works fine
#   BD_DensityPlot(X,Y)
#   ny, binsy, histoy = histogram(y, bins=256, normed=1, orientation='horizontal', histtype='step', color='white')
#   nx, binsx, histox = histogram(x, bins=256, normed=1, histtype='step', color='white')
#   ny, binsy, histoy = pl.hist(y, bins=256, normed=1, histtype='step', color='white', alpha=0.0)
#   nx, binsx, histox = pl.hist(x, bins=256, normed=1, histtype='step', orientation='horizontal', color='white', alpha=0.0)
#   Nbinsx = binsx[:-1]
#   Nbinsy = binsy[:-1]
#   print nx
#   print Nbinsx
#   pl.plot (Nbinsx,nx-2.5,color='white')
#   pl.plot (ny-2.5, Nbinsy,color='white')
###########

# fix from Ivan -- matplotlib user mailing list
   hx,edgesx = np.histogram(x, normed=1, bins=151)
   Hx=hx/hx.max()*dy/3.0+(min_y+0.005*abs(min_y))
   hy,edgesy = np.histogram(y, normed=1, bins=151)
   Hy=hy/hy.max()*dx/3.0+(min_x+0.005*abs(min_x))
   
   print Hx.max() , Hy.max(), dx, dy
#   pl.bar(edges[:-1],Hx, width=edges[1]-edges[0])
#   pl.bar(edges[:-1]-10,h, width=edges[1]-edges[0]) #offset
#   pl.step(edgesx[1:],Hx, color='white') # plot as step lines instead of bars
#   pl.step(Hy,edgesy[1:], color='white') # plot as step lines instead of bars
   BD_DensityPlot(X,Y)
#   hist(x,bins=binsx,color='white',normed=1, histtype='step')
#   plot (nx, histox,color='white')
#   hist(x, bins=256, normed=1, histtype='step', color='white', , 
#   hist(x, bins=256 , normed=1, histtype='step')
   
def BD_DensityPlot_w_Hprojec (X,Y):
#read astra distribution
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   
   hx,edgesx = np.histogram(x, normed=1, bins=151)
   Hx=hx/hx.max()*dy/3.0+(min_y+0.005*abs(min_y))
   hy,edgesy = np.histogram(y, normed=1, bins=151)
   Hy=hy/hy.max()*dx/3.0+(min_x+0.005*abs(min_x))
   
   print Hx.max() , Hy.max(), dx, dy
   pl.step(edgesx[1:],Hx, color='red') # plot as step lines instead of bars
   BD_DensityPlot(X,Y)

def BD_ChangeAxisTick (NlocsX, NlocsY):
  locs, labels = pl.yticks()
  locsmax= round(max(locs))
  locsmin= round(min(locs))
  print locsmax, locsmin
  print locsmax-locsmin
  Nlocs = NlocsY
  Newlocs = np.zeros((Nlocs)) 
  i=np.arange(Nlocs)
  for j in i:
    Newlocs[j] = round(locsmin + (locsmax-locsmin)*i[j]/(Nlocs-1.0))
    
  pl.yticks (Newlocs, Newlocs)


  locs, labels = pl.xticks()
  locsmax= round(max(locs))
  locsmin= round(min(locs))
  print locsmax, locsmin
  print locsmax-locsmin
  Nlocs = NlocsY
  Newlocs = np.zeros((Nlocs)) 
  i=np.arange(Nlocs)
  for j in i:
    Newlocs[j] = round(locsmin + (locsmax-locsmin)*i[j]/(Nlocs-1.0))
    
  pl.xticks (Newlocs, Newlocs)

def BD_CurrentProfile (X,Y,BunchCharge):
# bunchCharge is in Coulombs.  
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   currenthist,edgeshist = np.histogram(x,normed=0, bins=100)
   numparts=x.size
   numbins=currenthist.size
   binlength=dx/numbins
   binpop=currenthist
   maxpop=currenthist.max()
   maxChargePop=maxpop*BunchCharge/numparts
   binCharge=currenthist*BunchCharge/numparts
   totalCharge=sum(binCharge)
   peakCurrent=2.998e8*maxChargePop/binlength
   TotalParts=currenthist.sum()
   normTest=currenthist.sum()
   binCurrent=binCharge*2.998e8/binlength
#   plot(edgeshist[1:],binCurrent)
   plt.plot(edgeshist[1:],binCurrent)

def BD_CurrentProfile_Steps (X,Y,BunchCharge,bins):
# bunchCharge is in Coulombs.  
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   currenthist,edgeshist = np.histogram(x,normed=0, bins=bins)
   numparts=x.size
   numbins=currenthist.size
   binlength=dx/numbins
   binpop=currenthist
   maxpop=currenthist.max()
   maxChargePop=maxpop*BunchCharge/numparts
   binCharge=currenthist*BunchCharge/numparts
   totalCharge=sum(binCharge)
   peakCurrent=2.998e8*maxChargePop/binlength
   TotalParts=currenthist.sum()
   normTest=currenthist.sum()
   binCurrent=binCharge*2.998e8/binlength
#   plot(edgeshist[1:],binCurrent)
   plt.step(edgeshist[1:],binCurrent)

def BD_CurrentProfile_Steps_Color (X,Y,BunchCharge,bins,color):
# bunchCharge is in Coulombs.  
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   currenthist,edgeshist = np.histogram(x,normed=0, bins=bins)
   numparts=x.size
   numbins=currenthist.size
   binlength=dx/numbins
   binpop=currenthist
   maxpop=currenthist.max()
   maxChargePop=maxpop*BunchCharge/numparts
   binCharge=currenthist*BunchCharge/numparts
   totalCharge=sum(binCharge)
   peakCurrent=2.998e8*maxChargePop/binlength
   TotalParts=currenthist.sum()
   normTest=currenthist.sum()
   binCurrent=binCharge*2.998e8/binlength
#   plot(edgeshist[1:],binCurrent)
   plt.step(edgeshist[1:],binCurrent,color)


def BD_CurrentProfile_Labeled (X,Y,NumBins,BunchCharge,tag):
# bunchCharge is in Coulombs.  
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   currenthist,edgeshist = np.histogram(x,normed=0, bins=NumBins)
   numparts=x.size
   numbins=currenthist.size
   binlength=dx/numbins
   binpop=currenthist
   maxpop=currenthist.max()
   maxChargePop=maxpop*BunchCharge/numparts
   binCharge=currenthist*BunchCharge/numparts
   totalCharge=sum(binCharge)
   peakCurrent=2.998e8*maxChargePop/binlength
   TotalParts=currenthist.sum()
   normTest=currenthist.sum()
   binCurrent=binCharge*2.998e8/binlength
#   plot(edgeshist[1:],binCurrent)
   print peakCurrent
   plt.plot(edgeshist[1:],binCurrent,label=tag)
#   plt.legend()


def BD_ReturnPeakCurrent (X,Y,NumBins,BunchCharge,tag):
# bunchCharge is in Coulombs.  
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   currenthist,edgeshist = np.histogram(x,normed=0, bins=NumBins)
   numparts=x.size
   numbins=currenthist.size
   binlength=dx/numbins
   binpop=currenthist
   maxpop=currenthist.max()
   maxChargePop=maxpop*BunchCharge/numparts
   binCharge=currenthist*BunchCharge/numparts
   totalCharge=sum(binCharge)
   peakCurrent=2.998e8*maxChargePop/binlength
   TotalParts=currenthist.sum()
   normTest=currenthist.sum()
   binCurrent=binCharge*2.998e8/binlength
#   plot(edgeshist[1:],binCurrent)
#   print peakCurrent
   plt.plot(edgeshist[1:],binCurrent,label=tag)
   return peakCurrent

def BD_Xhistogram (X,Y,BinLength,PlotNumber,LineCode,Label):
# bunchCharge is in Coulombs.  
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   
   # Need way to determine bin size.  ceiling(dx/expectedBinLength)?
   NumBins=np.ceil(dx/BinLength)
   hx,edgesx = np.histogram(x, normed=0, bins=NumBins)
   Hx=hx/hx.max()*dy/3.0+(min_y+0.005*abs(min_y))
   hy,edgesy = np.histogram(y, normed=0, bins=NumBins)
   Hy=hy/hy.max()*dx/3.0+(min_x+0.005*abs(min_x))
   plt.figure(PlotNumber)
   print Hx.max() , Hy.max(), dx, dy
#   pl.step(edgesx[1:],Hx, color='red') # plot as step lines instead of bars
   pl.step(edgesx[1:],hx,label=Label) # plot as step lines instead of bars
#   plot(edgeshist[1:],binCurrent)
#   print peakCurrent
#-X.mean()
#   plt.plot(edgeshist[1:],binCurrent,LineCode,label=Label)
   plt.legend()





def BD_Xhistogram_Subplot_Location (X,Y,BinLength,LocationCode):
# bunchCharge is in Coulombs.  
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   
   # Need way to determine bin size.  ceiling(dx/expectedBinLength)?
   NumBins=np.ceil(dx/BinLength)
   hx,edgesx = np.histogram(x, normed=0, bins=NumBins)
   Hx=hx/hx.max()*dy/3.0+(min_y+0.005*abs(min_y))
   hy,edgesy = np.histogram(y, normed=0, bins=NumBins)
   Hy=hy/hy.max()*dx/3.0+(min_x+0.005*abs(min_x))
#   print Hx.max() , Hy.max(), dx, dy
   subplot(LocationCode)
   pl.step(edgesx[1:],hx) # plot as step lines instead of bars


def BD_Yhistogram (X,Y,BinLength):
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   NumBins=np.ceil(dx/BinLength)
   hx,edgesx = np.histogram(x, normed=0, bins=NumBins)
   Hx=hx/hx.max()*dy/3.0+(min_y+0.005*abs(min_y))
   hy,edgesy = np.histogram(y, normed=0, bins=NumBins)
   Hy=hy/hy.max()*dx/3.0+(min_x+0.005*abs(min_x))
   pl.step(hy,edgesy[1:])

def BD_Yhistogram_Subplot_Location (X,Y,BinLength,LocationCode):
# bunchCharge is in Coulombs.  
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   
   # Need way to determine bin size.  ceiling(dx/expectedBinLength)?
   NumBins=np.ceil(dx/BinLength)
   hx,edgesx = np.histogram(x, normed=0, bins=NumBins)
   Hx=hx/hx.max()*dy/3.0+(min_y+0.005*abs(min_y))
   hy,edgesy = np.histogram(y, normed=0, bins=NumBins)
   Hy=hy/hy.max()*dx/3.0+(min_x+0.005*abs(min_x))
#   print Hx.max() , Hy.max(), dx, dy
   subplot(LocationCode)
   pl.step(hy,edgesy[1:]) # plot as step lines instead of bars

def BD_Yhistogram_Subplot_Location (X,Y,BinLength,LocationCode):
# bunchCharge is in Coulombs.  
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   
   # Need way to determine bin size.  ceiling(dx/expectedBinLength)?
   NumBins=np.ceil(dx/BinLength)
   hx,edgesx = np.histogram(x, normed=0, bins=NumBins)
   Hx=hx/hx.max()*dy/3.0+(min_y+0.005*abs(min_y))
   hy,edgesy = np.histogram(y, normed=0, bins=NumBins)
   Hy=hy/hy.max()*dx/3.0+(min_x+0.005*abs(min_x))
#   print Hx.max() , Hy.max(), dx, dy
   subplot(LocationCode)
   pl.step(hy,edgesy[1:]) # plot as step lines instead of bars

