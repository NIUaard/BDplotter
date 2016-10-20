import sys
import os
#os.putenv ("PYTHONPATH","/opt/nicadd/contrib/piot/Local_Install/lib/python2.4/site-packages/")
#from pylab import *

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

#from pylab import *


__version__ = "2.0.0"
#Change Log:
## CRP=Christopher R. Prokop
## PP=Philippe Piot




# CRP Note, 11-20-2013:  Added Comments for each function with regards to their intended usages.  This will eventually be copied into some proper documentation, as it is annoying to try to remember what functions do what.  In all honesty, much of this could be avoided with some proper usage of Matplotlib commands, or flag functions.  That is a possible idea for a rewrite, which would be fairly simple as far as rewrites are concerned.



# CRP Note, 09-06-2013:   Formalized changes into a version system now for clarity and eventual passing on to other users.  


#### CRP Note, 09-06-2013:  SciPy really isn't used here at all.  It may be useful at some point, and is possible for re-inclusion.
#import scipy

#### CRP Note, 09-06-2013: decimal was used at some point, but was commented out.  
##from decimal import Decimal




# CRP Note, 09-06-2013:  Comments on general growth from PP initial version to the control of CRP:  bdplotter has grown tremendously since I originally received it from Philippe in early 2011.   The same basic math is performed over and over to start off each of the copied functions, and could be simplified by making the initial definitions of dx, nx, etc. part of a function.   The different versions come down to the display of histograms, with or without the density plot, histograms in what direction, with what parameters, etc.   Making a plotting tool for general use is complicated and inconsistent.  

### In all probability, it could be greatly simplified.  

# Last modified PP, 03-23-2011






##   x=X-X.mean()
##   y=Y-Y.mean()
##   xmin = x.min()
##   xmax = x.max()
##   ymin = y.min()
##   ymax = y.max()








#### CRP Note, 09-06-2013:  Pylab gives errors.  The code needs to be cleaned up to use pl. instead, i.e. that the "from pylab imprt *" should likely be removed.


# CRP Note, 11-20-2013:   This block is for simple, default definitions for plot parameters.  You can simple call bdplotter.BD_Init(), or define a similar block of code in your own script.  I prefer the latter option, as it is something that continually needs to be tweaked.
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



# CRP Note, 11-20-2013:  This is the historical "standard" of bdplotter usage.  It takes x and y data that would typically be used for a scatter plot, then deposits it on to a 2D hexbin.  This basic function uses 151 grides in both x and y, color codes them on a normalized log scale based on the number of particles in that hex.  Then it adjusts the axises.  
def BD_DensityPlot(X,Y):
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


# CRP Note, 11-20-2013:  This is based on BD_DensityPlot(X,Y), but with an additional argument that controls the number of grid bins.  It is a bit of a misnomer, as it is numbBins^2 for the total.
def BD_DensityPlot_wbin(X,Y,numBins):
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
   pl.axis([xminplot, xmaxplot, yminplot, ymaxplot])


# CRP Note, 11-20-2013:  This is based on BD_DensityPlot(X,Y), with a projection on subplots to the two sides.
def BD_DensityPlot_w_projec_sub (X,Y):
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
   


# CRP Note, 11-20-2013:  This is based on BD_DensityPlot(X,Y), with a projection on the same plot.
def BD_DensityPlot_w_projec (X,Y):
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
   BD_DensityPlot(X,Y)



# CRP Note, 11-20-2013:  This is based on BD_DensityPlot(X,Y), with a horizontal projection on the same plot.
def BD_DensityPlot_w_Hprojec (X,Y):
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


# CRP Note, 11-20-2013:  This is based on BD_DensityPlot(X,Y), with a projection that goes to zero at the edges.
def BD_DensityPlot_w_Hprojec_Zeroed (X,Y):
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   
   numbins=151
   xhist,edgeshist = np.histogram(x,normed=0, bins=numbins)
   numparts=x.size
   numbins=xhist.size
   binlength=dx/numbins*1.0
   binpop=xhist*1.0
   maxpop=xhist.max()

   totalpop=sum(binpop)
   adjustedpop=binpop/maxpop*(max_y-min_y)*0.3
   print binpop, adjustedpop


   newhx=pl.zeros(len(adjustedpop)+2)
   newhx[0]=0.0
   newhx[1:(len(adjustedpop)+1)]=adjustedpop
   newhx[-1]=0.0

   edgesZeros=pl.zeros(len(edgeshist[1:])+2)
   edgesZeros[0]=min(edgeshist[1:])-np.std(edgeshist[1:])*.03
   edgesZeros[1:(len(edgeshist[1:])+1)]=edgeshist[1:]
   edgesZeros[(len(edgeshist[1:])+1)]=max(edgeshist[1:])-np.std(edgeshist[1:])*.03


   pl.step(edgesZeros,newhx, color='red') # plot as step lines instead of bars
#pl.step(zeroededgesx[1:],zeroedHx, color='red') # plot as step lines instead of bars
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



# CRP Note, 11-20-2013:  This gives the current profile based on the longitudinal (X) and energy (Y) distribution, with bunch charge as a parameter.
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
   plt.plot(edgeshist[1:],binCurrent)

# CRP Note, 11-20-2013:  This is based on BD_DensityPlot(X,Y), with a projection represents the current of some bunch charge, and a user specified number of bins that is used for both the hex and 
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
   plt.step(edgeshist[1:],binCurrent)


# CRP Note, 11-20-2013:  This is based on BD_DensityPlot(X,Y), with a projection represents the current of some bunch charge, and a user specified number of bins that is used for the profile.  It also allows the color to be specified.  This was notably used for the Bunch Compressor study, to compared current profiles of different distributions on the same plot.
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

   binCurrentZeros=np.zeros(len(binCurrent)+2)
   binCurrentZeros[0]=0
   binCurrentZeros[1:(len(binCurrent)+1)]=binCurrent
   binCurrentZeros[(len(binCurrent)+1)]=0

   edgesZeros=np.zeros(len(edgeshist[1:])+2)
   edgesZeros[0]=min(edgeshist[1:])-np.std(edgeshist[1:])*.03
   edgesZeros[1:(len(edgeshist[1:])+1)]=edgeshist[1:]
   edgesZeros[(len(edgeshist[1:])+1)]=max(edgeshist[1:])-np.std(edgeshist[1:])*.03


   plt.step( edgesZeros,binCurrentZeros,color)


# Current Profile with a tag and bunch charge
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
   print peakCurrent
#   plt.plot(edgeshist[1:],binCurrent,label=tag)
   binCurrentZeros=np.zeros(len(binCurrent)+2)
   binCurrentZeros[0]=0
   binCurrentZeros[1:(len(binCurrent)+1)]=binCurrent
   binCurrentZeros[(len(binCurrent)+1)]=0

   edgesZeros=np.zeros(len(edgeshist[1:])+2)
   edgesZeros[0]=min(edgeshist[1:])-np.std(edgeshist[1:])*.0001
   edgesZeros[1:(len(edgeshist[1:])+1)]=edgeshist[1:]
   edgesZeros[(len(edgeshist[1:])+1)]=max(edgeshist[1:])-np.std(edgeshist[1:])*.0001

#   plt.step(edgeshist[1:],binCurrent,color)
   plt.step( edgesZeros,binCurrentZeros,label=tag)







def BD_CurrentProfile_Steps_Color_Labeled (X,Y,BunchCharge,bins,color,tag):
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

   binCurrentZeros=np.zeros(len(binCurrent)+2)
   binCurrentZeros[0]=0
   binCurrentZeros[1:(len(binCurrent)+1)]=binCurrent
   binCurrentZeros[(len(binCurrent)+1)]=0

   edgesZeros=np.zeros(len(edgeshist[1:])+2)
   edgesZeros[0]=min(edgeshist[1:])-np.std(edgeshist[1:])*.03
   edgesZeros[1:(len(edgeshist[1:])+1)]=edgeshist[1:]
   edgesZeros[(len(edgeshist[1:])+1)]=max(edgeshist[1:])-np.std(edgeshist[1:])*.03

#   plt.step(edgeshist[1:],binCurrent,color)
   plt.step( edgesZeros,binCurrentZeros,color,label=tag)






# CRP Note, 11-20-2013:  This returns the value of the peak current.  This is good for numerical computation, and not necessarily for plotting.
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
   binCurrentZeros=np.zeros(len(binCurrent)+2)
   binCurrentZeros[0]=0
   binCurrentZeros[1:(len(binCurrent)+1)]=binCurrent
   binCurrentZeros[(len(binCurrent)+1)]=0

   edgesZeros=np.zeros(len(edgeshist[1:])+2)
   edgesZeros[0]=min(edgeshist[1:])-np.std(edgeshist[1:])*.0001
   edgesZeros[1:(len(edgeshist[1:])+1)]=edgeshist[1:]
   edgesZeros[(len(edgeshist[1:])+1)]=max(edgeshist[1:])-np.std(edgeshist[1:])*.0001

#   plt.step(edgeshist[1:],binCurrent,color)
#   plt.step( edgesZeros,binCurrentZeros,label=tag)
   return peakCurrent




# CRP Note, 11-20-2013:  This plots histograms on the x and y axises.
def BD_Xhistogram (X,Y,BinLength):
# bunchCharge is in Coulombs.  
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
#   plt.figure(PlotNumber)
   print Hx.max() , Hy.max(), dx, dy

#   hx[0]

   pl.step(edgesx[1:],hx) # plot as step lines instead of bars
   plt.legend()









# CRP Note, 11-20-2013:  This plots a histogram on the x axis for a specified plot number.  It made sense at one point, but is not particularly useful now.
def BD_Xhistogram_PlotNumber (X,Y,BinLength,PlotNumber,LineCode,Label):
# bunchCharge is in Coulombs.  
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
   plt.figure(PlotNumber)
   print Hx.max() , Hy.max(), dx, dy
   pl.step(edgesx[1:],hx,label=Label) # plot as step lines instead of bars
   plt.legend()


# X Histogram, specifying the location.
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
   NumBins=np.ceil(dx/BinLength)
   hx,edgesx = np.histogram(x, normed=0, bins=NumBins)
   Hx=hx/hx.max()*dy/3.0+(min_y+0.005*abs(min_y))
   hy,edgesy = np.histogram(y, normed=0, bins=NumBins)
   Hy=hy/hy.max()*dx/3.0+(min_x+0.005*abs(min_x))
   subplot(LocationCode)
   pl.step(edgesx[1:],hx) # plot as step lines instead of bars

# CRP Note, 11-20-2013: Histogram on the Y axis.
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

# CRP Note, 11-20-2013: Histogram on the Y axis, for a specified plot number.
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
   
   NumBins=np.ceil(dx/BinLength)
   hx,edgesx = np.histogram(x, normed=0, bins=NumBins)
   Hx=hx/hx.max()*dy/3.0+(min_y+0.005*abs(min_y))
   hy,edgesy = np.histogram(y, normed=0, bins=NumBins)
   Hy=hy/hy.max()*dx/3.0+(min_x+0.005*abs(min_x))
   subplot(LocationCode)
   pl.step(hy,edgesy[1:]) # plot as step lines instead of bars


# Y histogram, specifying the subplot location
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
   
   NumBins=np.ceil(dx/BinLength)
   hx,edgesx = np.histogram(x, normed=0, bins=NumBins)
   Hx=hx/hx.max()*dy/3.0+(min_y+0.005*abs(min_y))
   hy,edgesy = np.histogram(y, normed=0, bins=NumBins)
   Hy=hy/hy.max()*dx/3.0+(min_x+0.005*abs(min_x))
#   print Hx.max() , Hy.max(), dx, dy
   subplot(LocationCode)
   pl.step(hy,edgesy[1:]) # plot as step lines instead of bars



def BD_Return_CurrentProfile_Steps (X,Y,BunchCharge,bins):
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
   plt.step(edgeshist[1:],binCurrent)
   return [edgeshist[1:], binCurrent]

#def Interpolate (X,Y,Xmin,Xmax,NumPoints):


def GiveHistogramDataScale(X,Y,numbins,scale):
   x=X-X.mean()
   y=Y-Y.mean()
   max_x=x.max()
   max_y=y.max()
   min_x=x.min()
   min_y=y.min()
   dx=max_x-min_x
   dy=max_y-min_y
   
###   numbins=151
   xhist,edgeshist = np.histogram(x,normed=0, bins=numbins)
   numparts=x.size
   numbins=xhist.size
   binlength=dx/numbins*1.0
   binpop=xhist*1.0
   maxpop=xhist.max()

   totalpop=sum(binpop)

   adjustedpop=binpop/maxpop*(max_y-min_y)*scale
#   print binpop, adjustedpop

   newhx=np.zeros(len(adjustedpop)+2)
   newhx[0]=0.0
   newhx[1:(len(adjustedpop)+1)]=adjustedpop
   newhx[-1]=0.0

   edgesZeros=np.zeros(len(edgeshist[1:])+2)
   edgesZeros[0]=min(edgeshist[1:])-pl.std(edgeshist[1:])*.03
   edgesZeros[1:(len(edgeshist[1:])+1)]=edgeshist[1:]
   edgesZeros[(len(edgeshist[1:])+1)]=max(edgeshist[1:])-pl.std(edgeshist[1:])*.03
  
   return edgesZeros, newhx

#[edges,pops]=GiveHistogramData(X,Y,numbins)


def Advanced_DensityPlot_HistogramSettings(X,Y,numbins,linecolor,y_offset,scale):
	[edges,pops]=GiveHistogramDataScale(X,Y,numbins,scale)
   	plt.step(edges,pops+y_offset,color='red')
        BD_DensityPlot_wbin(X,Y,numbins)





####
def another_test_split_plot(X,Y,numbins,linecolor,y_offset):
	[edges,pops]=GiveHistogramDataScale(X,Y,numbins,scale)
   	plt.step(edges,pops+y_offset,color='red')
        BD_DensityPlot_wbin(X,Y,numbins)








def test_histogramcall(X,Y,numbins):
	[edges,pops]=GiveHistogramDataScale(X,Y,numbins,scale)

def test_split_plot(X,Y,numbins):
	[edges,pops]=GiveHistogramDataScale(X,Y,numbins,scale)
   	plt.step(edges,pops)
        BD_DensityPlot(X,Y)











###################################  BELOW THIS POINT WILL BE BROKEN OFF IN TO ITS OWN FILE:








# CRP Note, 11-20-2013: This is the good plot to use.  It makes proper use of default values and flags to handle exceptions, rather than having separate functions for each.
# CRP Note, 11-22-2013: More improvements.  There is still an issue with going to zero that needs to be corrected, but it is not really noticeable on plots.
# CRP Note, 11-22-2013: TBD-  Should offset be a value, or a fraction fo the limit of the plot?  My guess is the former, as there are circumstances where only a section of density plot is displayed.
# CRP Note, 11-22-2013: BUG, or misunderstood behavior.  The color scale resets if you use a later function to 
def BD_DensityPlot_ElaborateOptions(X,Y, numHexes=51, numXHistBins=51, numYHistBins=51, show_density=1, show_xcurrent=0, show_xhist=0, show_yhist=1, xhist_scale=0.3,   yhist_scale=0.3, histogram_linecode='r-', bunch_charge=0, xoffset=0, yoffset=0, density_type='log', threshold_population=0, color_scale=pl.cm.Greys):

#### Most of these variables adjust the arguments used for Matplotlib's hexbin function:  
##   http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hexbin


#X, the X values of the data set.  MANDATORY
#Y, the Y values of the data set to be plotted.  MANDATORY

#numHexes=51    This is the number of bins on each side for which to perform the hexbinning.

#numXHistBins=51  These values set the number of bins for the histogram.  Default is the same as numHexes, but there are situations where a different scale is justified. 
#numYHistBins=51

#show_density=1
#  This parameter dictates whether to show the density plot.  By default, this is the only, as "bdplotter" is short for "beam density plotter".  There are some cases, such as current profiles, where this should be turned off.  

#show_xcurrent=0
### Shows the current projection on the X-axis.  Scaling is TBD.  This should not be used as an overlay to a density hexbin plot.


#show_xhist=0
#show_yhist=0
### These parameters determine whether a histogram projection is shown on the x and y axis, respectively.  

#xhist_scale=0.3
#yhist_scale=0.3
##### These two parameters set the normalization of the histogram heights, as a fraction of the total heigth and width of the plot.  



#histogram_linecode='r-'

#bunch_charge=0

#xoffset=0
#yoffset=0

#density_type='log'.  Log plots the density as a function log_10(i+1).  If you want a linear scale, you DO NOT enter 'linear'.  Rather, use an integer that sets the number of gradients in color (divided from max to min).  This can also be used a sequence of values that specify the range for each color.

#threshold_population=0.  This value is subtracted from the bin population.

#color_scale=pl.cm.Greys.  This uses standard matplotlib color scales.  brg and jet are two of the more popular ones, but examples of all standard color maps can be found at:  http://www.physics.ox.ac.uk/Users/msshin/science/code/matplotlib_cm/

#




# X and Y coordinates are mandatory. 
	x=X-X.mean()
	y=Y-Y.mean()
	max_x=x.max()
	max_y=y.max()
	min_x=x.min()
	min_y=y.min()
	dx=max_x-min_x
	dy=max_y-min_y


### # CRP Note, 11-22-2013:Original code had an extra buffer, but that makes plots look poor when non-Greys color scales (or other color maps for which cm(pop=0) is not white.
	xinter=(max_x-min_x)
	xminplot=min_x#-.05*xinter
	xmaxplot=max_x#+.05*xinter
	yinter=(max_y-min_y)
	yminplot=min_y#-.05*yinter
	ymaxplot=max_y#+.05*yinter


	


   	if xoffset==0:
		xoffset=xminplot

	if yoffset==0:
		yoffset=yminplot


	if show_density != 0:
		pl.hexbin(x,y, cmap=color_scale, bins=density_type,gridsize=numHexes, mincnt=threshold_population)
		pl.axis([xminplot, xmaxplot, yminplot, ymaxplot])

	if show_xcurrent != 0:

		currenthist,edgeshist = np.histogram(x,normed=0, bins=numXHistBins)
		numparts=x.size
		numbins=currenthist.size
		binlength=dx/numbins
		binpop=currenthist
		maxpop=currenthist.max()
		maxChargePop=maxpop*bunch_charge/numparts
		binCharge=currenthist*bunch_charge/numparts
		totalCharge=sum(binCharge)
		peakCurrent=2.998e8*maxChargePop/binlength
		TotalParts=currenthist.sum()
		normTest=currenthist.sum()
		binCurrent=binCharge*2.998e8/binlength

		binCurrentZeros=np.zeros(len(binCurrent)+2)
		binCurrentZeros[0]=0
		binCurrentZeros[1:(len(binCurrent)+1)]=binCurrent
		binCurrentZeros[(len(binCurrent)+1)]=0

		edgesZeros=np.zeros(len(edgeshist[1:])+2)
		edgesZeros[0]=min(edgeshist[1:])-np.std(edgeshist[1:])*.03
		edgesZeros[1:(len(edgeshist[1:])+1)]=edgeshist[1:]
		edgesZeros[(len(edgeshist[1:])+1)]=max(edgeshist[1:])-np.std(edgeshist[1:])*.03

		plt.step( edgesZeros,binCurrentZeros+yoffset,histogram_linecode)


	if show_xhist != 0:

		xhist,edgeshist = np.histogram(x,normed=0, bins=numXHistBins)
		numparts=x.size
		numbins=xhist.size
		binlength=dx/numbins*1.0
		binpop=xhist*1.0
		maxpop=xhist.max()



		binPopZeros=np.zeros(len(binpop)+2)
		binPopZeros[0]=0
		binPopZeros[1:(len(binpop)+1)]=binpop
		binPopZeros[(len(binpop)+1)]=0



		edgesZeros=np.zeros(len(edgeshist[1:])+2)
		edgesZeros[0]=min(edgeshist[1:])-np.std(edgeshist[1:])
		edgesZeros[1:(len(edgeshist[1:])+1)]=edgeshist[1:]
		edgesZeros[(len(edgeshist[1:])+1)]=max(edgeshist[1:])-np.std(edgeshist[1:])

		plt.step( edgesZeros,(binPopZeros/maxpop)*yinter*xhist_scale+yoffset,histogram_linecode)



	if show_yhist != 0:

		yhist, yedgeshist = np.histogram(y,normed=0, bins=numYHistBins)

		numparts=y.size
		numbins=yhist.size
		binlength=dy/numbins*1.0
		binpop=yhist*1.0
		maxpop=yhist.max()


		binPopZeros=np.zeros(len(binpop)+2)
		binPopZeros[0]=0
		binPopZeros[1:(len(binpop)+1)]=binpop
		binPopZeros[(len(binpop)+1)]=0


		edgesZeros=np.zeros(len(yedgeshist[1:])+2)
		edgesZeros[0]=min(yedgeshist[1:])-np.std(yedgeshist[1:])
		edgesZeros[1:(len(yedgeshist[1:])+1)]=yedgeshist[1:]
		edgesZeros[(len(yedgeshist[1:])+1)]=max(yedgeshist[1:])-np.std(yedgeshist[1:])

		plt.step((binPopZeros/maxpop)*xinter*yhist_scale+xoffset,edgesZeros,histogram_linecode)

