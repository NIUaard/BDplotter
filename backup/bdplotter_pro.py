
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


__version__ = "1.0.0"



def BD_Init():
	params = {'backend': 'ps',
	  	  'axes.labelsize': 25,
	    	  'text.fontsize': 25,
	  	  'legend.fontsize': 18,
	  	  'font.size': 20,
	  	  'label.size': 24,
	  	  'xtick.labelsize': 20,
	  	  'ytick.labelsize': 20,
	  	  'xtick.major.size' : 12,
	  	  'ytick.major.size' : 12,
	  	  'xtick.minor.size' : 9,
	  	  'ytick.minor.size' : 9,
	  	  'grid.linewidth'   :   0.8,
		  'figure.subplot.wspace': 0.25,
		  'figure.subplot.left': 0.13,
		  'figure.subplot.right': 0.94,
		  'figure.subplot.top': 0.94,
		  'figure.subplot.bottom': 0.12,
		  'figure.subplot.hspace': 0.15,
	  	  'text.usetex': True}
	plt.rcParams.update(params)



# CRP Note, 11-20-2013: This is the good plot to use.  It makes proper use of default values and flags to handle exceptions, rather than having separate functions for each.
# CRP Note, 11-22-2013: More improvements.  There is still an issue with going to zero that needs to be corrected, but it is not really noticeable on plots.
# CRP Note, 11-22-2013: TBD-  Should offset be a value, or a fraction fo the limit of the plot?  My guess is the former, as there are circumstances where only a section of density plot is displayed.

def BD_DensityPlot_Pro(X,Y, numHexes=51, numXHistBins=51, numYHistBins=51, show_density=1, show_xcurrent=0, show_xhist=0, show_yhist=1, xhist_scale=0.3,   yhist_scale=0.3, histogram_linecode='r-', bunch_charge=0, xoffset=None, yoffset=None, density_type='log', threshold_population=0, color_scale=pl.cm.Greys):

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


	

# Offset value for where to place the histogram projection.  When no value is entered, it is zeroed at the edge.
   	if xoffset==None:
		xoffset=xminplot
		
# Offset value for where to place the histogram projection.  When no value is entered, it is zeroed at the edge.
	if yoffset==None:
		yoffset=yminplot


	if show_density != 0:
		pl.hexbin(x,y, cmap=color_scale, bins=density_type,gridsize=numHexes, mincnt=threshold_population)
		pl.axis([xminplot, xmaxplot, yminplot, ymaxplot])

# Includes the projection of the horizontal distribution on the x axis as a temporal current, using the specified bunch charge.  Do not use this is bunch_charge=0.
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

# Includes the projection of the horizontal distribution on the x axis.
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

# Includes the projection of the vertical distribution on the y axis.
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



#	BD_Init()

