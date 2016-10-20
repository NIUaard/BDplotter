'''
Mimic Astra;s lineplot
usage 
        python -i AstraLinePlot rfgun 001
'''

from BDplotterInit import *
import BDplotter as BD
import sys

arg = sys.argv 
filename = arg[1]
run  = arg[2]


# load astra file [C is empty is Cemit is not available)
X,Y,Z,C=BD.LoadAstraEmit(filename,run)

BD.PlotEmit1plt(X,Y,Z)
BD.PlotSize1plt(X,Y,Z)
BD.PlotEnergy1plt(X,Y,Z)

plt.show()

