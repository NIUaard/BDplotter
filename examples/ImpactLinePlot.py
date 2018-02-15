'''
Mimic Astra;s lineplot
usage 
        python -i ImpactLinePlot rfgun 
'''

from BDplotterInit import *
import BDplotter as BD
import sys

arg = sys.argv 
filename = arg[1]


# load astra file [C is empty is Cemit is not available)
X,Y,Z=BD.LoadImpactSig(filename)

BD.PlotEmit1plt(X,Y,Z)
#plt.ylim(0,1e-4)
BD.PlotSize1plt(X,Y,Z)
#BD.PlotEnergy1plt(X,Y,Z)
# to plot enegy use this:
fig, ax1=plt.subplots()
ax1.plot (Z['z'],Z['rmsprime']/Z['avgp']*100,color='blue', linewidth=2.0, label=r'$\sigma_{Etot}$')
ax1.legend(loc=7)
ax1.set_ylim(0,5)
ax1.set_ylabel(r'relative energy spread (\%)', fontsize=22)
ax1.set_xlabel(r'distance (m)', fontsize=22)
BD.FormatLabelSci()
ax2 = ax1.twinx()
ax2.plot (Z['z'],Z['avgp']*0.511,color='red', linewidth=2.0, label=r'$\sigma_{z}$')
plt.xlabel('distance (m)', fontsize=22)
ax2.set_ylabel(r'total energy (MeV)', fontsize=22, color="red")
for label in ax2.get_yticklabels():
    label.set_color("red")
#   ax2.legend()
plt.tight_layout()
plt.show()

