from BDplotterInit import *
import BDplotter as BD
import sys

arg = sys.argv 
filename = arg[1]

filename='fort.40'

# load astra file
PHSP=BD.LoadImpactPhaseSpace(filename)

#print PHSP[0][]
BD.DensityPlot(PHSP['x'],PHSP['y'],21)

plt.figure()
#BD.DensityPlot_w_Hprojec_Zeroed (PHSP['x'],PHSP['y'],21)

# plot transverse phase space
BD. DensityplotwProjec2x2(PHSP['x'],PHSP['y'],21)

# plot longitudinal phase space
BD. DensityplotwProjec2x2(PHSP['z'],PHSP['pz'],21)



# display everything
plt.show()
