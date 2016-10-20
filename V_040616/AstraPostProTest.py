from BDplotterV102014Init import *
import BDplotterV102014 as BD
import sys

arg = sys.argv 
filename = arg[1]


# load astra file
PHSP=BD.LoadAstraPhaseSpace(filename)

#print PHSP[0][]
BD. DensityplotwProjec2x2(PHSP['x'],PHSP['y'],21)
#BD.DensityPlot(PHSP['x'],PHSP['y'],21)

plt.figure()
#BD.DensityPlot_w_Hprojec_Zeroed (PHSP['x'],PHSP['y'],21)

# plot transverse phase space
#BD. DensityplotwProjec2x2(PHSP['x'],PHSP['y'],21)

# plot longitudinal phase space
BD. DensityplotwProjec2x2(PHSP['z'],PHSP['pz'],21)

plt.figure()

cms=299792458.0
f,bff=BD.BunchFormFactor(PHSP['z']/cms, 1e10, 1e12, 500,'true')
plt.loglog (f,bff)

# display everything
plt.show()
