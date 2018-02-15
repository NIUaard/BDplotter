from BDplotterInit import *
import BDplotter as BD
import sys

#arg = sys.argv 
#filename = arg[1]

filename='rb2cav-023940.0841.001'


# load astra file
PHSP=BD.LoadAstraPhaseSpace(filename)

#print PHSP[0][]

z = PHSP['z']-np.mean(PHSP['z'])
pz= PHSP['pz']-np.mean(PHSP['pz'])

print np.mean(PHSP['pz'])
plt.figure()
BD.DensityPlot(1000*z,pz*1e-6,201)
plt.xlabel (r'z (mm)')
plt.ylabel (r'$\delta{p_z}$ (MeV/c)')
plt.tight_layout()

X=BD.UniformSliceAnalysis(PHSP, 201, 200e-12)

plt.figure()
plt.plot (X[:,0], X[:,4]*100)
plt.plot (X[:,0], X[:,10]*X[:,5])
plt.xlabel (r'z (mm)')
plt.ylabel (r'$\sigma_{p_z}^u$ (eV/c)')
plt.tight_layout()


plt.show()
