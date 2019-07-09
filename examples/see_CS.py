import BDplotter as BD
import matplotlib.pyplot as plt 


filename='flatbeamtrans062019tospectro'

CS=BD.LoadElegantTwiss(filename+'.twi')
Mag=BD.LoadElegantMag(filename+'.mag')

grid = plt.GridSpec(10,1, wspace=0.4, hspace=0.3)
ax1=plt.subplot (grid[0,0])
ax1.plot (Mag['z'], Mag['profile'],'r')
ax1.axis('off')

ax2=plt.subplot (grid[1:9,0])
ax2.plot (CS['s'], CS['betax'], label='$\beta_x$')
ax2.plot (CS['s'], CS['betay'], label='$\beta_y$')

ax2t=ax2.twinx()
ax2t.plot (CS['s'], CS['etax'],'g', label='$\eta_x$')

ax2.set_xlabel  (r'$s$ (m)')
ax2.set_ylabel  (r'$\beta$ functions (m)')
ax2t.set_ylabel (r'$\eta_x$ (m)')

plt.show()
