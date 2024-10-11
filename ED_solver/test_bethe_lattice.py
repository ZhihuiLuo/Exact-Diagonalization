import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson


theta = lambda x: 1.*(x>0)



w = np.linspace(-3,3,300)

Aw = lambda D: 2*np.sqrt((D**2-w**2)*(D>np.abs(w)))/(np.pi*D**2)



Aw(2)

# plt.plot(w,Aw(2))


beta = 3
zr = 1j*(2*np.arange(-100,100)+1)*np.pi/beta
# zr = np.linspace(-5,5,500)+0.02j


D = 2

gzf = Aw(D)/(zr[:,np.newaxis]-w)
gz = simpson(gzf, x=w, axis=-1)


plt.plot(zr.imag,gz.imag,'-o')
# plt.ylim(-100,100)


#%%

zr = 1j*(2*np.arange(-100,100)+1)*np.pi/beta

t = 1
D = lambda z: (z-np.sign(z.imag)*np.sqrt(z**2-4*t**2))/(2*t**2)
D = lambda z: (z-np.sqrt(z**2-4*t**2))/(2*t**2)

plt.plot(zr.imag,D(zr).imag,'-x')
# plt.ylim(-1,1)

#%%
D = lambda z: (z-np.sign(z)*np.sqrt(z**2-4*t**2))/(2*t**2)
zr = np.linspace(-5,5,500)+0.0j
plt.plot(zr.real,D(zr),'-x')