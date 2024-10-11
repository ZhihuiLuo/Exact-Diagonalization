import numpy as np
import matplotlib.pyplot as plt

t = 1
def Gbethe(z):
    return (z-np.sign(z)*np.sqrt(np.abs(z**2-4*t**2))*1j)/(2*t**2)



beta = 3
Nw = 20
wr = 1j*(2*np.arange(Nw)+1)*np.pi/beta



mu = 0

G = Gbethe(wr)
g0 = 1/(wr+mu-t**2*G)

# plt.plot(wr.imag,G.imag,'-x')
plt.plot(wr.imag,g0.imag,'-o')



#
G0 = g0[:,np.newaxis,np.newaxis]
# Nbath = 4

# assuming ts are all real

def gen_Hbath_V(ts,Nb,Nbath):
    assert len(ts)==(Nbath**2+Nbath)/2+Nbath*Nb, 'error'
    eb,tb = ts[:-Nb*Nbath], ts[-Nb*Nbath:]
    
    V = np.reshape(tb, [Nb,Nbath])  # hyb
    
    Hbath = np.diag(eb[:Nbath]) # bath site energies
    i0,i1 = Nbath,Nbath*2-1
    for i in range(1,Nbath):
        v = eb[i0:i1]
        Hbath += np.diag(v,k=i)+np.diag(v,k=-i).conj()
        i0 = i1
        i1 = i0+Nbath-i-1
        
    return Hbath,V

def gen_Himp(G0):
    # temporary taking impurity level from G0[-1]
    # assuming no direct hybridization between impurities
    return np.diag(np.linalg.inv(G0[-1]).diagonal().real)

def build_H_full(ts, Nbath,beta,G0):
    Nw,Nb = G0.shape[:2]
    Hbath,V = gen_Hbath_V(ts,Nb,Nbath)
    
    fc = complex if any(np.iscomplex(ts)) else float
    H = np.zeros([Nb+Nbath,Nb+Nbath],fc)
    
    wr = 1j*(2*np.arange(Nw)+1)[:,np.newaxis,np.newaxis]*np.pi/beta
    Himp = gen_Himp(G0)
    Hbath,V = gen_Hbath_V(ts, Nb, Nbath)
    
    H[:Nb,:Nb] = Himp
    H[Nb:,Nb:] = Hbath
    H[:Nb,Nb:] = V
    H[Nb:,:Nb] = V.T.conj()
    return H
    
def G_trial(ts,Nbath,beta,G0):
    Nw,Nb = G0.shape[:2]
    
    H = build_H_full(ts, Nbath,beta,G0)
    # Himp = H[:Nb,:Nb]; Hbath = H[Nb:,Nb:]; V = H[:Nb,Nb:]
    
    wr = 1j*(2*np.arange(Nw)+1)[:,np.newaxis,np.newaxis]*np.pi/beta
        
    # both are okey
    # Gt_inv = wr*np.eye(Nb)-(Himp+V@np.linalg.inv(wr*np.eye(Nbath)-Hbath)@V.T.conj() ) 
    Gt = np.linalg.inv(wr*np.eye(Nb+Nbath)-H)[:,:Nb,:Nb]    
    # print('dG=',np.max(np.abs(Gt_inv-Gt_inv2)))
    
    return Gt

def chi2(ts, Nbath,beta,G0):
    Nw = G0.shape[0]
    Gt = G_trial(ts,Nbath,beta,G0)
    
    return np.abs(np.linalg.inv(G0)-np.linalg.inv(Gt)).sum()/(Nw+1)/Nb**2

#%%
from scipy.optimize import minimize
def callback(x):
    # print(x[:4])
    pass
    
wx = (2*Nw+1)*np.pi/beta
Nbath = 3
Nb = 1

Nts = int((Nbath**2+Nbath)/2+Nbath*Nb)  # total parameters


bounds = [(-wx,wx) for i in range(Nts)]

# rd = 

bounds = None
options = {'disp':False}
tol = None
tol = 1e-20
method = 'BFGS'
# method = 'TNC' x
method = None
# method = 'SLSQP'
# method = 'L-BFGS-B' x 
# method = 'COBYLA' x
# method = 'Nelder-Mead' x
# method = 'CG' x
# method = 'Powell' x


Ntrail = 10
df = []
Nbin = []

for i in range(Ntrail):
    ts = [wx*0.1*np.random.random() for i in range(Nts)]
    res = minimize(chi2,x0=ts,args=(Nbath,beta,G0),bounds=bounds,callback=callback,
               options=options,tol=tol,method=method)

    df.append(res.fun)
    Nbin.append(res.x)
    
    print(res.fun)
    
tf = res.x

H = build_H_full(tf, Nbath, beta, G0)
print(H)

Gt = G_trial(tf, Nbath, beta, G0)

plt.plot(wr.imag,G0[:,0,0].real,'-xk')
plt.plot(wr.imag,(Gt[:,0,0]).real,'-xr')
plt.plot(wr.imag,G0[:,0,0].imag,'-ok')
plt.plot(wr.imag,(Gt[:,0,0]).imag,'-or')

#%%
from ED import *
from scipy.linalg import expm
M = 1 
# Bl = build_full_basis(M)
# bl = list(np.squeeze(Bl))
bl = [0,1,2,3]

shows(bl, M)  # show the basis


# load HU ================================
U,JH = 7,0 # on-site U, Hund's coupling
sl = [[0]]
Ul = [[U,JH]]
H = load_H_Kanamori(bl, M, sl, Ul,hartree_shift=True).toarray()

cu = map2mtx(bl, [c(b,0) for b in bl]).toarray()
cd = map2mtx(bl, [c(b,1) for b in bl]).toarray()

cdu = map2mtx(bl, [cdag(b,0) for b in bl]).toarray()
cdd = map2mtx(bl, [cdag(b,1) for b in bl]).toarray()


beta = 1
expm(-beta*H).trace()
