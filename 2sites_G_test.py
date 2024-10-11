#%% green's function
from ED import *
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt

M = 2
H00 = [[0,-1],[-1,0]]

# from scipy.linalg import expm

U,JH = 7,0 # on-site U, Hund's coupling
sl = [[0],[1]]
Ul = [[U,JH],[U,JH]]


Bl = build_full_basis(M)
# Bl = list(np.squeeze(Bl))
Bl = [b for bl in Bl for b in bl]

H0 = load_H00(Bl,H00,M)
HU = load_H_Kanamori(Bl, M, sl, Ul,hartree_shift=False)

H = H0+HU
# H = H.toarray()

beta = 3
pf = expm(-beta*H)
Z = pf.trace()

E = (pf@H).trace()/Z
print('E= %f'%E)

Ntau = 40
Gtau = np.zeros([Ns,Ntau,M,M])
taur = np.linspace(0,beta,Ntau)
for s,i,j,t in its.product(range(Ns),range(M),range(M),range(Ntau)):
    # print(s,i,j,t)
    ci = map2mtx(Bl, [c(b,i+Ns*s) for b in Bl])
    cdagj = map2mtx(Bl, [cdag(b,j+Ns*s) for b in Bl])
    
    tau = taur[t]
    
    gm = expm((tau-beta)*H)@ci@expm(-tau*H)@cdagj
    Gtau[s,t,i,j] = -gm.trace()/Z
    
    
plt.plot(taur,Gtau[0,:,0,0],'-o')
plt.plot(taur,Gtau[0,:,0,1],'-o')
# plt.plot(taur,Gtau[0,:,1,0],'-o')


#%%
Nb = M**4
vaf,vef = np.linalg.eigh(H.toarray())

# Energy cut off
# cutoff = 1e-12
# ld = np.exp(-beta*(va0-va0[0]))
# ic = np.argwhere(ld<cutoff)[0,0]
# print('cutoff=',ic)

ic = 10

va = vaf[:ic]; ve = vef[:,:ic]




pfp = np.exp(-beta*va)
Zp = pfp.sum()

Ep = np.sum(pfp*va)/Zp
print('Ep= %f'%Ep)

# Hd = sp.dia_array((va,[0]),shape=(Nb,Nb))
ve[abs(ve)<1e-10] = 0

# re-normalized
# vep = ve/np.sqrt(np.sum(ve*ve,axis=1))[:,np.newaxis]

Uf = sp.csr_array(ve)
UfT = Uf.T.conj()

#
Gtauc = np.zeros([Ns,Ntau,M,M])

for s,i,j,t in its.product(range(Ns),range(M),range(M),range(Ntau)):
    # print(s,i,j,t)
    ci = map2mtx(Bl, [c(b,i+Ns*s) for b in Bl])
    cdagj = map2mtx(Bl, [cdag(b,j+Ns*s) for b in Bl])
    
    tau = taur[t]
    
    # 1
    # eL = Uf@sp.diags(np.exp((tau-beta)*va) )@UfT
    # eR = Uf@sp.diags(np.exp(-tau*va) )@UfT
    
    
    # 2
    eL = sp.diags(np.exp((tau-beta)*va) )  # Unstable for tau=0 !!!!!!!!!
    ci = UfT@ci@Uf
    eR = sp.diags(np.exp(-tau*va) )
    cdagj = UfT@cdagj@Uf
    
    # 3
    # eL = sp.diags(np.exp(tau*va) )  # Unstable for tau=0 !!!!!!!!!
    # ci = UfT@ci@Uf
    
    # eR = sp.diags(np.exp(-tau*va) )
    # cdagj = UfT@cdagj@Uf@sp.diags(np.exp(-beta*va))

    gm = eL@ci@eR@cdagj
    
    if (s,i,j)==(0,0,0) and t==0: 
        print(eL,'\n\n',eR,'\n\n',gm,'\n\n',gm.trace()/Zp)
    
    Gtauc[s,t,i,j] = -gm.trace()/Zp
    
    # fixing !!
    # Gtauc[:,0] = -np.eye(M)-Gtauc[:,-1]
    


plt.plot(taur,Gtau[0,:,0,0],'-ok')
plt.plot(taur,Gtau[0,:,0,1],'-ok')

plt.plot(taur,Gtauc[0,:,0,0],'-or')
plt.plot(taur,Gtauc[0,:,0,1],'-or')

#%% Gtau
# ic = 8
# Zp = np.sum(np.exp(-beta*va))

Gtauc = np.zeros([Ns,Ntau,M,M])
for s,i,j,t in its.product(range(Ns),range(M),range(M),range(Ntau)):
    ci = UfT@map2mtx(Bl, [c(b,i+Ns*s) for b in Bl])@Uf
    cdagj = UfT@map2mtx(Bl, [cdag(b,j+Ns*s) for b in Bl])@Uf
    tau = taur[t]
    
    
    for m,n in its.product(range(ic),range(ic)):
        # if m==n: continue
        Gtauc[s,t,i,j] -= ci[n,m]*cdagj[m,n]*np.exp(tau*(va[n]-va[m])) \
            *np.exp(-beta*va[n])/Zp
        

plt.plot(taur,Gtau[0,:,0,0],'-ok')
plt.plot(taur,Gtau[0,:,0,1],'-ok')

plt.plot(taur,Gtauc[0,:,0,0],'-or')
plt.plot(taur,Gtauc[0,:,0,1],'-or')
    

#%% Giw
Nw = 101
wr = (2*np.arange(Nw)+1)*np.pi/beta

def getGiw(Nw,ic=None):
    vaf,vef = np.linalg.eigh(H.toarray())
    va = vaf[:ic]; ve = vef[:,:ic]

    ve[abs(ve)<1e-10] = 0
    Uf = sp.csr_array(ve)
    UfT = Uf.T.conj()
    
    Giw = np.zeros([Ns,Nw,M,M],complex)
    for s,i,j,w in its.product(range(Ns),range(M),range(M),range(Nw)):
        ci = UfT@map2mtx(Bl, [c(b,i+Ns*s) for b in Bl])@Uf
        cdagj = UfT@map2mtx(Bl, [cdag(b,j+Ns*s) for b in Bl])@Uf
        iw = 1j*wr[w]
        
        for m,n in its.product(range(ic),range(ic)):
            # if m==n: continue
            Giw[s,w,i,j] += ci[n,m]*cdagj[m,n]/(iw+va[n]-va[m]) \
                *(np.exp(-beta*va[n])+np.exp(-beta*va[m]))/Z
            
    return Giw
   
Giwf = getGiw(Nw,ic=16)
#%%
Giw = getGiw(Nw,ic=6)

plt.plot(wr,Giwf[0,:,0,0].real,'-xk')
plt.plot(wr,Giwf[0,:,0,0].imag,'-ok')
plt.plot(wr,Giw[0,:,0,0].real,'-xr')
plt.plot(wr,Giw[0,:,0,0].imag,'-or')


#%% take from fft of Gtau
# poor for large beta !!!
from scipy.interpolate import interp1d
from scipy.integrate import simpson
I = np.newaxis

def Gtau2Giw(Gtau,beta,Nw):
    wr = (2*np.arange(Nw)+1)*np.pi/beta
    
    Ntau = Gtau.shape[1]
    taur = np.linspace(0,beta,Ntau)
    
    taurp = np.linspace(0,beta, int(40*beta))
    Gtaup = interp1d(taur, Gtau, axis=1, kind='cubic')(taurp)
    print(Gtaup.shape)
     
    
    kern = np.exp(1j*wr[:,I]*taurp)[I,:,:,I,I] *Gtaup[:,I]
    print(kern.shape)
    
    Giw = simpson(kern, x=taurp,axis=2)
    
    return Giw

Giwf = Gtau2Giw(Gtau,beta,Nw)
plt.plot(wr,Giwf[0,:,0,0].real,'-xr')
plt.plot(wr,Giwf[0,:,0,0].imag,'-or')

#%% density
den = np.zeros([Ns,M,M])
for s,i,j in its.product(range(Ns),range(M),range(M)):
    # print(s,i,j,t)
    cdagi = map2mtx(Bl, [cdag(b,i+Ns*s) for b in Bl])
    cj = map2mtx(Bl, [c(b,j+Ns*s) for b in Bl])
    
    gm = expm(-beta*H)@cdagi@cj
    den[s,i,j] = gm.trace()/Z
    
print(den[0])    
#%%
denp = np.zeros([Ns,M,M])

for s,i,j in its.product(range(Ns),range(M),range(M)):
    
    cdagi = UfT@map2mtx(Bl, [cdag(b,i+Ns*s) for b in Bl])@Uf
    cj = UfT@map2mtx(Bl, [c(b,j+Ns*s) for b in Bl])@Uf
 
    
    for m,n in its.product(range(ic),range(ic)):
        # if m==n: continue
        denp[s,i,j] += cdagi[n,m]*cj[m,n]*np.exp(-beta*va[n])/Zp
        
print(denp[0])
