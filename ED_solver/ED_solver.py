import numpy as np
from scipy.optimize import minimize


class IMP:
    
    def __init__(self,beta,G0):
        self.beta = beta
        self.G0 = G0
        self.Nw,self.Nb = G0.shape[:2]
        self.wr = 1j*(2*np.arange(self.Nw)+1)[:,np.newaxis,np.newaxis]*np.pi/beta
        
        
    def gen_Hbath_V(self,ts,Nbath):
        Nb = self.Nb
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
    
    def gen_Himp(self):
        # temporary taking impurity level from G0[-1]
        # assuming no direct hybridization between impurities
        # return np.diag(np.linalg.inv(self.G0[-1]).diagonal().real)
    
        return np.linalg.inv(self.G0[-1]).real
    
    def build_H_full(self,ts, Nbath):
        Nb = self.Nb
        
        Himp = self.gen_Himp()
        Hbath,V = self.gen_Hbath_V(ts, Nbath)
        
        fc = complex if any(np.iscomplex(ts)) else float
        H = np.zeros([Nb+Nbath,Nb+Nbath],fc)
        H[:Nb,:Nb] = Himp
        H[Nb:,Nb:] = Hbath
        H[:Nb,Nb:] = V
        H[Nb:,:Nb] = V.T.conj()
        return H
        
    def build_G_full(self,ts,Nbath):
        Nb = self.Nb
        H = self.build_H_full(ts, Nbath)
        
        return np.linalg.inv(self.wr*np.eye(Nb+Nbath)-H)
    
    def chi2(self,ts, Nbath):
        Nb = self.Nb
        Gt = self.build_G_full(ts,Nbath)[:,:Nb,:Nb]
        return np.abs(np.linalg.inv(self.G0)-np.linalg.inv(Gt)).sum()/(self.Nw+1)/Nb**2

    # def init_ts(self,Nbath):
    #     Nts = int((Nbath**2+Nbath)/2+Nbath*self.Nb)
    #     wx = (2*self.Nw-1)*np.pi/self.beta
    #     return [wx*0.1*np.random.random() for i in range(Nts)]
    # def init_bounds(self):
    #     wx = (2*self.Nw-1)*np.pi/self.beta
    #     return 
    
    def find_H_full(self,Nbath,bounds=None,Ntrial=10,**kwargs):
        Nts = int((Nbath**2+Nbath)/2+Nbath*self.Nb)
        wx = (2*self.Nw-1)*np.pi/self.beta
        
        if bounds is not None: bounds = [(-wx,wx) for i in range(Nts)]
        
        dfun = []
        tss = []
        for i in range(Ntrial):
            ts = [wx*0.1*np.random.random() for i in range(Nts)]
            
            res = minimize(self.chi2,x0=ts,args=(Nbath,),bounds=bounds,
                       **kwargs)

            dfun.append(res.fun)
            tss.append(res.x)
            
            print('Error= ',res.fun)
            
        idx = np.argmin(dfun)
        print('Final error: ',dfun[idx])
        if dfun[idx]>1e-4: print('Warning: the result is not well converged!')
        
        self.H = self.build_H_full(tss[idx], Nbath)
        print('H_full=\n',self.H)

        return self.H
    
    def plot_last_result(self):
        import matplotlib.pyplot as plt
        Nfull = self.H.shape[0]
        G_full = np.linalg.inv(self.wr*np.eye(Nfull)-self.H)
        
        plt.plot(self.wr[:,0,0].imag,self.G0[:,0,0].real,'-xk',label='G0.real')
        plt.plot(self.wr[:,0,0].imag,self.G0[:,0,0].imag,'-ok',label='G0.imag')
        
        plt.plot(self.wr[:,0,0].imag,G_full[:,0,0].real,'-xr',label='Gt.real')
        plt.plot(self.wr[:,0,0].imag,G_full[:,0,0].imag,'-or',label='Gt.imag')
        
        plt.legend(frameon=False)
        
        
from ED import *
def solve_ED(beta,G0,Nbath,U):
    Nb = G0.shape[-1]
    M = Nb+Nbath
    
    Bl = build_full_basis(M)
    
    imp = IMP(beta,G0)
    H00 = imp.find_H_full(Nbath)
    
    sl = [[i] for i in range(Nb)]
    Ul = [[U,0] for i in range(Nb)]
    
    vas = []
    for bl in Bl:
        H0 = load_H00(bl,H00,M)
        
        HU = load_H_Kanamori(bl, M, sl, Ul)
        
        H = H0+HU
        va,ve = np.linalg.eigh(H.toarray())
        
        vas.append(va)
        
    return vas
    
    
# class ED_solver:  
#     def __init__(self,beta,G0,Nbath):
#         self.beta = beta
        
#         self.G0 = G0
#         self.Nb = G0.shape[-1]
#         self.Nbath = Nbath
#         self.M = self.Nb+self.Nbath
        
#         self.Bl = build_full_basis(self.M)
        
        
    
    
#     def find_H0_full(self,Nbath):
#         imp = IMP(self.beta,self.G0)
#         H_full = imp.find_H_full(Nbath)
    
    
if __name__=='__main__':
    t = 1
    def Gbethe(z):
        return (z-np.sign(z)*np.sqrt(np.abs(z**2-4*t**2))*1j)/(2*t**2)

    beta = 3
    Nw = 100
    wr = 1j*(2*np.arange(Nw)+1)*np.pi/beta

    mu = 0
    g0 = 1/(wr+mu-t**2*Gbethe(wr))
    G0 = g0[:,np.newaxis,np.newaxis]
    
    # imp = IMP(beta,G0)
    
    # Nbath = 4
    # H_full = imp.find_H_full(Nbath, Ntrial=10)
    
    
    Nstep = 10
    
    for i in range(Nstep):
        U = 2
        Nbath = 4
        vas = solve_ED(beta,G0,Nbath,U)