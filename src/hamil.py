import numpy as np
from .exact_diag import *
import itertools as its


def load_H00(bl,H00):
    H0 = 0
    M = len(H00)
    for s,i,j in its.product(range(Ns),range(M),range(M)):
        if H00[i][j]==0: continue
        H0 += H00[i][j]*map2mtx(bl, [cdagc(b,i+M*s,j+M*s) for b in bl])
    return H0


def load_H_Kanamori(bl,M,sl,Ul,density_only=False,hartree_shift=False):
    '''
    Parameters
    ----------
    bl : list
        basis.
    M : int
        sites.
    sl : list-like
        list of shells [[o1,o2,o3],[o6,o8],...].
        Sites inside each sl[i] are treated as an independent shell that would 
        be assigned by Kanamori interation Ul[i].
    Ul : list-like
        list of interaction [[U1,J1],[U2,J2],...]. len(Ul)==len(sl) is required.
        Up = U-2*J is assumed.
    density_only : bool, optional
        if only preserve the density-density type interaction. The default is False.
    hartree_shift : bool, optional
        if perform a hartree shift? The default is False.
    sparse : bool, optional
        if return a scipy.sparse array? The default is False. 
    Returns
    -------
    H : array
        H.

    '''
    assert len(sl)==len(Ul), 'sl, Ul unmatched!'
    
    H = 0
    for s,shell in enumerate(sl):
        U,JH = Ul[s]; Up = U-2*JH
        for i in shell:
            if U==0: continue
            H += U*map2mtx(bl, [opts(b,[n,n],[i,i+M]) for b in bl])
            
            if hartree_shift:
                H -= U/2*map2mtx(bl, [n(b,i) for b in bl])
                H -= U/2*map2mtx(bl, [n(b,i+M) for b in bl])

        for i,j in its.product(shell,shell):
            if i>=j: continue
            
            if Up!=0:
                H += Up*map2mtx(bl, [opts(b,[n,n],[i+M,j]) for b in bl])
                H += Up*map2mtx(bl, [opts(b,[n,n],[i,j+M]) for b in bl])
            
            if Up-JH!=0:
                H += (Up-JH)*map2mtx(bl, [opts(b,[n,n],[i,j]) for b in bl])
                H += (Up-JH)*map2mtx(bl, [opts(b,[n,n],[i+M,j+M]) for b in bl])
            
            if hartree_shift and Up-JH/2!=0:
                H -= (Up-JH/2)*map2mtx(bl, [n(b,i) for b in bl])
                H -= (Up-JH/2)*map2mtx(bl, [n(b,i+M) for b in bl])
                H -= (Up-JH/2)*map2mtx(bl, [n(b,j) for b in bl])
                H -= (Up-JH/2)*map2mtx(bl, [n(b,j+M) for b in bl])
            
            if (not density_only) and JH!=0:
                H += JH*map2mtx(bl, [opts(b,[cdag,cdag,c,c],[i,i+M,j+M,j]) for b in bl])
                H -= JH*map2mtx(bl, [opts(b,[cdag,c,cdag,c],[i,i+M,j+M,j]) for b in bl])
                
                H += JH*map2mtx(bl, [opts(b,[cdag,cdag,c,c],[j,j+M,i+M,i]) for b in bl])
                H -= JH*map2mtx(bl, [opts(b,[cdag,c,cdag,c],[j,j+M,i+M,i]) for b in bl])
                
    return H



# s=0 up s=1 dn
if __name__=='__main__':
    from block_diagonal import BD
    M = 2
    
    U,JH = 10,0
    sl = [[0],[1]]
    Ul = [[U,JH],[U,JH]]
    
    H00 = [[0,-1],[-1,3]]
    
    Bl = build_full_basis(M)
    
    bd = BD()
    for bl in Bl:
        H0 = load_H00(bl,H00).toarray()
        HU = load_H_Kanamori(bl, M, sl, Ul,hartree_shift=True).toarray()
        
        bd.push(H0+HU)
        
    vp = np.sort(np.hstack(bd.eigvalsh()))
    vp0 = [-7.4186377,  -7.,         -7.,         -7.,         -5.30277564,
           -5.30277564, -2.30277564, -2.30277564, -1.69722436, -1.69722436,
           0.,          0.26276496,  1.30277564,  1.30277564,  6.,
           6.15587274]

    if np.allclose(vp,vp0): print('Benchmark passed.')
    else: print('Banchmark failed')
