import numpy as np
import itertools as its
from .assist import *
import scipy.sparse as sp



Ns = 2  # DO NO MIDIFIED !!!
# ED core ==================================================
# Here define the most fundamental operations without making any furhter secure 
# check.
def bittest(b,i):
    return b&(1<<i)
def bitflip(b,i):
    return b^(1<<i)
def bitcount(b,i=0):
    return (b>>i).bit_count()
def commute(b,i=0):
    return 1-2*(bitcount(b,i+1)%2)

def invspin(b,M): # inverse all spin
    s = tobit(b,M)
    return int(s[:-Ns*M]+s[-M:]+s[-Ns*M:-M], 2)
# we want 0 to be also regarded as positive
def sign(b):
    return 2*(b>=0)-1

def c(b,i):
    if b is None or not bittest(abs(b), i): return None
    return commute(abs(b),i)*sign(b)*bitflip(abs(b),i)
def cdag(b,i):
    if b is None or bittest(abs(b), i): return None
    return commute(abs(b),i)*sign(b)*bitflip(abs(b),i)

# cdag_i c_j
def cdagc(b,i,j):
    return cdag(c(b,j), i)
def n(b,i):
    return cdag(c(b,i), i)

def opts(b,opt,idx):
    for i,o in zip(reversed(idx),reversed(opt)): b = o(b,i)
    return b

# build basis ==============================
def build_full_basis(M):
    Bl = []
    for Ni in range(Ns*M+1):
        for Nd in range(0,Ni+1):
            Nu = Ni-Nd
            if Nu>M or Nd>M: continue
            bl = build_basis(M, Nu, Nd)
            Bl.append(bl)
    return Bl


def build_basis(M,Nu,Nd):
    assert M>=Nu and M>=Nd, 'error input'
    b = list(its.product(build_basis0(M,Nd), build_basis0(M,Nu)))
    bl = [int(f'{ib:0{M}b}{jb:0{M}b}',2) for ib,jb in b]
    return bl
      
def build_basis0(M,Nu):
    basis = []
    minr = 0; maxr = 0
    for i in range(Nu):
        minr += 2**i
        maxr += 2**(M-i-1)
    for i in range(minr,maxr+1):
        nbit = 0
        for j in range(M):
            if bittest(i,j): nbit += 1
        if nbit==Nu:
            basis.append(i)
    return basis
# good quantum number =======================
def N(b):
    return bitcount(abs(b))

def Nd(b,M):
    return bitcount(abs(b), M)

def Nu(b,M):
    return N(b)-Nd(b,M)

def Sz(b,M):
    return N(b)/Ns-Nd(b,M)

# S2 =============================================
# gen S2 matrix under a (N,Sz) subspace
def S2(bl,M):
    Nb = len(bl)
    m = sp.dia_array(([Sz(bl[0],M)**2]*Nb, [0]), shape=(Nb,Nb)) # diagonal part
    for i,j in its.product(range(M),range(M)):
        m += map2mtx(bl, [opts(b,[cdag,c,cdag,c],[i,i+M,j+M,j]) for b in bl])/2
        m += map2mtx(bl, [opts(b,[cdag,c,cdag,c],[i+M,i,j,j+M]) for b in bl])/2
    return m
    
def gen_S2_trans(bl,M):
    S2va,S2ve = np.linalg.eigh(S2(bl,M).toarray())
    S2va = np.round(S2va).astype(int)
    S2set = np.sort(list(set(S2va)))
    S = np.round(np.sqrt(1+4*S2set)-1)/2
    
    idx = []
    for s in S2set:
        i,j = np.argwhere(S2va==s)[[0,-1],0]
        idx.append([i,j+1]) # j+1: for the convenience of indexing
    return S,idx,S2ve


sign_map = {1:'+',-1:'-'}
def analysis_S2(bl,M,prec=4,eps=1e-4,mode=0,fraction=False):
    '''
    print the basis transform  S2
    Parameters
    ----------
    bl : list
        basis.
    M : int
        sites.
    prec : float, optional
        show how many decimal place. The default is 4.
    eps : float, optional
        DESCRIPTION. The default is 1e-4.
    mode : 0 or 1, optional
        Different ways of printing the basis of bl. The default is 0.
    fraction : bool, optional
        Whether or not to find possible fraction expressions when printing the 
        basis transform. The default is False.

    Returns
    -------
    S : list
        good quantum number S.
    idx : list
        index information of blocks labeled by S. Convenience for 
        extracting blocks from a block-diagonal matrix.
    S2ve : matrix
        Eigenvectors of S2 matrix.

    '''
    Nb = len(bl)
    S,idx,S2ve = gen_S2_trans(bl,M)
    blh = [tobith(b,M) for b in bl]
    syml = toFractional(S2ve,remove_sign=True)
    cnt = 0
    for ib in range(Nb):
        if cnt<len(idx) and ib==idx[cnt][0]:
            print('S= %g, size= %d ============='%(S[cnt],idx[cnt][1]-idx[cnt][0]))
            cnt += 1
        ln = 'φ%d ='%ib
        for jb in range(Nb):
            val = abs(S2ve[jb,ib])
            if val<eps: continue
            sgn = sign_map[sign(S2ve[jb,ib])]
            if mode==0:
                if fraction: ln += f' {sgn}{syml[jb,ib]}|{blh[jb]}>'
                else: ln += f' {sgn}{val:.{prec}f}|{blh[jb]}>'
            elif mode==1:
                if fraction: ln += f' {sgn}{syml[jb,ib]}ϕ{jb}'
                else: ln += f' {sgn}{val:.{prec}f}ϕ{jb}'
        print(ln)
    print()
    return S,idx,S2ve

def Inv(bl,M):
    return map2mtx(bl, [invspin(b,M) for b in bl])

    


def reorder(b, order):
    s = tobit(abs(b), len(order)//Ns)[::-1]
    return sign(b)*int(''.join([s[_] for _ in order[::-1]]), 2)


def map2mtx(bi,bf):
    row,col,val = [],[],[]
    # biabs = [abs(b) for b in bi]
    for n,(i,f) in enumerate(zip(bi,bf)):
        if i!=None and f!=None:
            col.append(n)
            row.append(bi.index(abs(f)))
            # row.append(biabs.index(abs(f)))
            val.append(sign(f)/sign(i))
    # return row,col,val
    return sp.csr_array((val,(row,col)), shape=(len(bf),len(bi)))


