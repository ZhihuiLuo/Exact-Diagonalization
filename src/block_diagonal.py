import numpy as np
from scipy.linalg import block_diag
import copy

def check_hermitian(matrix):
    return np.allclose(matrix,np.swapaxes(matrix,-1,-2).conj())


class BD:
    
    def __init__(self,*mts):
        
        if not mts:
            self.mts = []
            self.blk_dims = []
            self.size = 0
        else:
            self.load(*mts)
            
    def load(self,*mts):
        self.size = len(mts)
        # assert self.check_hermitian()
        self.mts = [np.matrix(m) for m in mts]
        self.blk_dims = [len(m) for m in mts]
    
    def check_hermitian(self):
        return all(list(map(check_hermitian, self))) 
        
    def __call__(self):
        return self.mts
        
    def __repr__(self):
        return f"{type(self).__name__}(size={self.size},mts={self.mts})"
        
    def push(self,matrix):
        # assert check_hermitian(matrix)  #impose hermitian symmetry
        
        self.mts.append(np.matrix(matrix))
        self.blk_dims.append(len(matrix))
        self.size += 1
    
    def pop(self):
        assert self.size>0
        self.mts.pop()
        self.blk_dims.pop()
        self.size -= 1
    def zeros_like(self):
        mts_new = [np.zeros_like(m) for m in self]
        return BD(*mts_new)
    def copy(self):
        return copy.deepcopy(self)
        
    def __iter__(self):
        return iter(self.mts)
    
    def __getitem__(self,idx):
        return self.mts[idx]
    
    def __len__(self):
        return self.size
    
    def m(self):
        return block_diag(*self.mts)
    
    def __add__(self,other):
        assert isinstance(other, (list,tuple,BD,int,float))
        if other==0: return self.copy()
        
        mts_new = [l+r for l,r in zip(self,other)]
        return BD(*mts_new)
    def __radd__(self,other):
        return self.__add__(other)
    def __neg__(self):
        mts_new = [-m for m in self]
        return BD(*mts_new)
    def __sub__(self,other):
        assert isinstance(other, (list,tuple,BD,int,float))
        if other==0: return self.copy()

        mts_new = [l-r for l,r in zip(self,other)]
        return BD(*mts_new)
    def __rsub__(self,other):
        return -self.__sub__(other)
    
    def __mul__(self,other):
        assert isinstance(other, (list,tuple,BD,float,int))
        if isinstance(other, (float,int)):
            mts_new = [other*l for l in self]
        else:   
            mts_new = [l*r for l,r in zip(self,other)]
        return BD(*mts_new)
    def __rmul__(self,other):
        assert isinstance(other, (list,tuple,BD,float,int))
        if isinstance(other, (float,int)):
            mts_new = [other*l for l in self]
        else:  
            mts_new = [r*l for l,r in zip(self,other)]
        return BD(*mts_new)
        
    def eigvalsh(self):
        assert self.check_hermitian()
        vas = []
        for mtx in self.mts:
            vas.append(np.linalg.eigvalsh(mtx) )
        return vas
        
    def eigh(self):
        assert self.check_hermitian()
        vas,ves = [],[]
        for mtx in self.mts:
            va,ve = np.linalg.eigh(mtx)
            vas.append(va); ves.append(ve)
        return vas,ves
    def inv(self):
        mts_new = [np.linalg.inv(m) for m in self]
        return BD(*mts_new)
        
        
if __name__=='__main__':
    m = [[1]],[[1,5],[5,2]]
    bd = BD(*m)
    
    mp = [[2]],[[-1,3],[3,2]]
    bdp = BD(*mp)
    
    bd0 = mp*bd
    bd1 = bd*mp
    # bd = bdp-bd
    
    
