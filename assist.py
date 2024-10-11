import numpy as np
from fractions import Fraction

Ns = 2
# function for showing bits ===============================
def tobit(b, M, fmt=None):
    st = (b>=0)*f'{b:0{Ns*M}b}'+(b<0)*f'{b:0{Ns*M+1}b}'
    # st = st[:-N//2]+' '+st[-N//2:]
    return st

def shows(bs,M):
    for b in bs: show(b,M)
        
def show(b,M):
    print(tobith(b,M))
    
#dn up
maph = {'00':'0','01':'u','10':'d','11':'2'}
def tobith(b,M): # in a human-readable way
    N = Ns*M
    st = tobit(b,M)
    std,stu = st[-N:-N//2], st[-N//2:]
    sth = st[:-N]
    for d,u in zip(std,stu):
        sth += maph[d+u]
    return sth
    
def showh(b,M):
    print(tobith(b,M))
    
    
    
def toFractional(fl,eps=1e-15,remove_sign=False):
    '''
    Find possible fraction expressions for an array of float. If the error 
    exceeds 'eps', '?' will be appended behind the string expression.

    Parameters
    ----------
    fl : list or numpy.ndarray
        float data.
    eps : float, optional
        Threshold to throw '?'. The default is 1e-15.
    remove_sign : bool, optional
        Whether remove the minus sign from the expressions. The default is False.

    Returns
    -------
    syml : numpy.ndarray
        Array of strings.

    '''
    syml = np.empty_like(fl, dtype='<U256')
    flp = np.abs(fl) if remove_sign else np.array(fl)
    
    assert isinstance(flp.flat[0], (int,float)), 'Not implemented yat.'
    
    for i,f in enumerate(flp.flat):
        if np.abs(np.trunc(f)-f)<eps: 
            syml.flat[i] = str(np.trunc(f))
            continue
        
        # direct trail
        fr1 = Fraction.from_float(f).limit_denominator(10000)
        df1 = abs(f-fr1.numerator/fr1.denominator)
        
        fr2 = Fraction.from_float(f**2).limit_denominator(10000)
        df2 = abs(f-np.sign(fr2)*np.sqrt(fr2.numerator/fr2.denominator))
    
        if df1<=df2: 
            if df1<eps: syml.flat[i] = '%d/%d'%(fr1.numerator,fr1.denominator)
            else: syml.flat[i] = '%d/%d?'%(fr1.numerator,fr1.denominator)
        else:
            if df2<eps: syml.flat[i] = '√%d/%d'%(fr2.numerator,fr2.denominator)
            else: syml.flat[i] = '√%d/%d?'%(fr2.numerator,fr2.denominator)
    return syml

