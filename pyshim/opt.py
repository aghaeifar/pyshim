
import numpy as np
from scipy.optimize import lsq_linear

# ---------------------------------------------------------------------
# least square solution with constraints
# ---------------------------------------------------------------------
def lsqlin(A:np.ndarray, b:np.ndarray, lb:np.ndarray, ub:np.ndarray):
    '''
    Solving Ax = b 
    subject to lb <= x <= ub
    '''
    
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.size or lb.size != ub.size or lb.size != A.shape[1]:
        raise ValueError('A must be 2D and b must be 1D')
    
    res = lsq_linear(A, -b, bounds=(lb, ub))
    x = res.x
    err = np.std(res.fun)
    return x, err


# ---------------------------------------------------------------------
# magnitude least square solution for complex data
# ---------------------------------------------------------------------    
def mls(A:np.ndarray, b:np.ndarray, reg=0.0, n_iter=50):
    '''
    Solving argmin 0.5*||A*x - b||^2 + λ/2*||x||^2
    Solution: x = (A^H A + λI)^-1 A^H b    (H: Hermitian transpose)
    '''
    from tqdm import tqdm
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.size:
        raise ValueError('A must be 2D and b must be 1D')
    
    target = np.abs(b)
    Ainv = np.linalg.pinv(np.conj(A.T) @ A + reg*np.eye(A.shape[1])) @ np.conj(A.T)

    err = list()
    for i in (pbar := tqdm(range(n_iter))):
        x   = Ainv @ b  
        b   = A @ x
        err.append(np.mean((np.abs(b) - target)**2))
        pbar.set_description(f'Loss={100*(1-err[-1]/err[0]):.1f}')
        # update b with target magnitude and new phase 
        babs= np.abs(b)
        b   = target * np.divide(b, babs, out=np.ones(b.shape, dtype=b.dtype), where=babs!=0) # this is ~4x faster than np.exp(1j*np.angle(b)) ;  

    return x, err
