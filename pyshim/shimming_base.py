import numpy as np
import nibabel as nib
from nilearn.image import resample_img
from pathlib import Path
from tqdm import tqdm


prefix = 'pyshim_'

def set_work_directory(work_directory):
    w_dir = Path(work_directory)
    w_dir.mkdir(parents=True, exist_ok=True)
    # clear folder from previous runs
    p = w_dir.glob("pyshim_*.nii")
    print(f'Cleaning folder {w_dir}')
    for f in p:            
        f.unlink()


def create_standard_space(res=1.5, fov=300, nifti_target=None):
    """
    Calculate the affine transformation, defining the standard space
    res: resolution in the standard space (mm)
    fov: field-of-view in the standard space (mm)
    nifti_file: nifti file defining the standard space
    """
    if nifti_target is not None and Path(nifti_target).is_file():
        std_affine = nib.load(nifti_target).affine
        std_size   = nib.load(nifti_target).shape[0:3]
        return std_affine, std_size
    
    if fov<0 or res<0 or fov<res:
        raise ValueError('fov and res must be positive and fov must be larger than res')

    std_affine = np.hstack((np.eye(4,3)*res, np.array([-fov/2,-fov/2,-fov/2,1]).reshape(4,1)))
    std_size   = [int(x) for x in [fov/res]*3]
    return std_affine, std_size


def resample_to_standard_sapce(filenames, std_affine, std_size, work_directory):
    """
    transform input nifti to the stanard space
    filenames: list of nifti files
    """
    work_directory = Path(work_directory)
    target_filenames = list()
    for f in tqdm(filenames, desc='resampling to standard space'):
        if not Path(f).is_file():
            raise FileNotFoundError(f'The specified file {f} was not found.')
        target_filenames.append(work_directory.joinpath(prefix + Path(f).name).as_posix())
        img = nib.load(f)
        
        inter_method = 'nearest' if sorted(list(np.unique(img.get_fdata()))) == [0, 1] else 'continuous'
        img_out = resample_img(img, std_affine, std_size, interpolation=inter_method)
        nib.save(img_out, target_filenames[-1])

    return target_filenames


def write_shims(shims:list, filename):
    '''
    write shims to a file
    shims: list of 1D numpy arrays
    filename: name of the file to write
    '''
    shims_table = np.concatenate(shims, axis=0)
    np.savetxt(filename, shims_table, fmt='%.3f')


# ---------------------------------------------------------------------
# least square solution with constraints
# ---------------------------------------------------------------------
def opt_lsqlin(A:np.ndarray, b:np.ndarray, lb:np.ndarray, ub:np.ndarray):
    '''
    Solving Ax = b 
    subject to lb <= x <= ub
    '''
    from scipy.optimize import lsq_linear
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.size or lb.size != ub.size or lb.size != A.shape[1]:
        raise ValueError('A must be 2D and b must be 1D')
    
    res = lsq_linear(A, -b, bounds=(lb, ub))
    x = res.x
    err = np.std(res.fun)
    return x, err


# ---------------------------------------------------------------------
# magnitude least square solution for complex data
# ---------------------------------------------------------------------    
def opt_mls(A:np.ndarray, b:np.ndarray, reg=0.0, n_iter=50):
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
        x = Ainv @ b  
        b = A @ x
        err.append(np.mean((np.abs(b) - target)**2))
        pbar.set_description(f'error {err[-1]:.3f}/{err[0]:.3f}')
        b  = target * b / np.abs(b) # = target * np.exp(1j*np.angle(b)) ;  update b with target magnitude and new phase 
        b[np.isnan(b)] = target[np.isnan(b)] # in case np.abs(b) in line above was zero

    return x, err


