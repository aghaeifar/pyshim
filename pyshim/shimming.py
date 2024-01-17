import numpy as np
from pathlib import Path
from nilearn.image import resample_img, load_img
from tqdm import tqdm


prefix = 'pyshim_'
standard_space_filename = prefix + 'standard_space.nii.gz'

class shimming_base():
    _standard_space_affine = None
    _standard_space_size   = None
    _work_directory        = None
    _target_filenames      = None

    def __init__(self, res=1.5, fov=300, work_directory=None) -> None:
        work_directory = str(Path.home().joinpath('pyshim')) if work_directory is None else work_directory
        self.set_work_directory(work_directory)
        self.set_standard_space(res, fov)
        self._target_filenames = list()


    def set_work_directory(self, work_directory):
        w_dir = Path(work_directory)
        w_dir.mkdir(parents=True, exist_ok=True)
        # clear folder from previous runs
        p = w_dir.glob(prefix + '*.nii*')
        for f in p:
            f.unlink()
        self._work_directory = w_dir


    def set_standard_space(self, res, fov):
        """
        Calculate the affine transformation, defining the standard space
        res: resolution in the standard space (mm)
        fov: field-of-view in the standard space (mm)
        """
        if fov<0 or res<0 or fov<res:
            raise ValueError('fov and res must be positive and fov must be larger than res')

        self._standard_space_affine = np.hstack((np.eye(4,3)*res, np.array([-fov/2,-fov/2,-fov/2,1]).reshape(4,1)))
        self._standard_space_size   = [int(x) for x in [fov/res]*3]


    def resample_to_standard_sapce(self, filenames):
        """
        transform input nifti to the stanard space
        filenames: list of nifti files
        """
        self._target_filenames = list()
        std_affine = self._standard_space_affine
        std_size   = self._standard_space_size
        w_dir      = self._work_directory
        for f in tqdm(filenames, desc='resampling to standard space'):
            self._target_filenames.append(w_dir.joinpath(prefix + Path(f).name))
            resample_img(load_img(f), target_affine=std_affine, target_shape=std_size).to_filename(self._target_filenames[-1])


    def get_standard_space(self):
        return self._standard_space_affine, self._standard_space_size



def lsqlin(a:np.ndarray, b:np.ndarray, lb:np.ndarray, ub:np.ndarray):
    """
    Solving ax = b 
    subject to lb <= x <= ub
    """
    import cvxpy as cp
    x = cp.Variable(len(b))
    prob = cp.Problem(cp.Minimize(cp.sum_squares(a @ x - b)), [lb <= x, x <= ub])
    prob.solve()
    return x.value
    
