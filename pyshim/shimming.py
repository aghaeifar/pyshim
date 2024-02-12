import numpy as np
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_from_to

from tqdm import tqdm


prefix = 'pyshim_'

class output_shims():
    shims = None
    std_initials  = None
    std_residuals = None
    def __init__(self) -> None:
        self.shims = list()
        self.std_initials  = list()
        self.std_residuals = list()
    def clear(self):
        self.shims.clear()
        self.std_initials.clear()
        self.std_residuals.clear()


class shimming_base():
    _standard_space_affine = None
    _standard_space_size   = None
    _work_directory        = None    
    _target_filenames      = list()
    _output                = output_shims()

    def __init__(self, res=1.5, fov=300, work_directory=None) -> None:
        work_directory = str(Path.home().joinpath('pyshim')) if work_directory is None else work_directory
        self.set_work_directory(work_directory)
        self.set_standard_space(res, fov)
        self._target_filenames = None


    def set_work_directory(self, work_directory):
        w_dir = Path(work_directory)
        w_dir.mkdir(parents=True, exist_ok=True)
        # clear folder from previous runs
        p = w_dir.glob(prefix + '*.nii*')
        if len(list(p))>0:
            print(f'clearing {len(list(p))} files from work directory {w_dir}')
        for f in p:
            f.unlink()
        self._work_directory = w_dir


    def set_standard_space(self, res=1.5, fov=300, nifti_reference=None):
        """
        Calculate the affine transformation, defining the standard space
        res: resolution in the standard space (mm)
        fov: field-of-view in the standard space (mm)
        nifti_file: nifti file defining the standard space
        """

        if nifti_reference is not None and Path(nifti_reference).is_file():
            self._standard_space_affine = nib.load(nifti_reference).affine
            self._standard_space_size   = nib.load(nifti_reference).shape[0:3]
            return
        
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
            if not Path(f).is_file():
                raise FileNotFoundError(f'The specified file {f} was not found.')
            self._target_filenames.append(w_dir.joinpath(prefix + Path(f).name))
            img = nib.load(f)
            order = 0 if sorted(list(np.unique(img.get_fdata()))) == [0, 1] else 4
            if img.ndim == 3:
                img_out = resample_from_to(img, (std_size, std_affine), order=order)
            elif img.ndim == 4:
                img_out = [resample_from_to(img3, (std_size, std_affine), order=order) for img3 in nib.funcs.four_to_three(img)]
                img_out = nib.funcs.concat_images(img_out)
            else:
                raise ValueError(f'Input must be 2D or 3D: {f}')
            nib.save(img_out, self._target_filenames[-1])

        return self._target_filenames


    def get_standard_space(self):
        return self._standard_space_affine, self._standard_space_size

    def get_shims(self):
        return self._output.shims
    

    def write_shims(self, filename):
        """
        write shims to a file
        shims: 2D array of shims
        filename: name of the file to write
        """
        shims = np.concatenate(self._output.shims, axis=0)
        np.savetxt(filename, shims, fmt='%.3f')


# ---------------------------------------------------------------------
# least square solution with constraints
# ---------------------------------------------------------------------
def lsqlin(A:np.ndarray, b:np.ndarray, lb:np.ndarray, ub:np.ndarray):
    """
    Solving Ax = b 
    subject to lb <= x <= ub
    """
    from scipy.optimize import lsq_linear

    res = lsq_linear(A, -b, bounds=(lb, ub))
    solution = res.x
    err = np.std(res.fun)
    return solution, err

    
