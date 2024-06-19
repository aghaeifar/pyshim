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


def create_standard_space(res=1.5, fov=300, nifti_target:nib.Nifti1Image=None):
    """
    Calculate the affine transformation, defining the standard space
    res: resolution in the standard space (mm)
    fov: field-of-view in the standard space (mm)
    nifti_file: nifti file defining the standard space
    """
    if isinstance(nifti_target, nib.nifti1.Nifti1Image):
        std_affine = nifti_target.affine
        std_size   = nifti_target.shape[0:3]
        return std_affine, std_size
    
    if fov<0 or res<0 or fov<res:
        raise ValueError('fov and res must be positive and fov must be larger than res')

    std_affine = np.hstack((np.eye(4,3)*res, np.array([-fov/2,-fov/2,-fov/2,1]).reshape(4,1)))
    std_size   = [int(x) for x in [fov/res]*3]
    return std_affine, std_size


def resample_to_standard_sapce(*niis, std_affine, std_size):
    """
    transform input nifti to the stanard space
    niis: list of nifti objects
    """
    niis_out = list()
    for nii_in in tqdm(niis, desc='resampling to standard space'):
        inter_method = 'nearest' if sorted(list(np.unique(nii_in.get_fdata()))) == [0, 1] else 'continuous'
        niis_out.append(resample_img(nii_in, std_affine, std_size, interpolation=inter_method))
    return niis_out


def combine_masks(*mask_nii:nib.Nifti1Image):   
    nifti_arrays = [mask.get_fdata() for mask in mask_nii]
    return nib.Nifti1Image(np.prod(nifti_arrays, axis=0, dtype=np.float32), affine=mask_nii[0].affine)


def combine_nii(*nii:nib.Nifti1Image):
    nifti_arrays = [nii_in.get_fdata(dtype=nii_in.get_data_dtype()) for nii_in in nii]
    return nib.Nifti1Image(np.sum(nifti_arrays, axis=0), affine=nii[0].affine)


def scale_nii(nii:nib.Nifti1Image, factor):
    return nib.Nifti1Image(nii.get_fdata(dtype=nii.get_data_dtype())*factor, affine=nii.affine)

def calculate_err(input:nib.Nifti1Image, mask:nib.Nifti1Image, calc_std=True, calc_rmse=False, target_rmse=0):
    if np.prod(input.shape) != np.prod(mask.shape):  # input can be 4D but the last dimension must be 1
        raise ValueError('shape mismatch')
    
    input = input.get_fdata(dtype=input.get_data_dtype())  
    input = input.flatten()[np.nonzero(mask.get_fdata().flatten())]
    if np.iscomplex(input).any():
        input = np.abs(input)
    
    err = []
    if calc_std:
        err.append(np.std(input))
    if calc_rmse:
        err.append(np.sqrt(np.mean((input-target_rmse)**2)))
    return err


def write_shims(shims:list, filename):
    '''
    write shims to a file
    shims: list of 1D numpy arrays
    filename: name of the file to write
    '''
    shims_table = np.concatenate(shims, axis=0)
    np.savetxt(filename, shims_table, fmt='%.3f')



