
import numpy as np
import nibabel as nib
from . import opt


def calc_b0_shim(shimmaps_nii:nib.Nifti1Image, b0_nii:nib.Nifti1Image, masks:list[nib.Nifti1Image], lb:np.ndarray, up:np.ndarray):
    '''
    calculate the shim
    b0_and_masks: list of nifti files,
        we suppose the first element in the list is the b0 map, and the every other is the mask
        if b0_and_masks is None, we use the files outputed from resample_to_standard_sapce() function
    '''

    if shimmaps_nii.shape[0:3] != b0_nii.shape:
        raise ValueError('shimmaps and b0_map must have the same shape')
    if not np.array_equal(shimmaps_nii.affine , b0_nii.affine):
        raise ValueError('shimmaps and b0_map must have the same affine')
    if shimmaps_nii.shape[3] != lb.shape[0] and shimmaps_nii.shape[3] != up.shape[0]:
        raise ValueError('shimmaps and lb/up must have the same length')
    
    output = {'shims':list(), 'std_residuals':list(), 'std_initials':list()}
    for mask_nii in masks:
        if b0_nii.shape != mask_nii.shape or not np.array_equal(mask_nii.affine , b0_nii.affine):
            raise ValueError('b0_map and mask must have the same shape and affine')
        # weighting mask 
        mask_data = np.sqrt(mask_nii.get_fdata())
        b0_data   = (b0_nii.get_fdata() * mask_data).flatten()
        shimmaps_data = np.reshape(shimmaps_nii.get_fdata() * mask_data[..., np.newaxis], (-1, shimmaps_nii.shape[3]))
        # binary mask to reduce the size of the data and speed up the calculation
        mask = np.intersect1d(np.nonzero(b0_data), np.nonzero(np.sum(shimmaps_data, axis=-1)))
        if mask.shape[0] == 0:
            continue
        b0_data = b0_data[mask]
        shimmaps_data = shimmaps_data[mask, :]
        shims_value, err = opt.lsqlin(shimmaps_data, b0_data, lb, up)

        output['shims'].append(shims_value)
        output['std_residuals'].append(err)
        output['std_initials'].append(np.std(b0_data))
    return output
