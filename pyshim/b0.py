
import numpy as np
import nibabel as nib
from . import opt


def calc_b0_shim(shimmaps_nii, b0_nii, masks_nii, lb:np.ndarray, up:np.ndarray):
    '''
    calculate the shim
    b0_and_masks: list of nifti files,
        we suppose the first element in the list is the b0 map, and the every other is the mask
        if b0_and_masks is None, we use the files outputed from resample_to_standard_sapce() function
    '''
    b0_map   = nib.load(b0_nii)
    shimmaps = nib.load(shimmaps_nii)

    if shimmaps.header.get_data_shape()[0:3] != b0_map.header.get_data_shape():
        raise ValueError('shimmaps and b0_map must have the same shape')
    if not np.array_equal(shimmaps.affine , b0_map.affine):
        raise ValueError('shimmaps and b0_map must have the same affine')
    if shimmaps.header.get_data_shape()[3] != lb.shape[0] and shimmaps.header.get_data_shape()[3] != up.shape[0]:
        raise ValueError('shimmaps and lb/up must have the same length')
    
    output = {'shims':list(), 'std_residuals':list(), 'std_initials':list()}
    for m in masks_nii:
        mask = nib.load(m)
        if b0_map.header.get_data_shape() != mask.header.get_data_shape() or not np.array_equal(mask.affine , b0_map.affine):
            raise ValueError('b0_map and mask must have the same shape and affine')
        # weighting mask 
        mask_data = np.sqrt(mask.get_fdata())
        b0_data   = np.reshape(b0_map.get_fdata() * mask_data, (-1))
        shimmaps_data = np.reshape(shimmaps.get_fdata() * mask_data[..., np.newaxis], (-1, shimmaps.header.get_data_shape()[3]))
        # binary mask to reduce the size of the data and speed up the calculation
        mask_0 = (b0_data == 0) | (shimmaps_data[...,0] == 0)
        b0_data = b0_data[~mask_0]
        shimmaps_data = shimmaps_data[~mask_0, :]

        shims_value, err = opt.lsqlin(shimmaps_data, b0_data, lb, up)

        output['shims'].append(shims_value)
        output['std_residuals'].append(err)
        output['std_initials'].append(np.std(b0_data))
    return output
