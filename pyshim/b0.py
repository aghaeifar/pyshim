
import numpy as np
import nibabel as nib
from . import shimming_base, lsqlin

class shimming_b0(shimming_base):
    def __init__(self, res=1.5, fov=300, work_directory=None) -> None:
        super().__init__(res=res, fov=fov, work_directory=work_directory)
        # if file_protocol is not None:
        #     self.set_protocol(file_protocol) 
        # else:
        #     self.set_protocol(self._b0_obj.hdr['MeasYaps'])

    
    def get_data(self):
        pass
        # return self._b0_obj


    def calc_shim(self, shimmaps_nii, lb:np.ndarray, up:np.ndarray, b0_masks_nii=None):
        '''
        calculate the shim
        b0_and_masks: list of nifti files,
            we suppose the first element in the list is the b0 map, and the every other is the mask
            if b0_and_masks is None, we use the files outputed from resample_to_standard_sapce() function
        '''
        if b0_masks_nii is None:
            b0_masks_nii = self._target_filenames
        b0_nii    = b0_masks_nii[0]
        masks_nii = b0_masks_nii[1:]

        b0_map   = nib.load(b0_nii)
        shimmaps = nib.load(shimmaps_nii)

        if shimmaps.header.get_data_shape()[0:3] != b0_map.header.get_data_shape():
            raise ValueError('shimmaps and b0_map must have the same shape')
        if not np.array_equal(shimmaps.affine , b0_map.affine):
            raise ValueError('shimmaps and b0_map must have the same affine')
        if shimmaps.header.get_data_shape()[3] != lb.shape[0] and shimmaps.header.get_data_shape()[3] != up.shape[0]:
            raise ValueError('shimmaps and lb/up must have the same length')
        
        self._output.clear()
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

            shims_value, err = lsqlin(shimmaps_data, b0_data, lb, up)

            self._output.shims.append(shims_value)
            self._output.std_residuals.append(err)
            self._output.std_initials.append(np.std(b0_data))
