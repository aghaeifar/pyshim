import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine

from . import shimming_base, lsqlin


class shimming_b1(shimming_base):
    def __init__(self, work_directory=None) -> None:
        super().__init__(work_directory=work_directory)
        # if file_protocol is not None:
        #     self.set_protocol(file_protocol) 
        # else:
        #     self.set_protocol(self._b0_obj.hdr['MeasYaps'])

    
    def create_system_matrix(self, b1_cx, b0_hz, mask, kspace_traj:np.ndarray, rf_len:np.ndarray, rf_gap:np.ndarray):
        '''
            b1_cx :        B1 complex, mag in [nT/V], shape (x,y,z,coil)
            b0_hz:         B0 inhomogeneity, [Hz], shape (x,y,z)
            mask:          mask, shape (x,y,z)
            kspace_traj:   k-space trajectory, shape (num of RFs, num of kspace points, 3 [xk,yk,zk])
            rf_len:        RF length, [second], shape (num of RFs)
            rf_gap:        RF gap between consecutive RFs, [second], shape (num of RFs) -> the first element is normally 0
        '''
        # read nifti files
        b1      = nib.load(b1_cx)
        b0      = nib.load(b0_hz)
        mask    = nib.load(mask)
        affine  = b1.affine
        # check consistency of the input data
        if b1.header.get_data_shape()[:-1] != b0.header.get_data_shape() or b1.header.get_data_shape()[:-1] != mask.header.get_data_shape():
            raise ValueError('b1_mag, b1_phs, b0 and mask must have the same shape')
        if not np.array_equal(b1.affine , b0.affine) or not np.array_equal(b1.affine , mask.affine):
            raise ValueError('b1_mag, b1_phs, b0 and mask must have the same affine')
        if kspace_traj.shape[0] != rf_len.shape[0] or kspace_traj.shape[0] != rf_gap.shape[0] or kspace_traj.shape[2] != 3:
            raise ValueError('kspace_traj, rf_dwelltime and rf_gap must have the same length')
        # read data
        
        mask    = np.nonzero(mask.get_fdata().flatten())
        b0      = b0.get_fdata().flatten()[mask]
        b1      = 1e-9 * b1.get_fdata() # nT/V -> T/V

        sz      = b1.shape[0:3]
        nRF     = kspace_traj.shape[0]
        
        nPos    = mask.size
        T       = np.sum(rf_len) + np.sum(rf_gap)

        # calculate spatial position of voxels 
        xv, yv, zv = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]), np.arange(sz[2]), indexing='ij')
        xyz = apply_affine(affine, np.column_stack((xv.flatten()[mask], yv.flatten()[mask], zv.flatten()[mask] ))) # we must transform RAS to gradient coordinate !!!!!!!
        # is xyz in meter? !!!!!!!!!!
        A = np.zeros((nPos, nRF), dtype=np.complex128)
        for i, traj in enumerate(kspace_traj):
            t = np.sum(rf_len[:i]) + np.sum(rf_gap[:i]) + rf_len[i]/2 # time at the center of the RF
            kr = xyz @ traj.T # Mx3 @ 3xN -> MxN
            A[:,i] = 1j*2*np.pi*42.57638474e6*rf_len[i] * np.exp(1j*2*np.pi*b0*(t-T)) * np.exp(1j*2*np.pi*kr)

        A_pTx = list()
        for b1_cha in np.moveaxis(b1, -1, 0):
            A_pTx.append(A * b1_cha.flatten()[mask])
 
        return np.concatenate(A_pTx, axis=1)