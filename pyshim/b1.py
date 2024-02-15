import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine

gamma = 2*np.pi * 42.57638474e6 # rad/sec/T

def all_equal(*iterable):
    g = iter(iterable)
    try:
        first = next(g)
    except StopIteration:
        return True
    return all(np.array_equal(first, rest) for rest in g)


def create_system_matrix(b1_mag_nii, b1_phs_nii, b0_nii, mask_nii, kspace_traj:np.ndarray=np.array([[0.,0.,0.]]), rf_len:np.ndarray=np.array([1e-3]), rf_gap:np.ndarray=np.array([0.])):
    '''
        b1_mag:        B1 magnitude in [nT/V], shape (x,y,z,coil)
        b1_phs:        B1 phase in [rad], shape (x,y,z,coil)
        b0_hz:         B0 inhomogeneity, [Hz], shape (x,y,z)
        mask:          mask, shape (x,y,z)
        kspace_traj:   k-space trajectory, shape (num of RFs, 3 [xk,yk,zk])
        rf_len:        RF length, [second], shape (num of RFs)
        rf_gap:        RF gap between consecutive RFs, [second], shape (num of RFs) -> the first element is normally 0
    '''
    # read nifti files
    b1_mag  = nib.load(b1_mag_nii) # 
    b1_phs  = nib.load(b1_phs_nii)
    b0      = nib.load(b0_nii)
    mask    = nib.load(mask_nii)
    zooms   = np.array(b0.header.get_zooms()[:3]) # nib.Nifti1Header()
    offset  = b0.affine[:3,-1]

    # check consistency of the input data
    if not all_equal(b1_mag.header.get_data_shape()[:-1], b1_phs.header.get_data_shape()[:-1], b0.header.get_data_shape(), mask.header.get_data_shape()):
        raise ValueError('b1_mag, b1_phs, b0 and mask must have the same shape')
    if not all_equal(b1_mag.affine, b1_phs.affine, b0.affine, mask.affine):
        raise ValueError('b1_mag, b1_phs, b0 and mask must have the same affine')
    if not all_equal(kspace_traj.shape[0], rf_len.shape[0], rf_gap.shape[0]) or kspace_traj.shape[1] != 3:
        raise ValueError(f'kspace_traj {kspace_traj.shape}, rf_len {rf_len.shape} and rf_gap {rf_gap.shape} must have the same length')
    # read data
    mask    = np.nonzero(mask.get_fdata(dtype=mask.header.get_data_dtype()).flatten())
    b0      = b0.get_fdata(dtype=b0.header.get_data_dtype()).flatten()[mask]
    b1      = 1e-9 * b1_mag.get_fdata(dtype=b1_mag.header.get_data_dtype()) * np.exp(1j * b1_phs.get_fdata(dtype=b1_phs.header.get_data_dtype())) # nT/V -> T/V

    sz      = b1.shape[0:3]
    nRF     = kspace_traj.shape[0]    
    nPos    = mask[0].size
    T       = np.sum(rf_len) + np.sum(rf_gap)
    # calculate spatial position of voxels 
    xv, yv, zv = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]), np.arange(sz[2]), indexing='ij')
    xyz = np.column_stack((xv.flatten()[mask], yv.flatten()[mask], zv.flatten()[mask])) # we must transform RAS to gradient coordinate !!!!!!!
    xyz = xyz * zooms + offset  # in Gradient Coordinate System (a.k.a. PRS) 

    A = np.zeros((nPos, nRF), dtype=b1.dtype)
    for i, traj in enumerate(kspace_traj):
        t = np.sum(rf_len[:i]) + np.sum(rf_gap[:i]) + rf_len[i]/2 # time at the center of the RF
        kr = np.squeeze(xyz @ traj.T) # Mx3 @ 3xN -> MxN
        A[:,i] = 1j*gamma * rf_len[i] * np.exp(1j*2*np.pi * (b0*(t-T) + kr))

    A_pTx = list()
    for b1_cha in np.moveaxis(b1, -1, 0):
        A_pTx.append(A * b1_cha.flatten()[mask].reshape(-1, 1))

    return np.concatenate(A_pTx, axis=1)


