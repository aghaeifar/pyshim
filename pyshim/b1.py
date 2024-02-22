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


def create_system_matrix(b1_mag_nii, b1_phs_nii, b0_nii, mask_nii=None, kspace_traj:np.ndarray=np.array([[0.,0.,0.]]), rf_len:np.ndarray=np.array([1e-3]), rf_gap:np.ndarray=np.array([0.])):
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
    b1_mag_nib  = nib.load(b1_mag_nii) # 
    b1_phs_nib  = nib.load(b1_phs_nii)
    b0_nib      = nib.load(b0_nii)
    mask_nib    = nib.load(mask_nii) if mask_nii is not None else nib.Nifti1Image(np.ones(b0_nib.shape, dtype=np.uint8), b0_nib.affine)
    # check consistency of the input data
    if not all_equal(b1_mag_nib.header.get_data_shape()[:-1], b1_phs_nib.header.get_data_shape()[:-1], b0_nib.header.get_data_shape(), mask_nib.header.get_data_shape()):
        raise ValueError('b1_mag, b1_phs, b0 and mask must have the same shape')
    if not all_equal(b1_mag_nib.affine, b1_phs_nib.affine, b0_nib.affine, mask_nib.affine):
        raise ValueError('b1_mag, b1_phs, b0 and mask must have the same affine')
    if not all_equal(kspace_traj.shape[0], rf_len.shape[0], rf_gap.shape[0]) or kspace_traj.shape[1] != 3:
        raise ValueError(f'kspace_traj {kspace_traj.shape}, rf_len {rf_len.shape} and rf_gap {rf_gap.shape} must have the same length')
    
    # read data
    mask    = np.nonzero(mask_nib.get_fdata(dtype=mask_nib.header.get_data_dtype()).flatten())
    b0      = b0_nib.get_fdata(dtype=b0_nib.header.get_data_dtype()).flatten()[mask]
    b1      = 1e-9 * b1_mag_nib.get_fdata(dtype=b1_mag_nib.header.get_data_dtype()) * np.exp(1j * b1_phs_nib.get_fdata(dtype=b1_phs_nib.header.get_data_dtype())) # nT/V -> T/V
    # constant values
    sz      = np.array(b0_nib.shape)
    affine  = b0_nib.affine
    RAS2GR  = np.diag([-1, 1, -1])  # RAS coordinate to gradient coordinate system
    nRF     = kspace_traj.shape[0]  # number of sub-pulses    
    nPos    = mask[0].size          # number of spatial points
    T       = np.sum(rf_len) + np.sum(rf_gap) # Total duration of the RF pulse
    # calculate spatial position of voxels in Gx, Gy, Gz coordinates, a.k.a Device Coordinate System (DCS) in Siemens scanners
    xv, yv, zv = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]), np.arange(sz[2]), indexing='ij')
    xyz = np.column_stack((xv.flatten()[mask], yv.flatten()[mask], zv.flatten()[mask])) # we must transform RAS to gradient coordinate !!!!!!!
    xyz = RAS2GR @ apply_affine(affine, xyz).T # 3xM

    A = np.zeros((nPos, nRF), dtype=b1.dtype)
    for i, traj in enumerate(kspace_traj):
        t = np.sum(rf_len[:i]) + np.sum(rf_gap[:i]) + rf_len[i]/2 # time at the center of the RF
        kr = np.squeeze(traj @ xyz) # Nx3 @ 3xM -> NxM 
        A[:,i] = 1j*gamma * rf_len[i] * np.exp(1j*2*np.pi * (b0*(t-T) + kr.T))

    A_pTx = list()
    for b1_cha in np.moveaxis(b1, -1, 0):
        A_pTx.append(A * b1_cha.flatten()[mask].reshape(-1, 1))

    return np.concatenate(A_pTx, axis=1)
