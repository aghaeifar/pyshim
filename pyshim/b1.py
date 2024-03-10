import os
import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from nilearn.image import math_img

gamma = 267515315.1 # rad/sec/T

def all_equal(*iterable):
    g = iter(iterable)
    try:
        first = next(g)
    except StopIteration:
        return True
    return all(np.array_equal(first, rest) for rest in g)


def create_system_matrix(b1_nii:nib.Nifti1Image, b0_nii:nib.Nifti1Image, mask_nii:nib.Nifti1Image, kspace_traj:np.ndarray=np.array([[0.,0.,0.]]), rf_len:list=[1e-3], rf_gap:list=[]):
    '''
        b1_nii:        B1 transmit field, [nT/V], shape (x,y,z,coil), [nifti object]
        b0_nii:        B0 inhomogeneity, [Hz], shape (x,y,z), [nifti object]
        mask_nii:      mask, shape (x,y,z), [nifti object]
        kspace_traj:   excitation k-space trajectory, [rad.], shape (num of RFs, 3 [xk,yk,zk])
        rf_len:        RF length, [second], shape (num of RFs)
        rf_gap:        RF gap between consecutive RFs, [second], shape (num of RFs - 1)

        output:        A_pTx, shape (num of spatial points, num of RFs * num of Tx) -> col: [RF1_Tx1, RF2_Tx1, ..., RFN_Tx1, RF1_Tx2, RF2_Tx2, ..., RFN_Tx2, ...]
    '''
    rf_gap = [0] + rf_gap # add 0 to the beginning of the list to make the length of rf_gap the same as rf_len
    # check consistency of the input data
    if not all_equal(b1_nii.shape[:-1], b0_nii.shape, mask_nii.shape):
        raise ValueError('b1, b0 and mask must have the same shape')
    if not all_equal(b1_nii.affine, b0_nii.affine, mask_nii.affine):
        raise ValueError('b1, b0 and mask must have the same affine')
    if not all_equal(kspace_traj.shape[0], len(rf_len), len(rf_gap)) or kspace_traj.shape[1] != 3 or len(b1_nii.shape) != 4:
        raise ValueError(f'kspace_traj {kspace_traj.shape}, rf_len {len(rf_len)} and rf_gap {len(rf_gap)} must have the same length or kspace_traj must have 3 columns and b1 must have 4 dimensions (x,y,z,coil)')
    
    # read data
    mask    = np.nonzero(mask_nii.get_fdata().flatten())
    b0      = b0_nii.get_fdata(dtype=b0_nii.get_data_dtype()).flatten()[mask]
    b1      = 1e-9 * b1_nii.get_fdata(dtype=b1_nii.get_data_dtype()).reshape(-1, b1_nii.shape[-1])[mask] # nT/V -> T/V
    # constant values
    sz      = np.array(b0_nii.shape)
    affine  = b0_nii.affine
    RAS2GR  = np.diag([-1, 1, -1])  # RAS coordinate to gradient coordinate system
    nRF     = kspace_traj.shape[0]  # number of sub-pulses    
    nTx     = b1_nii.shape[-1]      # number of transmit channels
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
        A[:,i] = 1j*gamma * rf_len[i] * np.exp(1j*2*np.pi*b0*(t-T) + 1j*kr.T)

    A_pTx = [A * np.expand_dims(b1_cha, -1) for b1_cha in np.moveaxis(b1, -1, 0)]

    return np.concatenate(A_pTx, axis=1)


def system_matrix_to_mxy(A:np.ndarray, voltages:np.ndarray, mask_nii:nib.Nifti1Image):
    '''
    A:          system matrix, shape (num of spatial points, num of RFs * num of Tx) -> col: [RF1_Tx1, RF2_Tx1, ..., RFN_Tx1, RF1_Tx2, RF2_Tx2, ..., RFN_Tx2, ...]
    voltages:   transmit voltage, shape (nTx, nRF)
    mask_nii:   mask, shape (x,y,z), [nifti object]

    return:     mxy, shape (x,y,z), [nifti object]
    '''
    if A.shape[1] != voltages.size:
        raise ValueError(f'A.shape[1]={A.shape[1]} must be equal to voltages.size={voltages.size}')

    mask= np.nonzero(mask_nii.get_fdata().flatten())
    if A.shape[0] != mask[0].size:
        raise ValueError(f'A.shape[0]={A.shape[0]} must be equal to the number of non-zero elements in the mask={mask[0].size}')
    
    mxy = np.zeros(mask_nii.shape, dtype=np.complex64)
    mxy.ravel()[mask] = A @ voltages.flatten()
    return nib.Nifti1Image(mxy, mask_nii.affine)


def mxy_to_fa(mxy_nii:nib.Nifti1Image):
    '''
    mxy_nii:    mxy, shape (x,y,z), [nifti object]

    return:     fa, shape (x,y,z), [nifti object]
    '''
    mxy = np.abs(mxy_nii.get_fdata(dtype=mxy_nii.get_data_dtype()))
    fa  = np.arcsin(mxy, out=np.zeros(mxy.shape), where=~(np.isnan(mxy) | np.isinf(mxy) | (mxy>1) | (mxy <-1)) )
    fa  = np.rad2deg(fa)
    return nib.Nifti1Image(fa, mxy_nii.affine)


def calculate_SAR(voltages:np.ndarray, VOPs:np.ndarray, rf_len, TR):
    '''
    voltages : nTx x nRF
    VOPs: nTx x nTx x nQmat
    '''
    if all_equal(voltages.shape[0], VOPs.shape[0],  VOPs.shape[1]) is False or voltages.shape[1] != len(rf_len):
        raise ValueError('nTx of voltages and VOPs must be the same or nRF of voltages must be the same as rf_len')

    # print(f'Number of VOP Q-Matrices: {VOPs.shape[2]}')
    # loop over RF pulses and calculate SED for each Q-Matrix
    SED = [np.einsum('i,ijk,j->k', rf_ptx, VOPs, rf_ptx.conj())*l for rf_ptx, l in zip(voltages.T, rf_len)]
    localSARmax = np.max(np.abs(np.sum(SED, axis=0))) / TR # W/kg/sec
    return localSARmax


def create_cp_txscalefactor(nTx):
    import math
    txscalefactor = np.ones(nTx) / math.sqrt(nTx) * np.exp(1j*np.linspace(0, 2*np.pi, num=nTx, endpoint=False))
    return txscalefactor.astype(np.complex64)


def calculate_transmit_voltage(ref_transmit_voltage, flipangle, normalized_pulse_integral=1e-3):
    return flipangle * ref_transmit_voltage * 1e-3 / normalized_pulse_integral / 180


def G2K(gradient:np.ndarray, dt:np.ndarray):
    '''
    gradient: [T/m]   (num of RFs, 3 [x,y,z])
    dt:       [sec.]  (num of RFs, 1)
    '''
    dt = dt.flatten()[:,np.newaxis] # always Nx1
    if gradient.shape[0] != dt.size or gradient.shape[1] != 3:
        raise ValueError(f'gradient shape {gradient.shape} and/or dt shape {dt.shape} is incorrect')
    k = -gamma * gradient * dt
    k = np.cumsum(k[::-1], axis=0)[::-1]
    return k


def K2G(k:np.ndarray, dt:np.ndarray):
    '''
    k:        [rad/m] (num of RFs, 3 [x,y,z])
    dt:       [sec.]  (num of RFs, 1)
    '''
    dt = dt.flatten()[:,np.newaxis] # always Nx1
    if k.shape[0] != dt.size or k.shape[1] != 3:
        raise ValueError(f'k shape {k.shape} and/or dt shape {dt.shape} is incorrect')
    gradient = -np.diff(k[::-1], prepend=0)[::-1] 
    gradient = gradient / (gamma * dt)
    return gradient



def combine_pTx(b1_nii:nib.Nifti1Image, voltage:np.ndarray, is_nTv=False, keep_4th_dim=False):
    '''
    b1_nii:         B1 complex in [xT/V], shape (...,coil), [nifti object]
    voltage:        Transmit voltage in [V], shape (nTx, nRF)
    rf_len:         RF length, [sec.], shape (num of RFs)
    kspace_traj:    k-space trajectory, shape (num of RFs, 3 [xk,yk,zk])

    return:   combined B1 in [xT], shape (...), [nifti object]
    '''
    voltage = np.expand_dims(voltage, -1) if voltage.ndim==1 else voltage # nTx -> nTx x 1
    if voltage.shape[0] != b1_nii.shape[-1]:
        raise ValueError('voltage and b1 must have the same number of transmit channels')
    
    b1 = b1_nii.get_fdata(dtype=b1_nii.get_data_dtype()) # nT/V -> T/V
    b1 = b1*1e-9 if is_nTv else b1 # nT/V -> T/V

    combined_b1 = np.einsum('ij,...i->...j', voltage, b1)
    combined_b1 = np.squeeze(combined_b1, axis=-1) if keep_4th_dim==False else combined_b1 # remove the 4th dimension if it is 1
    return nib.Nifti1Image(combined_b1, b1_nii.affine)


def bloch_simulation(b1_nii:nib.Nifti1Image, gr_mom:np.ndarray=None, b0_nii:nib.Nifti1Image=None, mask_nii:nib.Nifti1Image=None, rf_len:list=[1e-3], rf_gap:list=[]):
    '''
    b1map  : (T)            [X,Y,Z, nRF]
    gr_mom : (Tesla.sec/m)  3 x (nRF-1)  
    b0     : (Hz)           [X,Y,Z] x 1
    rf_len : (sec)          nRF x 1
    rf_gap : (sec)          (nRF-1) x 1
    '''
    import pytools.pyTurboBloch as bloch 

    if gr_mom is None:
        gr_mom = np.zeros((3, len(rf_gap)))
    if b0_nii is None:
        b0_nii = nib.Nifti1Image(np.zeros(b1_nii.shape[:-1]), b1_nii.affine)
    if mask_nii is None:
        mask_nii = nib.Nifti1Image(np.ones(b1_nii.shape[:-1]), b1_nii.affine)
    # check consistency of the input data
    if not all_equal(b1_nii.shape[:-1], b0_nii.shape, mask_nii.shape):
        raise ValueError(f'b1_nii {b1_nii.shape[:-1]}, b0_nii {b0_nii.shape}, and mask_nii {mask_nii.shape} must have the same shape')
    if not all_equal(b1_nii.affine, b0_nii.affine, mask_nii.affine):
        raise ValueError('b1_mag, b1_phs, b0 and mask must have the same affine')
    if all_equal(len(rf_len), len(rf_gap)+1, gr_mom.shape[1]+1)==False:
        raise ValueError(f'must be equal: len(rf_len)={len(rf_len)} and len(rf_gap)+1={len(rf_gap)+1} and gr_mom.shape[1]+1={gr_mom.shape[1]+1}')
    

    shape   = b0_nii.shape
    mask    = np.nonzero(mask_nii.get_fdata().flatten())
    b0      = b0_nii.get_fdata(dtype=b0_nii.get_data_dtype()).flatten()[mask] * 2*np.pi / gamma # Hz -> Tesla
    # apply mask to combined B1
    b1_data = b1_nii.get_fdata(dtype=b1_nii.get_data_dtype()).reshape(-1, b1_nii.shape[-1])[mask] # shape (nPos, nRF)

    # calculate spatial position of voxels in Gx, Gy, Gz coordinates, a.k.a Device Coordinate System (DCS) in Siemens scanners
    RAS2GR  = np.diag([-1, 1, -1])  # RAS coordinate to gradient coordinate system
    xv, yv, zv = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    xyz = np.column_stack((xv.flatten()[mask], yv.flatten()[mask], zv.flatten()[mask])) # we must transform RAS to gradient coordinate !!!!!!!
    xyz = RAS2GR @ apply_affine(b0_nii.affine, xyz).T # 3xM

    #
    # reshape to fit into the bloch simulator
    #
    dt = rf_len + rf_gap
    # b1_data: (nPos, nRF) -> b1:(nRF + nGap, nPos)
    b1 = np.zeros((len(dt), b1_data.shape[0]), dtype=np.complex64)
    b1[0::2] = b1_data.T
    # gr: (3, nGap) -> (3, nRF + nGap)
    gr = np.zeros((3, len(dt)), dtype=np.float32)
    if len(rf_gap) > 0:
        gr[1::2] = gr_mom / rf_gap # gradient momentum (T*sec/m) to amplitude (T/m)

    # simulate 
    m1 = bloch.simulate(b1, gr, xyz, b0, dt=dt)
    FA = np.zeros(shape)
    FA.ravel()[mask] = np.degrees(np.arccos(m1[2,:]))
    return nib.Nifti1Image(FA, b1_nii.affine) 


