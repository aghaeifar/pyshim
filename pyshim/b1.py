import os
import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine

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
        b1_mag_nii:    B1 magnitude in [nT/V], shape (x,y,z,coil), [nifti file or object]
        b1_phs_nii:    B1 phase in [rad], shape (x,y,z,coil), [nifti file or object]
        b0_nii:        B0 inhomogeneity, [Hz], shape (x,y,z), [nifti file or object]
        mask_nii:      mask, shape (x,y,z), [nifti file or object]
        kspace_traj:   k-space trajectory, shape (num of RFs, 3 [xk,yk,zk])
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
    mask    = np.nonzero(mask_nii.get_fdata(dtype=mask_nii.get_data_dtype()).flatten())
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
        A[:,i] = 1j*gamma * rf_len[i] * np.exp(1j*2*np.pi * (b0*(t-T) + kr.T))

    A_pTx = [A * np.expand_dims(b1_cha, -1) for b1_cha in np.moveaxis(b1, -1, 0)]

    return np.concatenate(A_pTx, axis=1)


def calculate_SAR(voltages:np.ndarray, VOPs:np.ndarray, rf_len, TR):
    '''
    voltages : nTx x nRF
    VOPs: nTx x nTx x nQmat
    '''
    if all_equal(voltages.shape[0], VOPs.shape[0],  VOPs.shape[1]) is False or voltages.shape[1] != len(rf_len):
        raise ValueError('nTx of voltages and VOPs must be the same or nRF of voltages must be the same as rf_len')

    print(f'Number of VOP Q-Matrices: {VOPs.shape[2]}')
    # loop over RF pulses and calculate sum SED
    SED = np.zeros(VOPs.shape[2])
    for rf_ptx, l in zip(voltages.T, rf_len):
        rf_ptx = rf_ptx[:, np.newaxis] # nTx x 1
        pRF = rf_ptx * rf_ptx.conj().T  # nTx x nTx
        pRF = np.expand_dims(pRF, -1) # nTx x nTx x 1
        SED = SED + np.sum(pRF * VOPs, axis=(0,1)).squeeze() * l #  nQmat
    localSARmax = np.max(np.abs(SED)) / TR # W/kg/sec
    return localSARmax


def create_cp_txscalefactor(nTx):
    import math
    txscalefactor = np.ones(nTx) / math.sqrt(nTx) * np.exp(1j*np.linspace(0, 2*np.pi, num=nTx, endpoint=False))
    return txscalefactor.astype(np.complex64)


def calculate_transmit_voltage(ref_transmit_voltage, flipangle, normalized_pulse_integral=1e-3):
    return flipangle * ref_transmit_voltage * 1e-3 / normalized_pulse_integral / 180


def combine_pTx(b1_nii:nib.Nifti1Image, voltage:np.ndarray, is_nTv=False, keep_4th_dim=False):
    '''
    b1_nii:   B1 complex in [xT/V], shape (...,coil), [nifti object]
    voltage:  Transmit voltage in [V], shape (nTx, nRF)

    return:   combined B1 in [xT], shape (...), [nifti object]
    '''
    if voltage.shape[0] != b1_nii.shape[-1]:
        raise ValueError('voltage.shape[0] must be equal to b1_nii.shape[-1]')
    
    b1 = b1_nii.get_fdata(dtype=b1_nii.get_data_dtype()) # nT/V -> T/V
    b1 = b1*1e-9 if is_nTv else b1 # nT/V -> T/V
    voltage = np.expand_dims(voltage, -1) if voltage.ndim==1 else voltage # nTx -> nTx x 1
    combined_b1 = np.einsum('ij,...i->...j', voltage, b1)
    combined_b1 = np.squeeze(combined_b1, axis=-1) if keep_4th_dim==False else combined_b1 # remove the 4th dimension if it is 1
    return nib.Nifti1Image(combined_b1, b1_nii.affine)



def bloch_simulation(b1_nii:nib.Nifti1Image, b1volt:np.ndarray, gr_mom:np.ndarray=None, b0_nii:nib.Nifti1Image=None, mask_nii:nib.Nifti1Image=None, rf_len:list=[1e-3], rf_gap:list=[]):
    '''
    b1map  : (nT/v)         [X,Y,Z]
    b1volt : (V)            nTx x nRF
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
    if b1_nii.shape[-1]!=b1volt.shape[0] or all_equal(b1volt.shape[1], len(rf_len), len(rf_gap)+1, gr_mom.shape[1]+1)==False:
        raise ValueError(f'b1_nii.shape[-1]={b1_nii.shape[-1]} must be equal to b1volt.shape[0]={b1volt.shape[0]} and b1volt.shape[1]={b1volt.shape[1]} must be equal to len(rf_len)={len(rf_len)} and len(rf_gap)+1={len(rf_gap)+1} and gr_mom.shape[1]+1={gr_mom.shape[1]+1}')
    

    shape   = b0_nii.shape
    mask    = np.nonzero(mask_nii.get_fdata(dtype=mask_nii.get_data_dtype()).flatten())
    b0      = b0_nii.get_fdata(dtype=b0_nii.get_data_dtype()).flatten()[mask] * 2*np.pi / gamma # Hz -> Tesla
    # combine Tx
    b1_comb_nii = combine_pTx(b1_nii, b1volt, is_nTv=True, keep_4th_dim=True) # T/V, shape (...,nRF)
    # apply mask to combined B1
    b1_comb = b1_comb_nii.get_fdata(dtype=b1_comb_nii.get_data_dtype()).reshape(-1, b1_comb_nii.shape[-1])[mask] # shape (nPos, nRF)

    # calculate spatial position of voxels in Gx, Gy, Gz coordinates, a.k.a Device Coordinate System (DCS) in Siemens scanners
    RAS2GR  = np.diag([-1, 1, -1])  # RAS coordinate to gradient coordinate system
    xv, yv, zv = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    xyz = np.column_stack((xv.flatten()[mask], yv.flatten()[mask], zv.flatten()[mask])) # we must transform RAS to gradient coordinate !!!!!!!
    xyz = RAS2GR @ apply_affine(b0_nii.affine, xyz).T # 3xM

    #
    # reshape to fit into the bloch simulator
    #
    dt = rf_len + rf_gap
    # b1_comb: (nPos, nRF) -> b1:(nRF + nGap, nPos)
    b1 = np.zeros((len(dt), b1_comb.shape[0]), dtype=np.complex64)
    b1[0::2] = b1_comb.T
    # gr: (3, nGap) -> (3, nRF + nGap)
    gr = np.zeros((3, len(dt)), dtype=np.float32)
    if len(rf_gap) > 0:
        gr[1::2] = gr_mom / rf_gap # gradient momentum (T*sec/m) to amplitude (T/m)

    # simulate 
    m1 = bloch.simulate(b1, gr, xyz, b0, dt=dt)
    FA = np.zeros(shape)
    FA.ravel()[mask] = np.degrees(np.arccos(m1[2,:]))
    return nib.Nifti1Image(FA, b1_nii.affine) 


