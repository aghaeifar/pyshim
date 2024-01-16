import numpy as np
from tqdm import tqdm
from recotwix.prot_volumes import prot_volumes, volume
from recotwix.transformation import resample
from recotwix import recotwix


class shimming_base():
    _shim_volume_protocol  = None
    _shim_volume_resampled  = {'slc':volume(), 'adj':volume(), 'ptx':volume()} # just for intelicence
    _standard_space_affine = None
    _standard_space_size   = None

    def __init__(self, protocol=None, res=1.5, fov=300) -> None:
        self.calc_standard_space_affine(res=res, fov=fov)
        if protocol is not None:
            self.set_protocol(protocol) 

    def set_protocol(self, protocol):
        self._shim_volume_protocol = prot_volumes(protocol)
        self._shim_volume_resliced = dict() # clear to make it for the instance of shimming_base
        self.resample_volumes()

    def calc_standard_space_affine(self, res, fov):
        """
        Calculate the affine transformation, defining the standard space
        res: resolution in the standard space (mm)
        fov: field-of-view in the standard space (mm)
        """
        if fov<0 or res<0 or fov<res:
            raise ValueError('fov and res must be positive and fov must be larger than res')
        self._standard_space_affine = np.hstack((np.eye(4,3)*res, np.array([-fov/2,-fov/2,-fov/2,1]).reshape(4,1)))
        self._standard_space_size = [int(x) for x in [fov/res]*3]

    def resample_volumes(self):
        """
        transform all shimming volumes to the stanard space
        """
        t_affine = self._standard_space_affine
        t_size = self._standard_space_size
        with tqdm(total=self._shim_volume_protocol.num_volumes, desc='resample volumes') as pbar:
            for vol_name in self._shim_volume_protocol.get_volume_names():
                self._shim_volume_resliced[vol_name] = list()
                for vol in self._shim_volume_protocol.get(vol_name):
                    r = resample(vol.data(), vol.affine, target_affine=t_affine, target_size=t_size, interp_oder=5, fill_value=0)
                    self._shim_volume_resliced[vol_name].append(r)
                    pbar.update(1)


    def resample_image(self, image, affine):
        """
        transform image to the stanard space
        """
        t_affine = self._standard_space_affine
        t_size = self._standard_space_size
        return resample(image, affine, target_affine=t_affine, target_size=t_size, interp_oder=3, fill_value=0)
    

    def get_standard_space_affine(self):
        return self._standard_space_affine

    def get_shim_volume_data(self):
        return self._shim_volume_resliced  
    
    def get_shim_volume_protocol(self):
        return self._shim_volume_protocol


class shimming(shimming_base):
    _recotwix_obj = None
    def __init__(self, file_twix, file_protocol=None) -> None:
        super().__init__()
        self._recotwix_obj = recotwix(file_twix)
        self._recotwix_obj.runReco(method_sensitivity=None)
        if file_protocol is not None:
            self.set_protocol(file_protocol) 
        else:
            self.set_protocol(self._recotwix_obj.hdr['MeasYaps'])

    def get_recotwix(self):
        return self._recotwix_obj



def lsqlin(a:np.ndarray, b:np.ndarray, lb:np.ndarray, ub:np.ndarray):
    """
    Solving ax = b 
    subject to lb <= x <= ub
    """
    import cvxpy as cp
    x = cp.Variable(len(b))
    prob = cp.Problem(cp.Minimize(cp.sum_squares(a @ x - b)), [lb <= x, x <= ub])
    prob.solve()
    return x.value
    
