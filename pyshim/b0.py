
import torch
import numpy as np
from recotwix.sequences import recoB0
from shimming import shimming_base

class shimming_b0(shimming_base):
    _b0_obj = None

    def __init__(self, file_twix, file_protocol=None) -> None:
        self._b0_obj = recoB0(file_twix)
        if file_protocol is not None:
            self.set_shim_volume(file_protocol) 
        else:
            self.set_shim_volume(self._b0_obj.hdr['MeasYaps'])

    
    def get_data(self):
        return self._b0_obj

    def calc_shim(self, vol_name:str):
        shim_volume = self._shim_volume_obj.get(vol_name)

        pass
