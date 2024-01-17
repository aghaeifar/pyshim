
import torch
import numpy as np
from . import shimming_base

class shimming_b0(shimming_base):

    def __init__(self, file_twix, file_protocol=None) -> None:
        super().__init__()
        # if file_protocol is not None:
        #     self.set_protocol(file_protocol) 
        # else:
        #     self.set_protocol(self._b0_obj.hdr['MeasYaps'])

    
    def get_data(self):
        pass
        # return self._b0_obj


    def calc_shim(self, vol_name:str):
        pass
