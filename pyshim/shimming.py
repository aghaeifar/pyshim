from recotwix.prot_volumes import prot_volumes


class shimming_base():
    _shim_volume_obj = None

    def __init__(self, protocol) -> None:
        self._shim_volume_obj = self.set_shim_volume(protocol) 

    def set_shim_volume(self, protocol):
        self._shim_volume_obj = prot_volumes(protocol)

    def get_shim_volume(self):
        return self._shim_volume_obj