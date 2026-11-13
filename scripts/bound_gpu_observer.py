"""Read-only NVML polling of one physical UUID, without CUDA or CLI subprocesses."""
import ctypes as C
import ctypes.util
from datetime import datetime


class Memory(C.Structure):
    _fields_ = [('total', C.c_ulonglong), ('free', C.c_ulonglong), ('used', C.c_ulonglong)]


class Utilization(C.Structure):
    _fields_ = [('gpu', C.c_uint), ('memory', C.c_uint)]


class BoundGPUObserver:
    def __init__(self, uuid):
        self.uuid = uuid
        self.lib = C.CDLL(ctypes.util.find_library('nvidia-ml') or 'libnvidia-ml.so.1')
        self.lib.nvmlInit_v2.argtypes = []
        self.lib.nvmlInit_v2.restype = C.c_int
        self.lib.nvmlDeviceGetHandleByUUID.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        self.lib.nvmlDeviceGetHandleByUUID.restype = C.c_int
        self.lib.nvmlDeviceGetMemoryInfo.argtypes = [C.c_void_p, C.POINTER(Memory)]
        self.lib.nvmlDeviceGetMemoryInfo.restype = C.c_int
        self.lib.nvmlDeviceGetUtilizationRates.argtypes = [C.c_void_p, C.POINTER(Utilization)]
        self.lib.nvmlDeviceGetUtilizationRates.restype = C.c_int
        self.lib.nvmlErrorString.argtypes = [C.c_int]
        self.lib.nvmlErrorString.restype = C.c_char_p
        self._check(self.lib.nvmlInit_v2())
        self.handle = C.c_void_p()
        self._check(self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(self.handle)))

    def _check(self, code):
        if code != 0:
            message = self.lib.nvmlErrorString(code).decode(errors='replace')
            raise RuntimeError(f'NVML read failed ({code}): {message}')

    def read(self):
        memory, utilization = Memory(), Utilization()
        self._check(self.lib.nvmlDeviceGetMemoryInfo(self.handle, C.byref(memory)))
        self._check(self.lib.nvmlDeviceGetUtilizationRates(self.handle, C.byref(utilization)))
        return dict(at=datetime.now().astimezone().isoformat(), uuid=self.uuid,
                    total_mib=memory.total // 1048576, used_mib=memory.used // 1048576,
                    free_mib=memory.free // 1048576, utilization_percent=int(utilization.gpu))
