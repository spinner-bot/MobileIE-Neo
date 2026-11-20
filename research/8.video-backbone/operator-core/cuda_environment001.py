"""O1 execution correction: set deterministic CuBLAS workspace before Torch imports."""
import os
import sys
import json
from study_io import verify

verify('training-source-freeze001.json')
verify('cuda-execution-freeze001.json')
assert 'torch' not in sys.modules
command = sys.argv[1:]
if command[:1] == ['--']: command = command[1:]
assert command, 'An explicit original training command is required'
environment = os.environ.copy()
assert environment.get('CUDA_VISIBLE_DEVICES') == environment.get('MOBILEIE_SHARED_GPU_UUID')
environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
print(json.dumps(dict(O1_execution_amendment=1, CUBLAS_WORKSPACE_CONFIG=':4096:8', before_Torch_import=True)), flush=True)
os.execvpe(command[0], command, environment)
