"""Separate acceptance-mask restriction from cached teacher/reference flow choice."""
import hashlib

import numpy as np

from aligned_raw_history_r11 import align_history

REGIMES=('current','teacher_default','teacher_common','reference_common')


def array_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def prepare(data):
    low=data['low'];n,h,w,c=low.shape;assert n==16 and c==3
    for key in ['half_flow','reference_flow']:assert data[key].shape==(n-1,h,w,2) and np.isfinite(data[key]).all()
    for key in ['half_mask','reference_mask']:assert data[key].shape==(n-1,h,w)
    histories={r:np.empty_like(low,dtype=np.float32) for r in REGIMES}
    for value in histories.values():value[0]=low[0]
    histories['current'][:]=low
    fields={r:[] for r in REGIMES if r!='current'};records=[]
    for t in range(1,n):
        teacher,valid_teacher=align_history(low[t-1],low[t],data['half_flow'][t-1],data['half_mask'][t-1])
        _,valid_reference=align_history(low[t-1],low[t],data['reference_flow'][t-1],data['reference_mask'][t-1])
        common=valid_teacher&valid_reference
        matched_teacher,mt=align_history(low[t-1],low[t],data['half_flow'][t-1],common)
        matched_reference,mr=align_history(low[t-1],low[t],data['reference_flow'][t-1],common)
        assert np.array_equal(mt,mr) and np.array_equal(mt,common)
        histories['teacher_default'][t]=teacher;histories['teacher_common'][t]=matched_teacher;histories['reference_common'][t]=matched_reference
        fields['teacher_default'].append((data['half_flow'][t-1],valid_teacher))
        fields['teacher_common'].append((data['half_flow'][t-1],common))
        fields['reference_common'].append((data['reference_flow'][t-1],common))
        difference=np.linalg.norm(data['half_flow'][t-1]-data['reference_flow'][t-1],axis=-1)
        records.append({'frame_pair':[t-1,t],'teacher_valid_fraction':float(valid_teacher.mean()),
            'reference_valid_fraction':float(valid_reference.mean()),'common_valid_fraction':float(common.mean()),
            'flow_difference_mean_common_pixels':float(difference[common].mean()) if common.any() else None,
            'flow_difference_p95_common_pixels':float(np.percentile(difference[common],95)) if common.any() else None,
            'aligned_input_MAE255_common':float(np.abs(matched_teacher-matched_reference)[common].mean()) if common.any() else None,
            'history_array_sha256':{r:array_sha(histories[r][t]) for r in REGIMES},
            'all_common_accepted_pixels_equal':bool(np.array_equal(mt,mr))})
    return histories,fields,records
