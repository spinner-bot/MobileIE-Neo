"""Finite causal aggregation, separate from the original recursive output filter."""
import numpy as np

METHODS=('raw','pre01','pre05','postwarp01','postwarp05','postaligned01','postaligned05','recursive_half01')


def blend(current,history,alpha):
    if current.shape!=history.shape or alpha not in [.1,.5]:raise ValueError('matched finite RGB histories and fixed alpha required')
    return current+alpha*(history-current)


def finite_controls(low,raw,aligned_low,aligned_prediction,warped_prediction,forward):
    if not all(a.shape==low.shape for a in [raw,aligned_low,aligned_prediction,warped_prediction]):raise ValueError('matched RGB sequences required')
    results={'raw':raw}
    for suffix,alpha in [('01',.1),('05',.5)]:
        results['pre'+suffix]=forward(blend(low,aligned_low,alpha),raw[0])
        results['postwarp'+suffix]=blend(raw,warped_prediction,alpha)
        results['postaligned'+suffix]=blend(raw,aligned_prediction,alpha)
    for name,array in results.items():
        if not np.isfinite(array).all():raise FloatingPointError('nonfinite finite aggregation')
        if name!='raw':array[0]=raw[0]
    return results


def event_output(output,raw,events):
    events=np.asarray(events,dtype=bool)
    if events.shape!=(len(output),) or output.shape!=raw.shape:raise ValueError('one reset per matching frame required')
    result=output.copy();result[events]=raw[events];return result
