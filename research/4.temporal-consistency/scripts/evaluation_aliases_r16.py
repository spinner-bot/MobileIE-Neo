"""Both raw and aligned temporal models consume input-reset events."""
from multiscale_fst_r16 import MODES


def cache_keys(variant,history,reset,has_events,fusion):
    if variant not in MODES or history not in ['causal','current'] or fusion not in ['raw','half01']:raise ValueError('unregistered evaluation')
    model_reset=bool(variant!='spatial' and history=='causal' and reset and has_events)
    output_reset=bool(reset and has_events and fusion=='half01')
    return model_reset,(model_reset,fusion,output_reset)
