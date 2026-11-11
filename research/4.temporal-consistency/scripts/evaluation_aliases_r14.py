"""Exact reset invariants, separate from model and output-fusion resets."""


def cache_keys(variant,history,reset,has_events,fusion):
    if variant not in ['plain','copy','aligned_concat'] or history not in ['causal','current'] or fusion not in ['raw','half01']:
        raise ValueError('registered evaluation condition required')
    model_reset=bool(reset and has_events and variant=='aligned_concat' and history=='causal')
    output_reset=bool(reset and has_events and fusion=='half01')
    return model_reset,(model_reset,fusion,output_reset)
