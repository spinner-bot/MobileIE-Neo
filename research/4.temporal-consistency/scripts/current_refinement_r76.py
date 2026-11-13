"""Fixed four complete same-frame passes, no state crossing real frame boundaries."""
from rebotnet_reference import native_pair

PASSES=4


def refine_current(model,current):
    history=current
    for _ in range(PASSES):
        history=native_pair(model,history,current)
    return history
