"""CPU functional counterexample only; no trained enhancement model, video pixels or SGD."""
import argparse
import os
from pathlib import Path
import time

from causal_reference_io_r75 import ROOT,BASE,write,stamp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or args.out.exists():
        raise SystemExit('Require initially empty CUDA mask and a new result directory.')
    if not args.out.resolve().is_relative_to(ROOT):
        raise SystemExit('Result path must stay in the project.')
    args.out.mkdir(parents=True,exist_ok=False)
    import torch
    from current_refinement_r76 import refine_current
    from rebotnet_reference import native_pair,tile_layout
    torch.set_num_threads(2);started=time.perf_counter()

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.calls=0
        def forward(self,pair):
            self.calls+=1
            return .5*pair[:,0]+2*pair[:,1]

    x=torch.full((1,3,17,23),.04)
    model=Toy();self4=refine_current(model,x)
    assert model.calls==4*len(tile_layout(17,23))
    expected=(4-3/16)*x
    error=float((self4-expected).abs().max())
    assert error<=1e-6
    history=x;trajectory=[]
    for _ in range(4):
        history=native_pair(Toy(),history,x)
        trajectory.append(float(history.mean()))
    repeated_error=float((history-self4).abs().max())
    assert repeated_error<=1e-6
    current=native_pair(Toy(),x,x)
    b=torch.full_like(x,.09)
    after_a=refine_current(model,b)
    fresh_b=refine_current(Toy(),b)
    reset_error=float((after_a-fresh_b).abs().max())
    assert reset_error<=1e-6 and not torch.cuda.is_initialized()
    write(args.out/'gate.json',dict(at=stamp(),passed=True,distinct_input_values_in_trajectory=1,
        scalar_input=.04,raw_current_mean=float(current.mean()),causal_same_input_trajectory=trajectory,
        self4_mean=float(self4.mean()),closed_form_max_error=error,
        repeated_identical_frame_equivalence_error=repeated_error,
        across_real_frame_reset_error=reset_error,first_self4_toy_calls=4,
        new_trained_enhancement_CNN=0,SGD=0,new_video_pixels=0,CUDA_initialized=False,
        elapsed_seconds=time.perf_counter()-started,
        limits='Toy arithmetic and wrapper contract only; no real-model quality, novelty or general causal attribution established.'))
    print((args.out/'gate.json').read_text(),flush=True)


if __name__=='__main__':
    main()
