"""Toy Gaussian moment check; not evidence of a working video architecture."""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261110)
    n=200000
    rows=[]
    start=time.perf_counter()
    for u in [-1.,-.2,0.,.2,1.]:
        for sigma in [.05,.2,.5]:
            for rho in [0.,.5,1.]:
                a=rng.normal(0,sigma,n)
                b=rho*a+np.sqrt(1-rho*rho)*rng.normal(0,sigma,n)
                error=(u+a)*(u+b)-u*u
                bias=rho*sigma*sigma
                mse=2*u*u*sigma*sigma*(1+rho)+sigma**4*(1+2*rho*rho)
                measured_bias=float(error.mean())
                measured_mse=float((error**2).mean())
                se=np.sqrt((mse-bias*bias)/n)
                assert abs(measured_bias-bias)<6*se
                assert abs(measured_mse/mse-1)<.05
                rows.append({'feature':u,'sigma':sigma,'correlation':rho,'bias_theory':bias,
                             'bias_measured':measured_bias,'mse_theory':mse,'mse_measured':measured_mse})
    out.write_text(json.dumps({'seed':20261110,'samples_per_condition':n,'conditions':rows,
        'wall_seconds':time.perf_counter()-start,
        'assumptions':'joint Gaussian, equal clean signal, equal variance, zero mean; not a neural-feature or real-video test'},indent=2)+'\n')
    print('45 toy conditions passed; not a model result')


if __name__=='__main__':main()
