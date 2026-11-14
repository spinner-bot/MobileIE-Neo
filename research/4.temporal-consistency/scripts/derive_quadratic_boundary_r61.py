"""Exact constant-coefficient scalar boundary; no images, sampling or neural calls."""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import sympy as S

BASE = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main():
    out = BASE/'results/exploration-r61-quadratic-constraints001'; out.mkdir(exist_ok=False)
    start, complete = time.perf_counter(), False
    try:
        u, v, s, a, b, g, mu, mt, mp, rho, z, w = S.symbols('u v s a b g mu mt mp rho z w', real=True)
        sigma = S.Symbol('sigma', positive=True)
        A, B, C, D, E, H = S.symbols('A B C D E H', real=True)
        general = A*u**2+B*u*v+C*v**2+D*u+E*v+H
        equations = S.Poly(general.subs({u:s, v:s})-a*s**2-b, s).all_coeffs()
        equations += S.Poly(S.diff(general, v).subs({u:s, v:s}), s).all_coeffs()
        solution = S.solve(equations, [A, B, C, D, E, H], dict=True)
        assert len(solution) == 1
        family = S.expand(general.subs(solution[0]).subs(C, -g))
        expected_family = a*u**2-g*(u-v)**2+b
        checks = {}
        def check(name, expression):
            residual = S.simplify(expression); assert residual == 0, (name, residual)
            checks[name] = str(residual)
        check('family', family-expected_family)
        check('diagonal_consistency', family.subs({u:s, v:s})-a*s**2-b)
        check('no_history_first_derivative_at_equal_means', S.diff(family, v).subs({u:s, v:s}))
        moments = {(0,0):S.Integer(1), (2,0):sigma**2, (0,2):sigma**2,
                   (1,1):rho*sigma**2, (4,0):3*sigma**4, (0,4):3*sigma**4,
                   (3,1):3*rho*sigma**4, (1,3):3*rho*sigma**4,
                   (2,2):(1+2*rho**2)*sigma**4}
        def expectation(expression):
            result = S.Integer(0)
            for powers, coefficient in S.Poly(S.expand(expression), z, w).terms():
                assert sum(powers) <= 4
                moment = S.Integer(0) if sum(powers) % 2 else moments[powers]
                result += coefficient*moment
            return S.expand(result)
        noisy = family.subs({u:mu+z, v:mu+w})
        mean = expectation(noisy)
        variance = S.expand(expectation(noisy**2)-mean**2)
        bias = S.expand(mean-(a*mu**2+b))
        independent_bias = bias.subs(rho, 0)
        independent_variance = variance.subs(rho, 0)
        predicted_variance = 4*a**2*mu**2*sigma**2+(2*a**2-4*a*g+8*g**2)*sigma**4
        check('independent_bias', independent_bias-(a-2*g)*sigma**2)
        check('independent_variance', independent_variance-predicted_variance)
        check('linear_variance_coefficient_for_all_rho', S.Poly(variance, mu).coeff_monomial(mu**2)-4*a**2*sigma**2)
        check('correlated_bias', bias-(a-2*g*(1-rho))*sigma**2)
        drift = S.expand(expectation(family.subs({u:mt+z, v:mp+w})).subs(rho, 0)-(a*mt**2+b))
        check('deterministic_drift_penalty', drift-((a-2*g)*sigma**2-g*(mt-mp)**2))
        raw_variance = independent_variance.subs(g, 0)
        check('half_correction_zero_independent_bias', independent_bias.subs(g, a/2))
        check('half_correction_same_gaussian_variance_as_raw', independent_variance.subs(g, a/2)-raw_variance)
        mse = S.expand(independent_variance+independent_bias**2)
        check('independent_mse', mse-(4*a**2*mu**2*sigma**2+(3*a**2-8*a*g+12*g**2)*sigma**4))
        check('mse_stationary_coefficient', S.diff(mse, g).subs(g, a/3))
        protocol = BASE/'docs/exploration-r61-two-observation-quadratic-boundary-protocol.md'
        result = dict(at=datetime.now().astimezone().isoformat(), passed=True, sympy=S.__version__,
              constraints_solution={str(k):str(v) for k,v in solution[0].items()}, family=str(family),
              independent_bias=str(independent_bias), independent_variance=str(independent_variance),
              correlated_bias=str(bias), correlated_variance=str(variance), drift_bias=str(drift), independent_MSE=str(mse),
              exact_zero_residuals=checks, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
              new_CNN=0, SGD=0, image_draws=0, Monte_Carlo_draws=0, GPU=False,
              scope='Elementary scalar constant-coefficient Gaussian calculation, not a full-network bound or new theory/method.')
        write(out/'derivation.json', result); print(json.dumps(result, ensure_ascii=False))
        complete = True
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat(), new_CNN=0, SGD=0, GPU=False))


if __name__ == '__main__': main()
