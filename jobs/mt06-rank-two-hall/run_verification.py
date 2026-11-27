"""Run the finite reproducibility gates and record exact source hashes."""
import hashlib,json,platform
from pathlib import Path
import numpy as np
import sympy
import symbolic_checks
import known_rank_three_certificate
import multiway_audit
from verify_hall import audit,BASELINE

def main():
    out=Path('results');out.mkdir(exist_ok=True)
    reports={
        'symbolic':symbolic_checks.run(),
        'multiway_exact':multiway_audit.exact_case(),
        'multiway_numerical':multiway_audit.floating_cases(),
        'sharp_family':multiway_audit.sharp_family_exact(),
        'finite_volume_equality':multiway_audit.finite_volume_family_exact(),
        'known_rank_three':known_rank_three_certificate.run(),
        'published_baseline':audit(BASELINE),
    }
    assert abs(reports['published_baseline']['hall_A33']+51.44435697671941)<1e-9
    assert reports['published_baseline']['min_conductivity_eigenvalue']>0
    assert reports['published_baseline']['antisymmetry_residual']<1e-8
    errors=[x['error'] for x in reports['published_baseline']['derivative_convergence']]
    assert errors[1]<errors[0]/50 and errors[2]<errors[1]/50 and errors[3]<errors[2]/50
    reports['environment']={'python':platform.python_version(),'numpy':np.__version__,'sympy':sympy.__version__}
    reports['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(Path(__file__).parent.glob('*.py'))}
    reports['status']='PASS'
    reports['scope']='Analytic proof plus independent finite checks. CI does not replace the written infinite-family proof or peer review.'
    (out/'verification.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('PASS: exact identities, arbitrary multiway interface audit, sharp-family checks, published rank-three witness')
    print('120 seeded multiway cases; 20 exact finite-volume equalities; 16 dilute-family cases; one exact four-child identity')
    print('Published baseline A33:',reports['published_baseline']['hall_A33'])
    print('Maximum scaled identity error:',reports['multiway_numerical']['worst_errors']['scaled'])
    print('Result:',out/'verification.json')

if __name__=='__main__':main()
