"""Exact algebra checks supporting the written proof, not a formal proof system."""
import json
from pathlib import Path
import sympy as sp

def run():
    r,p,eps=sp.symbols('r p epsilon', positive=True)
    t=r-1; q=r-t*p; a=1+t*p; c=p/q; d=p-c
    checks={}
    def check(name,expression):
        assert sp.factor(expression)==0, (name,sp.factor(expression))
        checks[name]=True
    check('rank_one_normal_conductivity',a-t*d-r/q)
    check('rank_one_normal_KR_identity',(r/q)*p-p-t*p*c)
    check('rank_one_tangential_KR_identity',a*c-c-t*p*c)
    check('trace_bound_numerator',p+2*c+t*p*c-p*(r+2)/q)
    check('positive_denominator_remainder',(r+1)**2/4-a*q-t**2*(p-sp.Rational(1,2))**2)
    aa=(r+1)/2; hh=1-eps+eps/aa
    exact_ratio=(aa+1)/(aa**2*hh)+1/aa
    sharp=4*(r+2)/(r+1)**2
    check('dilute_sharp_limit',sp.limit(exact_ratio,eps,0,dir='+')-sharp)
    check('finite_sharp_family_positive_excess',exact_ratio-sharp-
          (aa+1)*(1-hh)/(aa**2*hh))
    # Rotation invariance permits an arbitrary top normal to be e3 here.
    x,y,z=sp.symbols('x y z')
    n=sp.Matrix([0,0,1]); b=sp.Matrix([x,y,z])
    E=sp.eye(3)+n*b.T
    check('rank_one_cofactor_identity',sum((E.cofactor_matrix().T-((1+z)*sp.eye(3)-n*b.T))))
    assert E.cofactor_matrix().T==((1+z)*sp.eye(3)-n*b.T)
    # Sharp finite-volume reduction: all denominators/signs are justified in the proof.
    Q,X,H,L,s,phi,v=sp.symbols('q x h L s phi v', positive=True)
    A=r+1-Q
    check('pointwise_interval_factorization',X*(r/Q-1)*(Q+1-(r+1)*X)-(1-X)**2
          +(r*A/Q)*(X-1/A)*(X-Q/r))
    D=s+1-(r+1)*H
    Lmin=(H+(1-H)**2/D)/r
    scalar=(r*(1/H+r+1)*Lmin+1/H-r*H-3)/t
    check('cauchy_substitution',scalar-(s+2)*(1-H)**2/(t*H*D))
    ss=r-t*phi
    denom=v*(ss+1-t*phi*v)
    check('quadratic_maximum',(ss+1)**2/(4*t*phi)-denom
          -t*phi*(v-(ss+1)/(2*t*phi))**2)
    pp=(1+phi)/2;aa=1+t*pp;qq=r-t*pp;ee=phi/pp
    check('attaining_harmonic_mean',1-ee+ee/aa-qq/aa)
    check('attaining_trace',phi*(2/qq+1/qq**2)-4*phi*(r+2-t*phi)/(r+1-t*phi)**2)
    return {'method':'exact SymPy identities', 'identities':checks,
            'claim_boundary':'The general theorem is proved in manuscript.tex; these checks audit algebra only.'}

if __name__=='__main__':
    out=run()
    Path('results').mkdir(exist_ok=True)
    Path('results/symbolic_checks.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
