"""Independent finite-field and exact audit of the multiway trace theorem.

All effective conductivities below come from 3m local interface equations;
neither the compact laminate formula nor cofactor homogenization is used.
"""
from __future__ import annotations

from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np

from verify_hall import cross, tangent_frame
from exact_verify import (add, scale, mul, solve, eye, zeros, transpose,
                          tangent_frame as exact_tangents, laminate_dual)


def multiway_interface(ks, weights, n):
    m = len(ks)
    n, t1, t2 = tangent_frame(n)
    a = np.zeros((3*m, 3*m), dtype=np.result_type(*ks))
    for i, w in enumerate(weights):
        a[:3, 3*i:3*i+3] = w*np.eye(3)
    for i in range(1, m):
        row = 3*i
        for j, tangent in enumerate([t1, t2]):
            a[row+j, :3] = -tangent
            a[row+j, 3*i:3*i+3] = tangent
        a[row+2, :3] = -n @ ks[0]
        a[row+2, 3*i:3*i+3] = n @ ks[i]
    rhs = np.zeros((3*m, 3), dtype=a.dtype)
    rhs[:3] = np.eye(3)
    es = np.linalg.solve(a, rhs)
    return sum(w*k @ es[3*i:3*i+3] for i, (w, k) in enumerate(zip(weights, ks)))


def multiway_dual_exact(children, weights, n):
    m = len(children)
    t1, t2 = exact_tangents(n)
    a0, a1 = zeros(3*m, 3*m), zeros(3*m, 3*m)
    for i, w in enumerate(weights):
        for j in range(3):
            a0[j][3*i+j] = w
    for i in range(1, m):
        row = 3*i
        for j, tangent in enumerate([t1, t2]):
            a0[row+j][:3] = [-x for x in tangent]
            a0[row+j][3*i:3*i+3] = tangent
        a0[row+2][:3] = [-x for x in mul([n], children[0][0])[0]]
        a0[row+2][3*i:3*i+3] = mul([n], children[i][0])[0]
        a1[row+2][:3] = [-x for x in mul([n], children[0][1])[0]]
        a1[row+2][3*i:3*i+3] = mul([n], children[i][1])[0]
    rhs = eye(3) + zeros(3*m-3, 3)
    e0 = solve(a0, rhs)
    e1 = solve(a0, scale(F(-1), mul(a1, e0)))
    k, derivative = zeros(3, 3), zeros(3, 3)
    for i, (weight, (ki, di)) in enumerate(zip(weights, children)):
        k = add(k, scale(weight, mul(ki, e0[3*i:3*i+3])))
        derivative = add(derivative, scale(weight, add(mul(di, e0[3*i:3*i+3]),
                                                      mul(ki, e1[3*i:3*i+3]))))
    return k, derivative


def cross_exact(v):
    x, y, z = v
    return [[F(0), -z, y], [z, F(0), -x], [-y, x, F(0)]]


def extract_vector(d):
    return [d[2][1], d[0][2], d[1][0]]


def exact_case():
    r = F(17, 2)
    h = [F(1, 5), F(2, 3), F(3, 7), F(1)]
    weights = [F(1, 10), F(1, 5), F(3, 10), F(2, 5)]
    normals = [[F(3, 5), F(4, 5), F(0)], [F(0), F(5, 13), F(12, 13)],
               [F(2, 3), F(1, 3), F(2, 3)], [F(1), F(0), F(0)]]
    top_normal = [F(6, 7), F(3, 7), F(2, 7)]
    ks, ss, total_s = [], [], zeros(3, 3)
    for j, field in enumerate(eye(3)):
        children = [laminate_dual((eye(3), zeros(3, 3)),
                                 (scale(r, eye(3)), cross_exact(field)),
                                 1-f, normal) for f, normal in zip(h, normals)]
        if j == 0:
            ks = [x[0] for x in children]
            ss = [zeros(3, 3) for _ in children]
        for i, child in enumerate(children):
            col = extract_vector(child[1])
            for row in range(3):
                ss[i][row][j] = col[row]
        _, top_derivative = multiway_dual_exact(children, weights, top_normal)
        col = extract_vector(top_derivative)
        for row in range(3):
            total_s[row][j] = col[row]
    n = top_normal
    t = r-1
    scalar_identity = []
    for k, s in zip(ks, ss):
        q = add(k, scale(t, s))
        scalar_identity.append(q == scale(q[0][0], eye(3)))
    kin = [mul(mul([n], k), transpose([n]))[0][0] for k in ks]
    harmonic = sum(w/v for w, v in zip(weights, kin))
    z = [sum(w*mul(s, transpose([n]))[a][0]/v
             for w, s, v in zip(weights, ss, kin)) for a in range(3)]
    z_n = sum(a*b for a, b in zip(z, n))
    z_t = [a-z_n*b for a, b in zip(z, n)]
    transverse = sum(w*(sum(s[i][i] for i in range(3))
                         -mul(mul([n], s), transpose([n]))[0][0])/v
                     for w, s, v in zip(weights, ss, kin))
    square = t*sum(x*x for x in z_t)
    product = harmonic*sum(w*mul(mul(mul([n], k), s), transpose([n]))[0][0]/v
                           for w, k, s, v in zip(weights, ks, ss, kin))
    lhs = harmonic*sum(total_s[i][i] for i in range(3))
    assert all(scalar_identity)
    assert lhs == transverse + square + product
    assert min(transverse, square, product) >= 0
    qs = [r-t*pi for pi in h]
    xs = [1/value for value in kin]
    mean_q = sum(w*q for w,q in zip(weights,qs))
    L = sum(w*x/q for w,x,q in zip(weights,xs,qs))
    D = mean_q+1-(r+1)*harmonic
    trace = sum(total_s[i][i] for i in range(3))
    scalar_trace = (r*(1/harmonic+r+1)*L+1/harmonic-r*harmonic-3)/t + square/harmonic
    assert trace == scalar_trace
    for q,x in zip(qs,xs):
        a = r+1-q
        gap = x*(r/q-1)*(q+1-(r+1)*x)-(1-x)**2
        factorization = -(r*a/q)*(x-1/a)*(x-q/r)
        assert gap == factorization
        assert gap >= 0
    cauchy_gap = (r*L-harmonic)*D-(1-harmonic)**2
    assert D > 0 and cauchy_gap >= 0
    phi = sum(w*pi for w,pi in zip(weights,h))
    finite_bound = 4*phi*(r+2-t*phi)/(r+1-t*phi)**2
    assert trace >= finite_bound
    return dict(children=4, contrast=str(r), exact_identity=True,
                scalar_relations_all_exact=all(scalar_identity),
                trace_S_exact=str(sum(total_s[i][i] for i in range(3))),
                trace_S=float(sum(total_s[i][i] for i in range(3))),
                decomposition_terms=[float(transverse), float(square), float(product)],
                finite_volume_scalar_trace_identity_exact=True,
                pointwise_factorizations_exact=True,
                pointwise_gaps_nonnegative=True,
                cauchy_gap_nonnegative=True,
                finite_volume_bound=float(finite_bound))


def floating_cases(count=120, seed=76403):
    rng = np.random.default_rng(seed)
    worst = dict(absolute=0.0, scaled=0.0, antisymmetry=0.0, scalar=0.0)
    min_trace = float('inf')
    min_bound_ratio = float('inf')
    min_finite_volume_ratio = float('inf')
    for trial in range(count):
        m = 2 + trial % 7
        r = 10**rng.uniform(0, 3)
        h = rng.uniform(0.001, 0.999, m)
        weights = rng.dirichlet(np.ones(m))
        normals = rng.normal(size=(m, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        ks = [multiway_interface([np.eye(3), r*np.eye(3)], [1-v, v], ni)
              for v, ni in zip(h, normals)]
        ss = [np.zeros((3, 3)) for _ in range(m)]
        total_s = np.zeros((3, 3))
        step = 1e-8
        for j in range(3):
            field = np.zeros(3, dtype=complex)
            field[j] = step*1j
            children = [multiway_interface([np.eye(3), r*np.eye(3)+cross(field)], [1-v, v], ni)
                        for v, ni in zip(h, normals)]
            for i, child in enumerate(children):
                derivative = child.imag/step
                ss[i][:, j] = extract_vector(derivative)
            derivative = multiway_interface(children, weights, n).imag/step
            total_s[:, j] = extract_vector(derivative)
            worst['antisymmetry'] = max(worst['antisymmetry'], np.max(np.abs(derivative+derivative.T)))
        kin = np.array([n @ k @ n for k in ks])
        harmonic = np.sum(weights/kin)
        z = sum(w*(s@n)/v for w, s, v in zip(weights, ss, kin))
        z_t = z-(z@n)*n
        transverse = sum(w*(np.trace(s)-n@s@n)/v for w, s, v in zip(weights, ss, kin))
        square = (r-1)*np.dot(z_t, z_t)
        product = harmonic*sum(w*(n@k@s@n)/v for w, k, s, v in zip(weights, ks, ss, kin))
        rhs = transverse+square+product
        lhs = harmonic*np.trace(total_s)
        error = abs(lhs-rhs)
        worst['absolute'] = max(worst['absolute'], error)
        worst['scaled'] = max(worst['scaled'], error/max(1.0, abs(lhs), abs(rhs)))
        for k, s in zip(ks, ss):
            q = k+(r-1)*s
            worst['scalar'] = max(worst['scalar'], np.max(abs(q-np.trace(q)*np.eye(3)/3)))
        min_trace = min(min_trace, np.trace(total_s))
        phi = np.dot(weights, h)
        bound = 4*(r+2)/(r+1)**2*phi
        finite_volume_bound = 4*phi*(r+2-(r-1)*phi)/(r+1-(r-1)*phi)**2
        min_bound_ratio = min(min_bound_ratio, float(np.trace(total_s)/bound))
        min_finite_volume_ratio = min(min_finite_volume_ratio, float(np.trace(total_s)/finite_volume_bound))
        assert np.trace(total_s) >= bound-1e-10, (trial, np.trace(total_s), bound)
        assert np.trace(total_s) >= finite_volume_bound-1e-10, (trial, np.trace(total_s), finite_volume_bound)
        assert error < 1e-10*max(1.0, abs(lhs), abs(rhs)), (trial, lhs, rhs)
        assert min(transverse, square, product) > -1e-10, trial
    return dict(seed=seed, cases=count, child_counts=[2,3,4,5,6,7,8],
                contrast_range=[1,1000], worst_errors=worst,
                minimum_observed_trace=float(min_trace),
                minimum_sharp_bound_ratio=min_bound_ratio,
                minimum_finite_volume_bound_ratio=min_finite_volume_ratio)


def sharp_family_exact():
    data = []
    for r in [F(1), F(3), F(16), F(400)]:
        p, a = F(1,2), (r+1)/2
        for epsilon in [F(1,2), F(1,10), F(1,1000), F(1,1000000)]:
            s_total = zeros(3,3)
            for j, field in enumerate(eye(3)):
                child = laminate_dual((eye(3), zeros(3,3)),
                                       (scale(r,eye(3)), cross_exact(field)),
                                       1-p, [F(1), F(0), F(0)])
                _, derivative = multiway_dual_exact([(eye(3),zeros(3,3)),child],
                                                    [1-epsilon,epsilon],
                                                    [F(0),F(0),F(1)])
                col = extract_vector(derivative)
                for row in range(3):
                    s_total[row][j] = col[row]
            trace = sum(s_total[j][j] for j in range(3))
            phi = epsilon*p
            harmonic = 1-epsilon+epsilon/a
            prediction = (a+1)/(a*a*harmonic)+1/a
            constant = 4*(r+2)/(r+1)**2
            assert trace/phi == prediction
            assert trace/phi >= constant
            data.append(dict(contrast=str(r),epsilon=str(epsilon),
                             trace_per_active_volume=float(trace/phi),
                             exact_finite_formula_verified=True,
                             sharp_constant=float(constant),
                             excess=float(trace/phi-constant)))
    return dict(exact_cases=len(data), cases=data)


def finite_volume_family_exact():
    data = []
    for r in [F(1), F(3), F(16), F(400)]:
        for phi in [F(0), F(1,10), F(1,2), F(4,5), F(1)]:
            p = (1+phi)/2
            epsilon = 2*phi/(1+phi)
            s_total = zeros(3,3)
            for j, field in enumerate(eye(3)):
                child = laminate_dual((eye(3), zeros(3,3)),
                                       (scale(r,eye(3)), cross_exact(field)),
                                       1-p, [F(1), F(0), F(0)])
                _, derivative = multiway_dual_exact([(eye(3),zeros(3,3)),child],
                                                    [1-epsilon,epsilon],
                                                    [F(0),F(0),F(1)])
                col = extract_vector(derivative)
                for row in range(3):
                    s_total[row][j] = col[row]
            trace = sum(s_total[j][j] for j in range(3))
            bound = 4*phi*(r+2-(r-1)*phi)/(r+1-(r-1)*phi)**2
            assert trace == bound
            assert epsilon*p == phi
            data.append(dict(contrast=str(r),active_volume=str(phi),
                             child_fraction=str(p),outer_child_weight=str(epsilon),
                             exact_trace=str(trace), trace=float(trace),
                             sharp_bound_attained_exactly=True))
    return dict(exact_cases=len(data),cases=data)


if __name__ == '__main__':
    result = dict(exact=exact_case(), numerical=floating_cases(),
                  sharp_family=sharp_family_exact(),finite_volume_family=finite_volume_family_exact())
    output = Path(__file__).with_name('multiway_audit_results.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
