"""Exact rational Hall sign certificate from local transmission equations.

Input format:
 {"contrast": "400", "nodes": [
   {"left": -1, "right": -2, "f": "0.5", "normal": ["1", "0", "1"]}
 ]}

The two leaves are -1 (sigma=I, Hall=0) and -2 (sigma=r I, Hall=I).
Each internal node references earlier node indices.  Only the derivative for a
magnetic field in direction 3 is required for an exact A33 sign certificate.
All numeric inputs are read as rational decimal literals.  Prefer JSON strings
when literal provenance, rather than float-round-trip precision, is required.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
from pathlib import Path


def number(x):
    return F(str(x))


def zeros(rows, cols):
    return [[F(0) for _ in range(cols)] for _ in range(rows)]


def eye(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def add(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scale(s, a):
    return [[s*x for x in row] for row in a]


def mul(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def transpose(a):
    return list(map(list, zip(*a)))


def solve(a, b):
    """Exact Gaussian elimination with full row operations."""
    n, m = len(a), len(b[0])
    aug = [ar[:] + br[:] for ar, br in zip(a, b)]
    for col in range(n):
        pivot = next((j for j in range(col, n) if aug[j][col]), None)
        if pivot is None:
            raise ValueError('Singular local transmission system')
        aug[col], aug[pivot] = aug[pivot], aug[col]
        d = aug[col][col]
        aug[col] = [x / d for x in aug[col]]
        for i in range(n):
            if i == col:
                continue
            c = aug[i][col]
            if c:
                aug[i] = [x-c*y for x, y in zip(aug[i], aug[col])]
    return [row[n:n+m] for row in aug]


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def tangent_frame(n):
    if not any(n):
        raise ValueError('Normal cannot be zero')
    for axis in eye(3):
        t1 = cross(n, axis)
        if any(t1):
            break
    t2 = cross(n, t1)
    return t1, t2


def local_equations(kx, ky, f, n):
    t1, t2 = tangent_frame(n)
    a = zeros(6, 6)
    for i in range(3):
        a[i][i], a[i][i+3] = f, 1-f
    a[3] = t1 + [-x for x in t1]
    a[4] = t2 + [-x for x in t2]
    a[5] = mul([n], kx)[0] + [-x for x in mul([n], ky)[0]]
    return a


def laminate_dual(x, y, f, n):
    kx, dx = x
    ky, dy = y
    a0 = local_equations(kx, ky, f, n)
    a1 = zeros(6, 6)
    a1[5] = mul([n], dx)[0] + [-z for z in mul([n], dy)[0]]
    rhs = eye(3) + zeros(3, 3)
    e0 = solve(a0, rhs)
    e1 = solve(a0, scale(F(-1), mul(a1, e0)))
    ex0, ey0, ex1, ey1 = e0[:3], e0[3:], e1[:3], e1[3:]
    k = add(scale(f, mul(kx, ex0)), scale(1-f, mul(ky, ey0)))
    d = add(scale(f, add(mul(dx, ex0), mul(kx, ex1))),
            scale(1-f, add(mul(dy, ey0), mul(ky, ey1))))
    return k, d


def det3(a):
    return sum(a[0][i] * (a[1][(i+1)%3]*a[2][(i+2)%3]
                          -a[1][(i+2)%3]*a[2][(i+1)%3]) for i in range(3))


def assert_physical(k, d):
    if k != transpose(k):
        raise AssertionError('Zero-field K is not exactly symmetric')
    if d != scale(F(-1), transpose(d)):
        raise AssertionError('Field derivative K is not exactly antisymmetric')
    minors = [k[0][0], k[0][0]*k[1][1]-k[0][1]*k[1][0], det3(k)]
    if any(x <= 0 for x in minors):
        raise AssertionError('Zero-field K is not positive definite')
    return minors


def rational_string(x):
    return f'{x.numerator}/{x.denominator}'


def floats(a):
    return [[float(x) for x in row] for row in a]


def verify(spec):
    r = number(spec['contrast'])
    if r <= 1:
        raise ValueError('Expected contrast > 1')
    # Derivative of inverse(r^-1 I + B3 [e3]_cross) at B3=0.
    high_derivative = [[F(0), r*r, F(0)], [-r*r, F(0), F(0)], [F(0)]*3]
    leaves = {-1: (eye(3), zeros(3, 3)), -2: (scale(r, eye(3)), high_derivative)}
    vals, node_reports = [], []
    for i, node in enumerate(spec['nodes']):
        left, right = int(node['left']), int(node['right'])
        if left >= i or right >= i or left < -2 or right < -2:
            raise ValueError('Node references must be valid leaves or earlier nodes')
        f = number(node['f'])
        if not 0 < f < 1:
            raise ValueError('Strictly interior volume fractions required')
        n = [number(x) for x in node['normal']]
        x = leaves[left] if left < 0 else vals[left]
        y = leaves[right] if right < 0 else vals[right]
        value = laminate_dual(x, y, f, n)
        minors = assert_physical(*value)
        vals.append(value)
        node_reports.append(dict(node=i, positive_definite=True, exact_reciprocity=True,
                                 principal_minor_signs=[int(x > 0) for x in minors]))
    k, d = vals[-1]
    rho = solve(k, eye(3))
    rho_derivative = scale(F(-1), mul(mul(rho, d), rho))
    if rho_derivative != scale(F(-1), transpose(rho_derivative)):
        raise AssertionError('Resistivity derivative is not exactly antisymmetric')
    a33 = rho_derivative[1][0]
    return dict(method='Exact rational transmission-equation differentiation',
                candidate=spec, conductivity=floats(k),
                B3_conductivity_derivative=floats(d),
                B3_resistivity_derivative=floats(rho_derivative),
                A33=float(a33), A33_exact=rational_string(a33),
                A33_numerator_sign=(a33 > 0)-(a33 < 0),
                sign_inversion_certified=bool(a33 < 0),
                intermediate_node_checks=node_reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(json.loads(args.candidate.read_text()))
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)
