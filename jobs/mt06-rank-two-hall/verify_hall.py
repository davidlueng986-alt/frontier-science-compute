"""Independent Hall audit using six interface equations, not cofactor recursion.

Each laminate is solved as a cell transmission problem at finite magnetic field.
The matrix of effective resistivities is then differentiated to extract its Hall
tensor.  No recursion for the weak-field Hall tensor is used in this verifier.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def cross(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.result_type(v))


def normal(theta):
    return np.array([np.sin(theta), 0.0, np.cos(theta)])


def tangent_frame(n):
    n = np.asarray(n, dtype=float)
    n /= np.linalg.norm(n)
    axis = np.eye(3)[np.argmin(np.abs(n))]
    t1 = np.cross(n, axis)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return n, t1, t2


def laminate_interface(kx, ky, f, n):
    """Solve mean E, tangential E continuity, normal j continuity."""
    n, t1, t2 = tangent_frame(n)
    dtype = np.result_type(kx, ky)
    equations = np.zeros((6, 6), dtype=dtype)
    equations[:3, :3] = f * np.eye(3)
    equations[:3, 3:] = (1.0 - f) * np.eye(3)
    equations[3, :3], equations[3, 3:] = t1, -t1
    equations[4, :3], equations[4, 3:] = t2, -t2
    equations[5, :3], equations[5, 3:] = n @ kx, -n @ ky
    rhs = np.zeros((6, 3), dtype=dtype)
    rhs[:3] = np.eye(3)
    solution = np.linalg.solve(equations, rhs)
    ex, ey = solution[:3], solution[3:]
    return f * kx @ ex + (1.0 - f) * ky @ ey


def phase(r, field, finite_resistivity=True):
    """High phase A=I, low phase A=0; low sigma normalized to one."""
    field = np.asarray(field)
    low = np.eye(3, dtype=np.result_type(field))
    if finite_resistivity:
        high = np.linalg.inv(np.eye(3) / r + cross(field))
    else:
        high = r * np.eye(3) - r * r * cross(field)
    return low, high


def hierarchical(params, field, finite_resistivity=True):
    p = params
    if 'nodes' in p:
        low, high = phase(float(p['contrast']), field, finite_resistivity)
        leaves, nodes = {-1: low, -2: high}, []
        for node in p['nodes']:
            left, right = int(node['left']), int(node['right'])
            x = leaves[left] if left < 0 else nodes[left]
            y = leaves[right] if right < 0 else nodes[right]
            nodes.append(laminate_interface(x, y, float(node['f']),
                                           np.array(node['normal'], dtype=float)))
        return nodes[-1]
    low, high = phase(p['r'], field, finite_resistivity)
    d = laminate_interface(low, high, p['fF'], normal(p['thetaD']))
    e = laminate_interface(low, high, p['fH'], [0, 1, 0])
    c = laminate_interface(d, e, p['fD'], normal(p['thetaC']))
    a = laminate_interface(low, c, p['fB'], normal(p['thetaA']))
    return a


def tensor_from_complex_step(params, step=1e-7, finite_resistivity=True):
    k0 = hierarchical(params, np.zeros(3), finite_resistivity)
    result = np.zeros((3, 3))
    residuals = []
    for j in range(3):
        field = np.zeros(3, dtype=complex)
        field[j] = step * 1j
        resistivity = np.linalg.inv(hierarchical(params, field, finite_resistivity))
        derivative = resistivity.imag / step
        residuals.append(np.linalg.norm(derivative + derivative.T))
        result[:, j] = [derivative[2, 1], derivative[0, 2], derivative[1, 0]]
    return k0, result, residuals


def tensor_from_central_difference(params, step=1e-6, finite_resistivity=True):
    result = np.zeros((3, 3))
    residuals = []
    for j in range(3):
        field = np.zeros(3)
        field[j] = step
        plus = np.linalg.inv(hierarchical(params, field, finite_resistivity))
        minus = np.linalg.inv(hierarchical(params, -field, finite_resistivity))
        derivative = (plus-minus) / (2*step)
        residuals.append(np.linalg.norm(derivative + derivative.T))
        result[:, j] = [derivative[2, 1], derivative[0, 2], derivative[1, 0]]
    return result, residuals


def audit(params):
    k, a, residual = tensor_from_complex_step(params, 1e-10)
    data = dict(parameters=params, conductivity=k.tolist(), hall_tensor=a.tolist(),
                hall_A33=float(a[2, 2]), min_conductivity_eigenvalue=float(np.linalg.eigvalsh(k).min()),
                antisymmetry_residual=max(residual), derivative_convergence=[])
    for h in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        ac, er = tensor_from_central_difference(params, h)
        al, _ = tensor_from_central_difference(params, h, False)
        data['derivative_convergence'].append(dict(h=h, finite_resistivity_A33=float(ac[2, 2]),
                                                 linear_conductivity_A33=float(al[2, 2]),
                                                 error=float(np.max(np.abs(ac-a))), antisymmetry_residual=max(er)))
    return data


BASELINE = dict(r=400.0, thetaA=2.7224, thetaC=0.2078, thetaD=0.4712,
                fB=0.9590, fD=0.3834, fF=0.8670, fH=0.2652)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    params = json.loads(args.params.read_text()) if args.params else BASELINE
    data = audit(params)
    encoded = json.dumps(data, indent=2) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(encoded)
