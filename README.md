# Frontier science: public reproducibility jobs

This repository hosts executable checks for research in the companion
frontier-science-open-problems repository. A successful workflow certifies
only the checks actually run; it is not peer review or a formal proof of
an infinite family.

## MT-06: the sharp depth-two Hall-trace bound

The job at jobs/mt06-rank-two-hall checks a proposed exact minimum for
two-phase weak-field Hall response over finite multiway laminates of
depth at most two. The normalized phases are K_L=I, K_H=rI, R_L=0,
R_H=I; phi is the high-phase fraction. The analytic minimum is

    tr(R) >= 4*phi*(r+2-(r-1)*phi)/(r+1-(r-1)*phi)^2.

The attaining response is classical. The candidate research contribution
is its proved optimality over arbitrary oblique and noncoplanar
depth-two structures. The proof and manuscript are maintained with the
research record. This does not solve the unrestricted MT-06 problem.

### Reproduce

From jobs/mt06-rank-two-hall, using Python 3.12:

    python -m pip install -r requirements.txt
    python run_verification.py
    python make_figures.py

The workflow runs exact rational interface calculations, symbolic
identities, 120 seeded multiway checks, 20 exact fixed-volume equality
cases, 16 dilute-family cases, a rational certificate for an existing
rank-three negative-trace design, and 30 independently computed figure
points. Results include source SHA-256 hashes.

The known rank-three baseline is reproduced from Kern,
Physical Review B 108, 235303 (2023),
<https://doi.org/10.1103/PhysRevB.108.235303>. It is not a new design.
Related coated-cylinder responses and bounds are discussed by Kern and
Milton, Advanced Materials 38, e17068 (2026),
<https://doi.org/10.1002/adma.202517068>.
