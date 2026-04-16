import numpy as np


def beam_3pt_deflection(L, EI, F, n=100):
    """Return x positions and deflection v(x) for a simply supported beam
    under 3-point bending (central point load F)."""
    x = np.linspace(0, L, n)                           # Discretized beam axis
    xr = np.minimum(x, L - x)                          # Use symmetry (mirror right side)
    v = (F * xr / (48 * EI)) * (3 * L**2 - 4 * xr**2)  # Analytical deflection
    return x, v


def beam_4pt_deflection(x, F, E, I, L, a, b):
    RB=F*(a+b)/(2*L)
    RA=F-RB
    C1=(F*((L-a)**3+(L-b)**3)/12 - RA*L**3/6)/L
    if x<=a: 
        v=RA*x**3/6 + C1*x
    elif x<=b: 
        v=RA*x**3/6 - F*(x-a)**3/12 + C1*x
    else: 
        v=RA*x**3/6 - F*(x-a)**3/12 - F*(x-b)**3/12 + C1*x
    return v/(E*I)

def beam_clamped_tension(F, E, S, L, n=100):
    x = np.linspace(0, L, n)
    ux = F * x / (E * S)
    return x, ux