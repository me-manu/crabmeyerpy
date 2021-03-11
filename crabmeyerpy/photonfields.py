import numpy as np
import astropy.constants as c


def black_body(eps, T):
    """
    return photon density for black body in photons / eV / cm^3 for temperature T

    Parameters
    ----------
    eps: array-like
        energy in eV
    T: float
        black body temperature

    Returns
    -------
    result: array of photon densities in photons / eV / cm^3
    """

    kx = eps / T / (c.k_B.value / c.e.value)  # k/e = 8.617e-5, exponent unitless for eps in eV 
    result = np.exp(kx) - 1.
    result = eps ** 2. / result
    result /= (c.hbar * c.c).to('eV cm').value**3. * np.pi**2.
    return result

