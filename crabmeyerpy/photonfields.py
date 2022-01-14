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


def intensity_black_body(nu, T):
    """
    Return specific intensity B_nu of black body in erg / s / Hz / cm^2 / sr

    Parameters
    ----------
    nu: array-like
        frequency in Hz
    T: float
        black body temperature in K

    Returns
    -------
    Return array with specific intensity B_nu of black body in erg / s / Hz / cm^2 / sr
    """
    kx = c.h.value * nu / T / c.k_B.value
    result = np.exp(kx) - 1.
    result = 2. * c.h.value * nu**3. / result
    result /= c.c.cgs.value**2.


def j_grey_body_gauss(r, nu, **params):
    """
    Return the volume emissivity for a grey body ergs / s / cm^3 / Hz / sr
    distributed as a Gaussian

    Parameters
    ----------
    r: array-like
        distance from center; must have same
        units as 'dust-extension' parameter
    nu: array-like
        frequency in hz
    params: dict
        Dictionary with parameters

    Returns
    -------
    result: array of volume emissivity for a grey body ergs / s / cm^3 / Hz / sr
    """

    # specific intensity in erg / s / Hz / cm^2 / sr
    result = intensity_black_body(nu, params['T'])

    # multiply with some arbitrary norm in units 1 / cm
    result *= params['dust_norm']

    # multiply the extension
    result *= np.exp(-r ** 2. / 2. / params['dust_extension'] ** 2.)

    return result


def j_grey_body_shell(r, nu, **params):
    """
    Return the volume emissivity for a grey body ergs / s / cm^3 / Hz / sr
    distributed in a shell between r_min and r_max

    Parameters
    ----------
    r: array-like
        distance from center; must have same
        units as 'dust-extension' parameter
    nu: array-like
        frequency in hz
    params: dict
        Dictionary with parameters

    Returns
    -------
    result: array of volume emissivity for a grey body ergs / s / cm^3 / Hz / sr
    """
    # specific intensity in erg / s / Hz / cm^2 / sr
    result = intensity_black_body(nu, params['T'])

    # multiply with some arbitrary norm in units 1 / cm^2
    result *= params['dust_norm']

    # multiply the extension
    volume = 4. / 3. * np.pi * (params['r_max'] - params['r_min']) ** 3.
    result /= volume

    # set emissivity to zero outside
    # the shell
    mask = (r >= params['r_min']) & (r <= params['r_max'])
    result[~mask] = 0.

    return result


def j_dust_carbon_shell(r, nu, **params):
    """
    Return the volume emissivity for a grey body ergs / s / cm^3 / Hz / sr
    for the amorphous carbon model discussed in Horns et al. 2022
    within in a shell

    Parameters
    ----------
    nu: array-like
        frequency in Hz
    params: dict
        Dictionary with parameters

    Returns
    -------
    result: array of volume emissivity for a grey body ergs / s / cm^3 / Hz / sr
    """
    result = np.zeros_like(r)

    # compute the wavelength in mu
    l_mu = c.c.value / nu * 1e6

    # specific intensity in erg / s / Hz / cm^2 / sr
    mask_nu = (l_mu >= params['dust_l_min'])
    mask_nu &= (l_mu <= params['dust_l_max'])

    b1 = intensity_black_body(nu[mask_nu], params['T1'])
    b2 = intensity_black_body(nu[mask_nu], params['T2'])
    result[mask_nu] = 10.**params['log10_M1'] * c.M_sum.cgs.value * b1
    result[mask_nu] += 10.**params['log10_M2'] * c.M_sum.cgs.value * b2

    # multiply with absorption term in cm^2 / g
    result[mask_nu] *= params['abs_norm'] * l_mu[mask_nu] ** -params['abs_index']

    # multiply the extension
    volume = 4. / 3. * np.pi * (params['r_max'] - params['r_min']) ** 3.
    result /= volume

    # set emissivity to zero outside
    # the shell
    mask = (r >= params['r_min']) & (r <= params['r_max'])
    result[~mask] = 0.

    return result
