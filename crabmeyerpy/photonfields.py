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
    m = kx < 1e-8
    result = np.exp(kx) - 1.
    m = result == 0.
    result[m] = kx[m] + kx[m] * kx[m] / 2. + kx[m] * kx[m] * kx[m] / 6.  # use taylor expansion - 1

    if np.any(result == 0.):
        raise ValueError("Zero encountered in BB component, will lead to nans in IC component!")
    result = 2. * c.h.cgs.value * nu**3. / result
    result /= c.c.cgs.value**2.
    return result


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
    result = intensity_black_body(nu, params['dust_T'])

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
    result = intensity_black_body(nu, params['dust_T'])

    # multiply with some arbitrary norm in units cm^2
    result *= params['dust_norm']

    # multiply the extension
    volume = 4. / 3. * np.pi * (params['r_max'] ** 3. - params['r_min'] ** 3.)
    result /= volume

    # set emissivity to zero outside
    # the shell
    mask = (r >= params['r_min']) & (r <= params['r_max'])
    #result[~mask] = 0.
    result[~mask] = 1e-100

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
    # compute the wavelength in mu
    l_mu = c.c.value / nu * 1e6

    # specific intensity in erg / s / Hz / cm^2 / sr
    b1 = intensity_black_body(nu, params['dust_T1'])
    b2 = intensity_black_body(nu, params['dust_T2'])
    result = 10.**params['log10_M1'] * c.M_sun.cgs.value * b1
    result += 10.**params['log10_M2'] * c.M_sun.cgs.value * b2

    # multiply with absorption term in cm^2 / g
    result *= params['abs_norm'] * np.power(l_mu, -params['abs_index'])

    # multiply the extension
    volume = 4. / 3. * np.pi * (params['r_max'] ** 3. - params['r_min'] ** 3.)
    result /= volume

    # set emissivity to zero outside
    # the shell
    mask = (r >= params['r_min']) & (r <= params['r_max'])
    result[~mask] = 0.

    return result
