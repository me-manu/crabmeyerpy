import numpy as np
from astropy import constants as c
from collections import Iterable
from scipy.integrate import simps


def nel_crab(gamma, **params):
    """
    Computes total electron number spectrum per unit gamma interval dN / dgamma
    as used for crab nebula

    gamma: array-like
        gamma factors
    params: dict 
        function parameters

    Returns
    -------
    Total electron spectrum as array
    """
    result = np.zeros(gamma.shape)

    # Wind electron spectrum
    m_wind = (gamma > params['gmin']) & (gamma <= 1. / params['gwind_b'])
    m_wind_br = (gamma > 1. / params['gwind_b']) & (gamma < params['gwind_max'])
    m_wind_tot = (gamma > params['gmin']) & (gamma < params['gwind_max'])
    result[m_wind] += np.power(gamma[m_wind] * params['gwind_b'], params['Swind'])
    result[m_wind_br] += np.power(gamma[m_wind_br] * params['gwind_b'], params['Swind'] + params['Sbreak'])
    result[m_wind_tot] *= params['Nwind']
    result[m_wind_tot] *= np.exp(-np.power(params['gwind_min'] / gamma[m_wind_tot], params['sup_wind']))

    # Radio electron spectrum
    m_radio = (gamma > params['gradio_min']) & (gamma < params['gradio_max'])
    result[m_radio] += np.power(gamma[m_radio], params['Sradio']) * params['Nradio']
    return result


def nel_crab_radio(gamma, **params):
    """
    Computes radio electron number spectrum per unit gamma interval dN / dgamma
    as used for crab nebula

    Paramters
    ---------
    gamma: array-like
        gamma factors
    params: dict 
        function parameters
        Parameter names are:
        Nradio, gradio_min, gradio_max, Sradio

    Returns
    -------
    Electron spectrum as array

    Notes
    -----
    See Eq. 1 in https://arxiv.org/pdf/1008.4524.pdf
    """
    result = np.zeros(gamma.shape)
    m_radio = (gamma > params['gradio_min']) & (gamma < params['gradio_max'])
    result[m_radio] += np.power(gamma[m_radio], params['Sradio']) * params['Nradio']
    return result


def nel_crab_radio_cutoff(gamma, **params):
    """
    Computes radio electron number spectrum per unit gamma interval dN / dgamma
    with exponential cutoff at maximum gamma value

    Paramters
    ---------
    gamma: array-like
        gamma factors
    params: dict
        function parameters
        Parameter names are:
        Nradio, gradio_min, gradio_max, Sradio

    Returns
    -------
    Electron spectrum as array

    Notes
    -----
    See Eq. 1 in https://arxiv.org/pdf/1008.4524.pdf
    """
    result = np.zeros(gamma.shape)
    m_radio = (gamma > params['gradio_min'])
    result[m_radio] = np.power(gamma[m_radio], params['Sradio']) * params['Nradio']
    result[m_radio] *= np.exp(-gamma[m_radio] / params['gradio_max'])
    return result


def nel_crab_wind(gamma, **params):
    """
    Computes wind electron number spectrum per unit gamma interval dN / dgamma
    as used for crab nebula

    Paramters
    ---------
    gamma: array-like
        gamma factors
    params: dict 
        function parameters. 
        Parameter names are:
        Nwind, gmin, gmax, gwind_b, Swind, Sbreak, sup_wind

    Returns
    -------
    Electron spectrum as array

    Notes
    -----
    See Eq. 2 in https://arxiv.org/pdf/1008.4524.pdf
    """
    result = np.zeros(gamma.shape)
    m_wind = (gamma > params['gmin']) & (gamma <= 1. / params['gwind_b'])
    m_wind_br = (gamma > 1. / params['gwind_b']) & (gamma < params['gwind_max'])
    m_wind_tot = (gamma > params['gmin']) & (gamma < params['gwind_max'])
    result[m_wind] += np.power(gamma[m_wind] * params['gwind_b'], params['Swind'])
    result[m_wind_br] += np.power(gamma[m_wind_br] * params['gwind_b'], params['Swind'] + params['Sbreak'])
    result[m_wind_tot] *= params['Nwind']
    result[m_wind_tot] *= np.exp(-np.power(params['gwind_min'] / gamma[m_wind_tot], params['sup_wind']))
    return result


def electron_distribution_width_old(gamma, **params):
    """
    Calculate the energy-dependent width of the electron distribution

    Parameters
    ----------
    gamma: array-like
        gamma values

    params: dict
        dict with parameters

    Returns
    -------
    Width of electron distribution as a function of gamma
    """

    rho = np.zeros(gamma.shape)
    # nebula size constant in energy for electron energies below 34 GeV
    m = gamma < params['gamma_br_const']
    rho[m] = params['radio_size_cm']
    rho[~m] = params['offset'] + params['amplitude'] * np.power(gamma[~m] / params['gamma_br_const'], params['index'])

    return rho


def electron_distribution_width(gamma, **params):
    """
    Calculate the energy-dependent width of the electron distribution


    Parameters
    ----------
    gamma: array-like
        gamma values

    params: dict
        dict with parameters

    Returns
    -------
    Width of electron distribution as a function of gamma
    """

    rho = np.zeros(gamma.shape)
    # nebula size constant in energy for electron energies below 34 GeV
    m = gamma < params['gamma_br_const']
    rho[m] = params['radio_size_cm']
    rho[~m] = params['radio_size_cm'] + params['amplitude'] * (
            np.power(gamma[~m] / params['gamma_br_const'], params['index']) - 1.)
    
    if "norm_spatial" in params:
        rho *= params["norm_spatial"]

    return rho


def nel_crab_extension(r, gamma, **params):
    """
    Spatial distribution of electron density,
    energy dependent, radially symmetric

    r: array-like
        distance from nebula center in cm
    gamma: array-like
        gamma factors
    params: dict
        function parameters

    Returns
    -------
    spatial distribution
    """
    rho = electron_distribution_width(gamma, **params)

    result = np.exp(-r ** 2. / rho ** 2. / 2.)
    
    if "norm_spatial" in params:
        result /= params["norm_spatial"]**2  # keeps flux relatively constant when changing norm_spatial
        
    return result



def vz_sq(z, sigma):
    """
    Compute v * z^2 in KC model

    Parameters
    ----------
    z: array-like
        distance from nebula center normalized to shock radius

    sigma: float
        sigma parameter

    Returns
    -------
    v * z^2 from Eq. A7a in KC
    """
    # compute the downstream flow four velocity
    # Eq. A4 in KC
    u2_sq = (1. + 9. * sigma) / 8.

    # compute Delta values, KC Eq. A7d
    Delta = (u2_sq / sigma - 0.5) / (u2_sq + 0.25)

    # compute y^2, KC Eq. A7c
    y_sq = 27. * (1. + Delta) ** 2. / 2. / Delta ** 3. * z ** 2.

    # compute G, KC Eq. A7b
    G = 1. + np.power(1. + y_sq + np.sqrt((1 + y_sq) ** 2. - 1.), 1. / 3.) \
        + np.power(1. + y_sq - np.sqrt((1. + y_sq) ** 2. - 1), 1. / 3.)

    # compute v z^2, KC Eq. A7a
    vz_sq = np.power(Delta * G / (1. + Delta) / 3., 3.)

    return vz_sq


def gamma_max_kc(z, x_steps=30, eps=1e-4, **params):
    """
    Compute the maximum gamma factor
    in the KC model from Eq. 2.10e
    """
    if not isinstance(z, Iterable):
        z = [z]
    z = np.asarray(z)

    # calculate prefactor
    # checked that this is dimensionless
    # if you use statC = cm^3/2 g^1/2 s^-1
    gamma_max = 1. / (8. * np.sqrt(2.) * c.e.esu.value ** 4. / c.m_e.cgs.value ** 3. / c.c.cgs.value ** 7.)
    gamma_max *= params['r_shock'] / params['sigma'] / params['spin_down_lumi']

    vzz = vz_sq(z, params['sigma'])
    gamma_max /= np.power(vzz, 1. / 3.)

    # I integral is from 1 to z for each z value
    # in a possibly multidimensional array
    # work around: flatten z array, build 2d array
    # integrate and reshape
    z_flat = z.flatten()

    # build integration array
    x = np.empty((z_flat.size, x_steps))
    for i, zi in enumerate(z_flat):
        x[i] = np.linspace(1.-eps, zi+eps, x_steps)

    # kernel
    kernel = x ** 4. / np.power(vz_sq(x, params['sigma']), 10. / 3.)
    I = simps(kernel, x, axis=-1).reshape(z.shape)

    gamma_max /= I

    return gamma_max


def nel_shock(gamma, **params):
    """
    electron density at shock
    from Atoyan & Aharonian (1996)
    """
    result = params['Nwind'] * np.power(params['gwind_min'] + gamma, params['Swind'])
    result *= np.exp(- gamma / params['gwind_max'])

    return result


def nel_wind_kc(gamma, r, fill_value=1e-80, **params):
    """
    Computes electron number density dn / dV dgamma
    of the wind electrons as in Atoyan & Atoyan (1996)
    which is based on the Kennel & Coroniti MHD model

    gamma: array-like
        gamma factors

    r: array-like
        distance from nebula center in cm

    params: dict
        function parameters

    Returns
    -------
    electron density as array
    """
    sigma = params['sigma']

    # distance normalized to shock radius
    z = r / params['r_shock']

    # compute v z^2
    vzz = vz_sq(z, sigma)

    # compute gamma_max (gamma_infty in KC)
    # from Eq. 2.10e in KC
    gamma_max = gamma_max_kc(z, **params)

    # compute gamma' the initial energy
    # with which the electrons where injected
    result = np.full(gamma.shape, fill_value)
    m = gamma < gamma_max
    gamma_prime = gamma[m] * np.power(vzz[m], 1. / 3.) / (1. - gamma[m] / gamma_max[m])

    result[m] = (gamma_prime / gamma[m]) ** 2. * nel_shock(gamma_prime, **params)
    result[m] /= np.power(vzz[m], 4. / 3.)

    return result


def B_kc(r, **params):
    """
    Magnetic field in KC MHC flow model
    """
    sigma = params['sigma']

    # distance normalized to shock radius
    z = r / params['r_shock']

    # B field at shock, KC Eq. 2.2
    B_s = np.sqrt(params['spin_down_lumi'] / c.c.cgs.value / params['r_shock'] ** 2. * \
                  sigma / (1. + sigma))

    # B field down stream of the shock, KC Eq. A5
    B_down = B_s * 3. * (1. - 4. * sigma)

    # B field in nebula
    B_nebula = B_down * z / vz_sq(z, sigma)
    return B_nebula


