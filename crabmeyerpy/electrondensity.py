import numpy as np


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
