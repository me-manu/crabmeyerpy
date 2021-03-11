import yaml
from scipy.special import kv  # Bessel function
from scipy.integrate import simps
from scipy.interpolate import interp1d

# imports to speed up integrations:
from numpy import meshgrid, linspace, ones, zeros
from numpy import log, exp, pi, sqrt, power, tan

# import functions for photon fields
from .photonfields import *
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import Planck15 as cosmo

# define conversion factors
kpc2cm = u.kpc.to('cm')
eV2Hz = 1. / (c.h.value * u.J.to('eV'))
eV2erg = u.eV.to('erg')
m_e_eV = (c.m_e * c.c**2.).to('eV').value
arcmin2rad = u.arcmin.to('rad')


def ic_kernel(nu, gamma, e):
    """
    Calculate the full inverse Compton Kernel, unitless

    Parameters
    ----------
    nu: array-like
        final photon frequency in Hz
    gamma: array-like
        gamma factor of electrons
    e: array-like
        initial photon energy in eV

    Returns
    -------
    Inner IC kernel including KN limit

    Notes
    -----
    gamma, e, and e1 need to have same shape.
    See Blumenthal & Gould 1970, Eq. 2.47 - 2.51
    """
    q = nu / eV2Hz / 4. / gamma ** 2. / e / (1. - nu / eV2Hz / m_e_eV / gamma)

    m = (q <= 1.) & (q >= 1. / 4. / gamma ** 2.)

    f = zeros(q.shape)

    f[m] = 2. * q[m] * log(q[m]) + (1. + 2. * q[m]) * (1. - q[m]) + \
        (4. * e[m] / m_e_eV * gamma[m] * q[m]) ** 2. \
        / 2. / (1. + 4. * e[m] / m_e_eV * gamma[m] * q[m]) \
        * (1. - q[m])

    return f


class CrabSSC(object):
    def __init__(self, config, n_el, B=124.e-6, d=2., nu_sync_min=1e7, nu_sync_max=1e30):
        """
        Initialize the class

        Parameters
        ----------
        config: str or dict
            path to config file with model parameters.
            Should contain three dictionaries:
            - params_n_el: parameters for the electron density
            - params_n_seed: parameters for the photon density

        n_el: function pointer
            electron density spectrum. Should be called with n_el(gamma, **params_n_el)

        {options}
        B: float
            magnetic field of the nebula in G

        d: float
            distance to the nebula in kpc

        nu_sync_min: float
            minimum frequency considered for syncrotron radiation

        nu_sync_max: float
            maximum frequency considered for syncrotron radiation
        """

        # read in config file
        if isinstance(config, dict):
            conf = config
        else:
            with open(config) as f:
                conf = yaml.safe_load(f)

        self._params_n_el = conf['params_n_el']
        self._params_n_seed = conf['params_n_seed']

        self._nu_sync_min = nu_sync_min
        self._nu_sync_max = nu_sync_max
        self._n_el = n_el
        self._B = B
        self._d = d

        # Interpolate x F(x) of synchrotron function,
        # see e.g. Fig. 13 in Blumenthal & Gould 1970
        steps = 100
        self.__start = -40  # upper limit for x F (x) integration
        self.__end = 20  # upper limit for x F (x) integration

        # build a 2d array for interpolation
        logx = np.linspace(self.__start, self.__end+1, steps)

        for i, s in enumerate(logx):
            if not i:
                logx_arr = np.linspace(s, self.__end, steps)
            else:
                logx_arr = np.vstack((logx_arr, np.linspace(s, self.__end, steps)))

        xF = np.exp(logx) * simps(kv(5./3., np.exp(logx_arr)) * np.exp(logx_arr), logx_arr, axis=1)
        xF[xF < 1e-40] = np.full(np.sum(xF < 1e-40), 1e-40)
        self.log_xF = interp1d(logx, np.log(xF))

        self.FSyncInterp = None

        return

    @property
    def params_n_el(self):
        return self._params_n_el

    @property
    def params_n_seed(self):
        return self._params_n_seed

    @property
    def n_el(self):
        return self._n_el

    @property
    def B(self):
        return self._B

    @property
    def d(self):
        return self._d

    @n_el.setter
    def n_el(self, n_el):
        self._n_el = n_el

    @B.setter
    def B(self, B):
        self._B = B

    @d.setter
    def d(self, d):
        self._d = d

    def sync(self, nu, g_steps=50, gmin=None, gmax=None):
        """
        Spectral synchrotron luminosity F_nu in erg/s/Hz/cm^2 as integral over electron distribution

        Parameters:
        -----------
        nu: array-like
            frequencies in Hz

        {options}

        g_steps: int
            number of integration steps

        gmin: float or None
            minimum lorentz factor

        gmax: float or None
            maximum lorentz factor

        Returns:
        --------
        array with spectral luminosity F_nu  density at frequency nu
        """
        if gmin is None:
            gmin = self._params_n_el['gradio_min']
        if gmax is None:
            gmax = self._params_n_el['gwind_max']

        # 2d grid for Freq and gamma factors
        nn, gg = meshgrid(nu, linspace(log(gmin), log(gmax), g_steps), indexing='ij')

        # x = nu / nu_c as 2d grid,
        # nu_c: critical frequency for B in G; Longair vol.2 p. 261
        nu_c = 4.199e10 * self._B * u.G.to('T') * exp(gg)**2.
        x = nn / nu_c

        # define a mask for integration
        m = (log(x) > self.__start) & (log(x) < self.__end)
        result = np.full(x.shape, 1e-40)

        # synchrotron function
        result[m] = exp(self.log_xF(log(x[m])))

        # multiply with electron spectrum
        result *= self._n_el(exp(gg), **self._params_n_el)

        # integrate over gamma
        result = simps(result * exp(gg), gg, axis=1)

        # pre factors: sqrt(3) * e^3 / mc^2 with B in G, see e.g. B&G 4.44
        # this has then units Fr^3 s^2 B g-1 cm-2
        # When you use Fr G s^2 / (cm g) = 1 you get
        # units Fr^2 / cm and with Fr = cm^3/2 g^1/2 s^-1
        # this becomes g cm^2 s^2 = erg = erg / Hz / s.
        # The pre factor is then consistent with 18.36 in Longair Vol.2
        # since he calculates in W and for B in Tesla
        result *= ((c.e.esu**3.) / (c.m_e.cgs * c.c.cgs**2.) * sqrt(3.)).value
        # this is equal to 2.344355730864404e-22

        # average over all pitch angles gives 2/3
        result *= self._B * sqrt(2.0/3.0)

        # divide by the distance squared
        # change from intrinsic luminosity to flux
        result /= 4. * pi * self._d * self._d * kpc2cm * kpc2cm

        # returns value in unites erg/s/Hz/cm^2 
        return result

    def interp_sync_init(self, g_steps=100):
        """
        Initialize interpolation of Spectral synchrotron luminosity F_nu in erg/s/Hz/cm^2 for given electron spectrum,
        in log - log space.
        Sets self.FSyncInterp function pointer.

        Parameters
        ----------
        g_steps: int,
            number of integration steps
        """
        nu = np.logspace(np.log10(self._nu_sync_min), np.log10(self._nu_sync_max), 200)
        F_sync = self.sync(nu, g_steps=g_steps)
        self.FSyncInterp = interp1d(log(nu), log(F_sync))

    def grey_body_old(self, nu):
        """
        Return grey body nu F_nu spectrum in erg/s/cm^2

        Parameters
        ----------
        nu: array like
            array with frequencies in Hz

        Returns
        -------
        array with grey body flux in erg/s/cm^2

        Note
        ----
        TODO: I don't think that this is correct.
        TODO: From the photon density you should simply
        TODO: multiply with (h nu) * c / 4 pi to get the specific intensity
        """

        # photons dens of black body in photons/eV/cm^3
        result = black_body(nu / eV2Hz, self._params_n_seed['dust_T'])
        result *= self._params_n_seed['dust_norm']

        # this is in units of photons/cm^3/eV 
        # assume an emitting volume, using the scale length
        # suggested by Hillas: 1.3 arcmin 
        # now this is in units of photons / eV
        result *= 4.0 / 3.0 * pi * power(tan(self._params_n_seed['dust_extension'] * arcmin2rad)
                                         * self._d * kpc2cm, 3.)

        # calculate erg per s per cm**2 
        result *= (nu * nu / eV2Hz / eV2Hz) * eV2erg
        result /= 4.0 * pi * (self._params['d'] * kpc2cm * self._d * kpc2cm)
        return result

    def grey_body(self, nu):
        """
        Return grey body nu F_nu spectrum in erg/s/cm^2

        Parameters
        ----------
        nu: array like
            array with frequencies in Hz

        Returns
        -------
        array with grey body flux in erg/s/cm^2/Hz
        """

        # photons dens of black body in photons/eV/cm^3
        result = black_body(nu / eV2Hz, self._params_n_seed['dust_T'])
        result *= self._params_n_seed['dust_norm']

        # change to dens in photon / Hz / cm^3, dn / d nu = dn / de * de / d nu = dn / de * h
        result *= c.h.to('eV s').value

        # multiply with energy to get energy density per Hz
        result *= nu * c.h.to('erg s').value

        # multiply with c / 4 pi to get energy flux in erg / s / cm^2 / Hz
        result *= c.c.cgs.value / 4. / pi

        # rescale this from sphere of emitting region
        # suggested by Hillas: 1.3 arcmin to distance of the Crab
        # 4 pi tan(theta) d ** 2 / 4 pi d**2 = tan(theta)
        result *= tan(self._params_n_seed['dust_extension'] * arcmin2rad)
        return result

    def sync_phot_dens(self, eps, gamma):
        """
        Calculate synchrotron photon number density of Crab nebula according to Hillas et al. (1998)

        Parameters
        ----------
        eps: array-like
            n-dim array with energy of photons, in eV
        gamma: array
            m-dim array with gamma factor of electrons

        Returns
        -------
        m x n-dim array with photon densities in photons / eV / cm^3

        Notes
        -----
        See https://arxiv.org/pdf/1008.4524.pdf Eq. (A3)
        """

        # eps is in units of eV 
        # get synchrotron luminosity in units of erg/s/cm^2/Hz, F_nu
        S = np.full(eps.shape[0], 1e-40)

        # include synchrotron photon density
        if self._params_n_seed['ic_sync']:

            # initialize synchrotron interpolation
            if self.FSyncInterp is None:
                self.interp_sync_init()

            # mask for frequencies
            m = (log(eps * eV2Hz) > log(self._nu_sync_min)) & \
                (log(eps * eV2Hz) < log(self._nu_sync_max))

            # calculate synchrotron intergral from interpolation
            S[m] = exp(self.FSyncInterp(log(eps * eV2Hz)[m]))

        # conversion:
        # Now in units of erg/s/cm^2
        # nu F_nu
        S *= eps * eV2Hz

        # convert in units of photons/cm^2/s
        #S /= (eps * u.eV.to('J') / c.h.value) * u.eV.to('erg')
        S /= (eps * eV2erg)

        # total production rate of photons in units of 1/s */
        S *= (4.0 * pi * (self._d * kpc2cm)**2.)

        # calculate the scale length of the electrons "seeing" the photons according to Hillas et al. (1998)
        rho = zeros(gamma.shape)
        m = gamma * m_e_eV / 1e9 < 34.
        rho[m] = tan(1.35 * arcmin2rad) * self._d * kpc2cm

        extension = 0.15 + 1.2*power(gamma[~m] * m_e_eV / 34. / 1e9, -0.17)
        rho[~m] = tan(extension * arcmin2rad) * self._d * kpc2cm

        # calculate scale length of photon density in the nebular
        sigma = zeros(eps.shape)
        m = eps < 0.02
        sigma[m] = tan(1.35 * arcmin2rad) * self._d * kpc2cm
        extension = 0.16 + 1.19 * power(eps[~m]/0.02, -0.09)
        sigma[~m] = tan(extension * arcmin2rad) * self._d * kpc2cm

        # Add Dust Component and line emission
        if self._params_n_seed['ic_dust']:
            S_dust = self.grey_body(eps * eV2Hz)
            S_dust *= eps * eV2Hz
            S_dust /= (eps * eV2erg)
            S_dust *= (4.0 * pi * (self._d * kpc2cm)**2.)
            # calculate scale length of photon density in the nebular
            sigma_dust = tan(self._params_n_seed['dust_extension'] * arcmin2rad) * self._d * kpc2cm

        # TODO: check if this combination is the right way to do it
        # TODO: or if the overlap has to be calculated differently
        # calculate photon density in photons/cm**3/eV
        if len(sigma.shape) == 1 and not sigma.shape[0] == rho.shape[0]:
            ss, rr = meshgrid(sigma, rho)
            S, _ = meshgrid(S, rho)
            ee, _ = meshgrid(eps, gamma)
            S /= (4.0 * pi * c.c.cgs.value * (ss * ss + rr * rr))

            if self._params_n_seed['ic_dust']:
                sd, _ = meshgrid(sigma_dust, rho)
                S_dust, _ = meshgrid(S_dust, rho)
                S_dust /= (4.0 * pi * c.c.cgs.value * (sd * sd + rr * rr))
                S += S_dust

            S /= ee
        else:
            S /= (4.0 * pi * c.c.cgs.value * (sigma * sigma + rho * rho))
            if self._params_n_seed['ic_dust']:
                S_dust /= (4.0 * pi * c.c.cgs.value * (sigma_dust * sigma_dust + rho * rho))
                S += S_dust
            S /= eps

        return S

    def ic(self, nu, g_steps=200, e_steps=90):
        """
        Spectral luminosity F_nu in erg/s/Hz/cm^2 for inverse Compton scattering.

        Parameters:
        -----------
        nu: array-like
            n-dim array with frequencies in Hz

        {options}

        g_steps: int
            number of integration steps for gamma
        e_steps: int
            number of integration steps for energy

        Returns:
        --------
        n-dim numpy array spectral luminosity F_nu  density at frequency nu
        """

        log_g = linspace(log(self._params_n_el['gmin']), log(self._params_n_el['gmax']), g_steps)
        gamma = exp(log_g)

        result = zeros(nu.shape[0])

        # generate the arrays for observed freq nu, gamma factor, in energy of photon field
        nn, gg = meshgrid(nu, log_g, indexing='ij')
        nnn, ggg, eee = meshgrid(nu, log_g, linspace(0., 1., e_steps), indexing='ij')
        x1 = log(nnn / eV2Hz / 4. / ggg ** 2.)
        x1[x1 < 1e-18] = 1e-18
        x2 = log(nnn / eV2Hz)

        log_eee = zeros(nnn.shape)
        m = zeros(nnn.shape, dtype=np.bool)
        for i, n in enumerate(nu):
            for j, lg in enumerate(log_g):
                x1 = max(log(n / eV2Hz / 4. / gamma[j] ** 2.), log(1e-18))
                x2 = log(n / eV2Hz)
                # now log_eps has shape g_steps x e_steps
                log_eee[i, j] = linspace(x1, x2, e_steps)
                if x2 > x1:
                    m[i, j] = True

        # calculate photon densities:
        # these are in photons / eV / cm^3
        phot_dens = np.zeros(eee.shape)

        if self._params_n_seed['ic_sync'] or self._params_n_seed['ic_dust']:
            phot_dens[m] = self.sync_phot_dens(exp(log_eee[m]), exp(ggg[m]))

        if self._params_n_seed['ic_cmb']:
            phot_dens[m] += black_body(exp(log_eee[m]), cosmo.Tcmb0.value)

        # IC scattering kernel
        f = ic_kernel(nnn, exp(ggg), exp(log_eee))

        # multiply the two in integrate over initial photon energy
        kernel_in = phot_dens * f

        # kernel needs to be divided by exp(log_eee) but
        # cancels since we're integrating over log(energy).
        # now in photons / cm^3 / eV
        kernel_out = simps(kernel_in, log_eee, axis=2)
        kernel_out *= self._n_el(exp(gg), **self._params_n_el) / exp(gg) ** 2.

        # integrate over electron gamma factor
        result = simps(kernel_out * exp(gg), gg, axis=1)

        # result of integration is in units of photons/cm**3/eV
        # multiplying with Thomson*c*energy gives and convert to
        # units of erg/sec/eV
        result *= 3. / 4. * (c.sigma_T.cgs * c.c.cgs).value * nu / eV2Hz * eV2erg
        # convert to erg / sec / Hz
        # this is the spectral luminosity L_nu
        result /= eV2Hz
        #  divide by the distance squared to get the flux
        result /= 4. * pi * (self._d * kpc2cm)**2.
        return result
