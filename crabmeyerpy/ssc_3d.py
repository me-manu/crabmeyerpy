import yaml
import sys
from scipy.special import kv  # Bessel function
from scipy.integrate import simps, trapz
from scipy.interpolate import RectBivariateSpline, interp1d

# imports to speed up integrations:
from numpy import meshgrid, linspace, ones, zeros
from numpy import log, exp, pi, sqrt, power, tan

# import functions for photon fields
from .photonfields import *
from .nb_utils import multi_5dim_piecewise, multi_5dim_simps, black_body_nb, multi_3dim_piecewise
from .ssc import kpc2cm, eV2erg, eV2Hz, m_e_eV, arcmin2rad, ic_kernel
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import Planck15 as cosmo

from fast_interp.fast_interp import interp2d as fast_interp2d

import time
import logging


def kernel_r(y, x):
    """
    Calculate Kernel from Atoyan & Aharonian Eq. 15
    """
    kernel = y / x * (log(x + y) - log(np.abs(x - y)))
    return kernel


def los_to_radius(theta, d_kpc=2., r_max_pc=1.8, s_steps=100):
    """"
    Calculate the distance to the nebula center (radius)
    for a line of sight distance s and an angular separation theta between
    the line of sight an the nebula center

    Parameters
    ----------
    theta: array-like
        angular separation between line of sight
        and nebula center in arcmin

    d_kpc: float
        distance to the nebula

    r_max_pc: float
        maximum size of the nebula in pc

    s_steps: int
        steps along line of sight within nebula

    Returns
    -------
    A 2d array with the radius values along the line of sight
    for each theta value with shape (`theta.size`, `s_steps`)
    """
    theta_rad = theta * arcmin2rad
    d_cm = d_kpc * kpc2cm

    # derive line of sight that goes through the nebula
    r_max_cm = r_max_pc / 1e3 * kpc2cm

    # maximum theta angle in rad:
    theta_max_rad = np.arctan(r_max_cm / d_cm)
    mask = theta_rad < theta_max_rad

    discriminant = r_max_cm ** 2. - d_cm ** 2. * np.sin(theta_rad[mask]) ** 2.

    s_min_cm = np.zeros_like(theta_rad)
    s_max_cm = np.zeros_like(theta_rad)

    s_min_cm[mask] = d_cm * np.cos(theta_rad[mask]) - np.sqrt(discriminant)
    s_max_cm[mask] = d_cm * np.cos(theta_rad[mask]) + np.sqrt(discriminant)

    # build a line of sight array through the nebula
    # for each theta value
    s_cm = np.full((theta_rad.size, s_steps), np.nan)
    for i in range(theta_rad.size):
        if s_min_cm[i] < s_max_cm[i]:
            s_cm[i] = np.linspace(s_min_cm[i], s_max_cm[i], s_steps)

    tt, _ = np.meshgrid(theta_rad, range(s_steps), indexing='ij')

    r_cm = np.sqrt(d_cm ** 2. + s_cm ** 2. - 2. * d_cm * s_cm * np.cos(tt))

    return r_cm, s_cm


class CrabSSC3D(object):
    def __init__(self, config,
                 n_el,
                 B,
                 d_kpc=2., r0_pc=1.8,
                 nu_sync_min=1e7, nu_sync_max=1e30,
                 integration_mode="scipy_simps",
                 log_level="INFO",
                 use_fast_interp=False):
        """
        Initialize the class

        Parameters
        ----------
        config: str or dict
            path to config file with model parameters.
            Should contain three dictionaries:
            - params_n_el: parameters for the electron density
            - params_n_seed: parameters for the photon density
            - params_B: parameters for the magnetic field

        n_el: function pointer
            electron density spectrum. Should be called with n_el(gamma, r, **params_n_el)

        B: function pointer
            magnetic field of the nebula in G. Should be called with B(r, **params_B)

        d_kpc: float
            distance to the nebula in kpc

        r0_pc: float
            Radius of spherical nebula in pc

        nu_sync_min: float
            minimum frequency considered for syncrotron radiation

        nu_sync_max: float
            maximum frequency considered for syncrotron radiation

        integration_mode: str
            specify how you want to compute your integrals.
            Options are:
            - "scipy_simps" use scipy implementation of simpsons rule
            - "numba_simps" use custom numba implementation of simpsons rule
            - "numba_piecewise" use custom numba implemetation of piecewise multiplication
        """

        # read in config file
        if isinstance(config, dict):
            conf = config
        else:
            with open(config) as f:
                conf = yaml.safe_load(f)

        self._params_n_el = conf['params_n_el']
        self._params_n_seed = conf['params_n_seed']
        self._params_B = conf['params_B']

        self._nu_sync_min = nu_sync_min
        self._nu_sync_max = nu_sync_max
        self._n_el = n_el
        self._B = B
        self._d = d_kpc * kpc2cm
        self._r0 = r0_pc * kpc2cm / 1e3
        self._integration_mode = integration_mode
        self._set_integration_mode()

        self._use_fast_interp = use_fast_interp

        self.__init_logger(log_level)

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
        self._j_sync_interp = None
        self._j_ic_interp = None

        return

    def __init_logger(self, log_level):
        logging.basicConfig(format='\033[0;36m%(filename)10s:\033'
                                   '[0;35m%(lineno)4s\033[0;0m --- %(levelname)7s: %(message)s')
        self._logger = logging.getLogger('SSC_3D')
        self._logger.setLevel(log_level)
        logging.addLevelName(logging.DEBUG, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
        logging.addLevelName(logging.INFO, "\033[1;36m%s\033[1;0m" % logging.getLevelName(logging.INFO))
        logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
        logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

    @property
    def params_n_el(self):
        return self._params_n_el

    @property
    def params_n_seed(self):
        return self._params_n_seed

    @property
    def params_B(self):
        return self._params_B

    @property
    def n_el(self):
        return self._n_el

    @property
    def B(self):
        return self._B

    @property
    def d(self):
        return self._d

    @property
    def r0(self):
        return self._r0

    @property
    def j_sync_interp(self):
        return self._j_sync_interp

    @property
    def integration_mode(self):
        return self._integration_mode

    @property
    def use_fast_interp(self):
        return self._use_fast_interp

    @n_el.setter
    def n_el(self, n_el):
        self._n_el = n_el

    @B.setter
    def B(self, B):
        self._B = B

    @d.setter
    def d(self, d):
        self._d = d

    @r0.setter
    def r0(self, r0):
        self._r0 = r0

    @integration_mode.setter
    def integration_mode(self, integration_mode):
        self._integration_mode = integration_mode
        self._set_integration_mode()

    @use_fast_interp.setter
    def use_fast_interp(self, use_fast_interp):
        self._use_fast_interp = use_fast_interp

    def _set_integration_mode(self):
        """Set the method for integrating multi dimension arrays"""
        if self._integration_mode == "scipy_simps":
            self._integrate_5d = simps
        elif self._integration_mode == "numba_simps":
            self._integrate_5d = multi_5dim_simps
        elif self._integration_mode == "numba_piecewise":
            self._integrate_5d = multi_5dim_piecewise
        else:
            raise ValueError("Unkown integration mode chosen")

    def j_sync(self, nu, r, g_steps=50, gmin=None, gmax=None, g_axis=2):
        """
        Computes the spectral volume emissivity j_nu = j(nu, r) = dE / (dt dV dnu dOmega) for synchrotron radiation
        in erg/s/Hz/cm^3/sr as integral over electron distribution
        at a distance r from the nebula center

        Parameters:
        -----------
        nu: array-like
            frequencies in Hz

        r: array-like
            distance from the nebula center in cm

        {options}

        g_steps: int
            number of integration steps

        gmin: float or None
            minimum lorentz factor

        gmax: float or None
            maximum lorentz factor

        g_axis: int
            The axis of the gamma factor

        Returns:
        --------
        array with spectral luminosity F_nu  density at frequency nu
        """
        if gmin is None:
            gmin = self._params_n_el['gradio_min']
        if gmax is None:
            gmax = self._params_n_el['gwind_max']

        # 3d grid for Freq, gamma factors, and r values
        log_g = linspace(log(gmin), log(gmax), g_steps)
        if len(nu.shape) == 1:
            nnn, rrr, ggg = meshgrid(nu,
                                     r,
                                     log_g,
                                     indexing='ij')
            _, rr = meshgrid(nu,
                             r,
                             indexing='ij')

        elif len(nu.shape) > 1 and np.all(np.equal(nu.shape, r.shape)):
            # stack the nu and r arrays along new g axis
            nnn = np.tile(nu[..., np.newaxis], list(log_g.shape))
            rrr = np.tile(r[..., np.newaxis], list(log_g.shape))
            ggg = np.tile(log_g, list(nu.shape) + [1])

            rr = r
        else:
            raise ValueError("nu and theta have inconsistent shapes")

        # x = nu / nu_c as 2d grid,
        # nu_c: critical frequency for B in G; Longair vol.2 p. 261
        nu_c = 4.199e10 * self._B(rrr, **self._params_B) * u.G.to('T') * exp(ggg)**2.
        x = nnn / nu_c

        # define a mask for integration
        m = (log(x) > self.__start) & (log(x) < self.__end)

        # and for the maximum extension
        m &= rrr <= self._r0
        result = np.full(x.shape, 1e-40)

        # synchrotron function
        result[m] = exp(self.log_xF(log(x[m])))

        # electron spectrum
        # dN / dV d gamma
        result[m] *= self._n_el(exp(ggg[m]), rrr[m], **self._params_n_el)

        # integrate over gamma
        result = simps(result * exp(ggg), ggg, axis=g_axis)

        # pre factors: sqrt(3) * e^3 / mc^2 with B in G, see e.g. B&G 4.44
        # this has then units Fr^3 s^2 B g-1 cm-2
        # When you use Fr G s^2 / (cm g) = 1 you get
        # units Fr^2 / cm and with Fr = cm^3/2 g^1/2 s^-1
        # this becomes g cm^2 s^2 = erg = erg / Hz / s.
        # The pre factor is then consistent with 18.36 in Longair Vol.2
        # since he calculates in W and for B in Tesla.
        result *= ((c.e.esu**3.) / (c.m_e.cgs * c.c.cgs**2.) * sqrt(3.)).value
        # this is equal to 2.344355730864404e-22

        # average over all pitch angles gives 2/3
        result *= self._B(rr, **self._params_B) * sqrt(2.0/3.0)

        # Together with electron spectrum, this has now units
        # erg / Hz / s / cm^3, i.e. is the Volume emissivity
        # Since emission is assumed to be isotropic, divide by 4 pi
        # to get volume emissivity per solid angle
        result /= 4. * pi

        # returns value in unites erg/s/Hz/cm^3/sr
        return result

    def interp_sync_init(self, r_min, r_max, nu_steps=100, g_steps=101, r_steps=50):
        """
        Initialize 2D interpolation of synchrotron emissivity j_nu in erg/s/Hz/cm^3/sr
        over frequency nu and radius  for given electron spectrum in log - log space.
        Sets self.FSyncInterp function pointer.

        Parameters
        ----------
        r_min: float
            minimum radius for interpolation
        r_max: float
            maximum radius for interpolation
        r_steps: int,
            number of steps in radius
        g_steps: int,
            number of integration steps for gamma
        """
        log_nu_intp, log_nu_intp_steps = np.linspace(np.log(self._nu_sync_min),
                                                     np.log(self._nu_sync_max),
                                                     nu_steps, retstep=True)
        r_intp, r_intp_steps = np.linspace(r_min, r_max, r_steps, retstep=True)
        log_nn, rr = np.meshgrid(log_nu_intp, r_intp, indexing='ij')
        j_sync = self.j_sync(np.exp(log_nn), rr, g_steps=g_steps)

        if self._use_fast_interp:
            self._j_sync_interp = fast_interp2d([log_nu_intp[0], r_intp[0]],
                                                [log_nu_intp[-1], r_intp[-1]],
                                                [log_nu_intp_steps, r_intp_steps],
                                                log(j_sync),
                                                k=3,
                                                p=[False, False],
                                                c=[True, True],
                                                e=[0, 0]
                                                )
        else:
            self._j_sync_interp = RectBivariateSpline(log_nu_intp, r_intp, log(j_sync), kx=3, ky=3, s=0)

    def j_grey_body(self, nu, r):
        """
        Return volume emissivity of grey body j_nu erg/s/cm^3/Hz/sr,
        assumes dust component to scale as radial gaussian from nebula center

        Parameters
        ----------
        nu: array like
            array with frequencies in Hz

        r: array-like
            distance from the nebula center in cm


        Returns
        -------
        array with grey body flux in erg/s/cm^2/Hz/sr
        """

        if len(nu.shape) == len(r.shape) == 1:
            nn, rr = np.meshgrid(nu, r, indexing='ij')
        elif len(nu.shape) > 1 and np.all(np.equal(nu.shape, r.shape)):
            nn = nu
            rr = r
        else:
            raise ValueError("nu and theta have inconsistent shapes")

        # photons dens of black body in photons/eV/cm^3
        t0 = time.time()
        if self._use_fast_interp:
            result = np.zeros(nn.flatten().size)
            black_body_nb(result, nn.flatten() / eV2Hz, self._params_n_seed['dust_T'])
            result = result.reshape(nn.shape)
        else:
            result = black_body(nn / eV2Hz, self._params_n_seed['dust_T'])
        t1 = time.time()
        self._logger.debug(f"Black body calculation in grey body function took {t1 - t0:.3f}s")

        # change to dens in photon / Hz / cm^3, dn / d nu = dn / de * de / d nu = dn / de * h
        result *= c.h.to('eV s').value

        # multiply with energy to get energy density per Hz
        result *= nn * c.h.to('erg s').value

        # multiply with c / 4 pi to get energy flux in erg / s / cm^2 / Hz
        result *= c.c.cgs.value / 4. / pi

        # convert to dust luminosity in erg / s / Hz (4 pi cancels out when assuming isotropic emission)
        result *= self._d * self._d * 4. * np.pi

        # multiply with spatial dependence, norm is in units of 1/cm^3
        result *= self._params_n_seed['dust_norm']

        # multiply with gaussian extension
        sigma = tan(self._params_n_seed['dust_extension'] * arcmin2rad) * self._d

        t2 = time.time()
        # using numba here does not buy as anything in time
        #if self._use_fast_interp:
        #    extension = np.zeros(result.flatten().size)
        #    radial_gaussian_nb(extension, rr.flatten(), sigma)
        #    result *= extension.reshape(result.shape)
        #else:
        #    extension = np.exp(-rr ** 2. / 2. / sigma ** 2.)
        #    result *= extension
        result *= np.exp(-rr ** 2. / 2. / sigma ** 2.)
        t3 = time.time()
        self._logger.debug(f"extension calculation in grey body function took {t3 - t2:.3f}s")

        # assume isotropic emission
        result /= 4. * np.pi
        # results in emissivity erg / s / Hz / cm^3 / sr
        return result

    def phot_dens(self, eps, gamma, r, r_steps=100, r1_steps=100):
        """
        Calculate photon number density of Crab nebula according to Hillas et al. (1998)
        for the synchrotron and / or dust compoment

        Parameters
        ----------
        eps: array-like
            n-dim array with energy of photons, in eV
        gamma: array
            m-dim array with gamma factor of electrons
        r: array-like
            angular offset from nebula center in deg
        r_steps: int
            number of steps in radius for synchrotron interpolation


        Returns
        -------
        m x n-dim array with photon densities in photons / eV / cm^3

        Notes
        -----
        See https://arxiv.org/pdf/1008.4524.pdf Eq. (A3)
        """

        t0 = time.time()
        r_max = np.max([r.max(), 1. * self._r0])
        r_min = np.min([r.min(), 1e-5])

        # radius for integration of photon emissivity
        r1 = np.logspace(np.log10(r_min), np.log10(r_max), r1_steps)
        #r1 = np.linspace(r_min, r_max, r1_steps)

        # stack the eps array along new r1 axis
        ee = np.tile(eps[..., np.newaxis], list(r1.shape))
        rr = np.tile(r[..., np.newaxis], list(r1.shape))
        r1r1 = np.tile(r1, list(eps.shape) + [1])
        yy = r1r1 / r_max
        xx = rr / r_max

        # photon emissivity
        #j_nu = np.full(ee.shape, 1e-10, dtype=np.float32)
        j_nu = np.full(ee.shape, 1e-40, dtype=np.float64)
        t1 = time.time()

        if self._params_n_seed['ic_sync']:

            # initialize synchrotron interpolation
            if self._j_sync_interp is None:
                self.interp_sync_init(r_min=r_min, r_max=r_max, r_steps=r1_steps)

            # mask for frequencies
            m = (log(ee * eV2Hz) > log(self._nu_sync_min)) & \
                (log(ee * eV2Hz) < log(self._nu_sync_max))

            # get synchrotron volume emissivity in units of erg/s/cm^3/Hz/sr
            # from log-log interpolation
            if self._use_fast_interp:
                j_nu[m] = np.exp(self._j_sync_interp(
                    log(ee[m] * eV2Hz).flatten(),
                    log(r_max * yy[m]).flatten()
                )).reshape(ee[m].shape)
            else:
                j_nu[m] = np.exp(self._j_sync_interp(log(ee[m] * eV2Hz), log(r_max * yy[m]), grid=False))

            t2 = time.time()
            # conversion to photon emissivity in photons/s/cm^3/eV/sr
            # Now in units of photons/s/Hz/cm^3/sr
            j_nu[m] /= ee[m] * eV2erg

            # convert in units of photons/eV/cm^3/s
            j_nu[m] *= eV2Hz

            # seed photon density at distance r now calculated
            # through integration over r1, see Eq. 15
            # in Atoyan & Aharonian
            kernel = kernel_r(yy, xx)
            kernel *= j_nu

            # seed photon density in photons/eV/cm^3
            self._logger.debug(kernel.shape)
            self._logger.debug(f"Integrating using {self._integration_mode}")
            phot_dens = self._integrate_5d(kernel * yy, log(yy)) * 0.5 * r_max / c.c.cgs.value
            t3 = time.time()

        else:
            phot_dens = np.full(eps.shape, 1e-40)

        if self._params_n_seed['ic_dust']:
            # get dust volume emissivity in units of erg/s/cm^3/Hz/sr
            t01 = time.time()
            j_dust = self.j_grey_body(ee * eV2Hz, yy * r_max)

            # conversion to photon emissivity in photons/s/cm^3/eV/sr
            # Now in units of photons/s/Hz/cm^3/sr
            j_dust /= ee * eV2erg

            # convert in units of photons/eV/cm^3/s/sr
            j_dust *= eV2Hz

            t02 = time.time()
            kernel_dust = j_dust * kernel_r(yy, xx)
            #phot_dens_dust = simps(kernel_dust * yy, log(yy), axis=-1) * r_max * 0.5 / c.c.cgs.value
            phot_dens_dust = self._integrate_5d(kernel_dust * yy, log(yy)) * r_max * 0.5 / c.c.cgs.value
            phot_dens += phot_dens_dust
            t03 = time.time()
            self._logger.debug("time to calculate grey body {0:.3f}s "
                         " time for integration of dust component  {1:.3f}s".format(t02-t01, t03 - t02))

        self._logger.debug("phot_dens: time for interpolation of Sync: {0:.3f}s,"
                           " time for integration of SSC component  {1:.3f}s, "
                           " time for filling arrays {2:.3f}s ".format(t2-t1, t3 - t2, t1 - t0))



        return phot_dens

    def j_ic(self, nu, r, g_steps=200, e_steps=90, r1_steps=100):
        """
        Spectral luminosity F_nu in erg/s/Hz/cm^2 for inverse Compton scattering.

        Parameters:
        -----------
        nu: array-like
            n-dim array with frequencies in Hz
        r: array-like
            distance from the nebula center in cm

        {options}

        g_steps: int
            number of integration steps for gamma
        e_steps: int
            number of integration steps for energy

        Returns:
        --------
        n-dim numpy array spectral luminosity F_nu  density at frequency nu
        """

        t0 = time.time()
        log_g = linspace(log(self._params_n_el['gmin']), log(self._params_n_el['gmax']), g_steps)
        gamma = exp(log_g)

        # generate the arrays for observed freq nu, gamma factor, radius, and energy of photon field
        nnn, ggg, _, rrr = meshgrid(nu, log_g, np.arange(e_steps), r, indexing='ij')
        _, gg, rr = meshgrid(nu, log_g, r, indexing='ij')
        nn, _ = meshgrid(nu, r, indexing='ij')

        log_eee = zeros(nnn.shape)

        m = zeros(nnn.shape, dtype=np.bool)
        for i, n in enumerate(nu):
            for j, lg in enumerate(log_g):
                x1 = max(log(n / eV2Hz / 4. / gamma[j] ** 2.), log(1e-18))
                x2 = log(n / eV2Hz)
                # now log_eps has shape g_steps x e_steps
                log_eee[i, j], _ = np.meshgrid(linspace(x1, x2, e_steps), r, indexing='ij')
                if x2 > x1:
                    m[i, j] = True

        t1 = time.time()
        # calculate photon densities:
        # these are in photons / eV / cm^3
        if self._params_n_seed['ic_sync'] or self._params_n_seed['ic_dust']:
            phot_dens = self.phot_dens(exp(log_eee), exp(ggg), rrr, r1_steps=r1_steps)
        else:
            phot_dens = np.full(log_eee.shape, 1e-40)

        if self._params_n_seed['ic_cmb']:
            phot_dens += black_body(exp(log_eee), cosmo.Tcmb0.value)

        t2 = time.time()
        phot_dens[~m] = 1e-40
        m_isnan = np.isnan(phot_dens)
        phot_dens[m_isnan] = 1e-40

        # IC scattering kernel
        f = ic_kernel(nnn, exp(ggg), exp(log_eee))

        # multiply the two in integrate over initial photon energy
        kernel_in = phot_dens * f

        # kernel needs to be divided by exp(log_eee) but
        # cancels since we're integrating over log(energy).
        # now in photons / cm^3 / eV / cm^3
        self._logger.debug(f"kernel shape for integration over photon dens energy: {kernel_in.shape}")
        kernel_out = simps(kernel_in, log_eee, axis=2)
        kernel_out *= self._n_el(exp(gg), rr, **self._params_n_el) / exp(gg) ** 2.

        # integrate over electron gamma factor
        self._logger.debug(f"kernel shape for integration over gamma factor: {kernel_out.shape}")
        result = simps(kernel_out * exp(gg), gg, axis=1)

        # result of integration is in units of photons/cm^3/eV/cm^3
        # multiplying with Thomson*c*energy converting units gives to
        # units of erg/sec/eV/cm^3
        result *= 3. / 4. * (c.sigma_T.cgs * c.c.cgs).value * nn / eV2Hz * eV2erg
        # convert to erg / sec / Hz / cm^3
        # this is the spectral luminosity L_nu
        result /= eV2Hz
        #  divide by 4 pi to get
        # volume emissitivity in units of erg / sec / Hz / cm^3 / sr
        result /= 4. * pi
        t3 = time.time()
        self._logger.debug(f"integration over photon dens and gamma took {t3-t2:.3f}s")

        # interpolate result
        self._j_ic_interp = RectBivariateSpline(log(nu), r, log(result), kx=3, ky=3, s=0)

        return result


    def theta_max_arcmin(self):
        """
        Calculate the maximum angular extension of the nebula
        in arcmin
        """
        theta_max = np.tan(self._r0 / self._d) / arcmin2rad
        return theta_max

    def intensity(self, nu, theta, which='sync', r_steps=100, r_min=0., **kwargs):
        """
        Compute the specific intensity for the synchrotron emission
        as a line of sight integral over the volume emissivity
        for different angular separations theta

        Parameters
        ----------
        nu: array-like
            Array with frequencies in Hz

        theta: int or array-like
            Angular separation for which intensity is calculated.
            If integer, use linear spacing between 0 and angular separation
            of r0 (given distance d).
            If array, it gives the angular separation in arcmin

        r_steps: int
            steps used for integration over r

        which: str
            specify for which radiation you want to calculate the
            intensity. Either 'sync', 'ic', or 'dust'

        kwargs: dict
            options passed to either interp_sync_init, j_ic, or j_grey_body
            depending on value of 'which'

        Returns
        -------
        2D array with intensity as function of frequency nu and theta
        and the theta array
        """
        if isinstance(theta, int):
            theta_arcmin = np.linspace(0., self.theta_max_arcmin() * 0.99, theta)
        else:
            theta_arcmin = theta

        # build the array over radius
        r = np.linspace(r_min, self._r0, r_steps)

        tt, rr = np.meshgrid(theta_arcmin, r, indexing='ij')
        _, tntn = np.meshgrid(nu, theta_arcmin, indexing='ij')

        # build 3D arrays over nu, theta, r
        nnn = np.tile(nu[:, np.newaxis, np.newaxis], list(rr.shape))
        rrr = np.rollaxis(np.tile(rr[..., np.newaxis], list(nu.shape)), -1)
        ttt = np.rollaxis(np.tile(tt[..., np.newaxis], list(nu.shape)), -1)

        # build the array over the line of sight
        # within the nebula
        sss = (self._d - rrr) / np.cos(ttt * arcmin2rad)

        # the interpolation of the emissivity
        # is over nu and r
        # thus, for each point along the line of sight
        # you need to calculate the corresponding distance to the nebula center
        xxx = np.sqrt(sss ** 2. + self._d ** 2. - 2. * self._d * sss * np.cos(ttt * arcmin2rad))

        # get the volume emissivity
        if which == 'sync':
            if self._j_sync_interp is None:
                self.interp_sync_init(r_min=r.min(), r_max=r.max(), **kwargs)

            if self._use_fast_interp:
                j = np.exp(self._j_sync_interp(np.log(nnn), xxx))
            else:
                j = np.exp(self._j_sync_interp(np.log(nnn), xxx, grid=False))

        elif which == 'dust':
            j = self.j_grey_body(nnn, xxx, **kwargs)

        elif which == 'ic':
            if self._j_sync_interp is None:
                self.j_ic(nu, r, **kwargs)
            j = np.exp(self._j_ic_interp(np.log(nnn), xxx, grid=False))

        else:
            raise ValueError("Unknown value provided for 'which'. Options are 'sync', 'dust', or 'ic'")

        # integrate
        # Since j_nu is in ergs / s / cm^3 / sr / Hz
        # the result is now in ergs / s / cm^2 / sr /Hz
        # the extra cos factor comes from the substitution from s to r
        I_nu = simps(j, rrr, axis=-1) / np.cos(tntn * arcmin2rad)
        return I_nu, theta_arcmin


