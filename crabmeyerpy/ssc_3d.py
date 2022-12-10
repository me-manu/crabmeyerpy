import yaml
from scipy.special import kv  # Bessel function
from scipy.integrate import simps, romb
from scipy.interpolate import RectBivariateSpline, interp1d

# imports to speed up integrations:
from numpy import meshgrid, linspace, zeros
from numpy import log, exp, pi, sqrt, tan

# import functions for photon fields
from .photonfields import *
from .nb_utils import multi_5dim_piecewise, multi_5dim_simps, black_body_nb, multi_5dim_romb
from .ssc import kpc2cm, eV2erg, eV2Hz, m_e_eV, arcmin2rad, ic_kernel
from astropy import constants as c
from astropy.cosmology import Planck15 as cosmo

from fast_interp.fast_interp import interp2d as fast_interp2d

import time
import logging
import warnings

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
                 j_dust,
                 d_kpc=2., r0_pc=3.6,
                 nu_sync_min=5e8, nu_sync_max=5e23,
                 integration_mode="scipy_simps",
                 log_level="INFO",
                 ic_sync=True,
                 ic_dust=True,
                 ic_cmb=True,
                 dust_radial_dependence='shell',
                 use_fast_interp=False):
        """
        Initialize the class

        Parameters
        ----------
        config: str or dict
            path to config file with model parameters.

        n_el: function pointer
            electron density spectrum. Should be called with n_el(gamma, r, **config)

        B: function pointer
            magnetic field of the nebula in G. Should be called with B(r, **config)

        j_dust: function pointer
            volume emissivity of the dust in the nebula in erg / s / cm^3 / Hz / sr.
            Should be called with j_dust(r, nu, **config)

        d_kpc: float
            distance to the nebula in kpc

        r0_pc: float
            Radius of spherical nebula in pc

        nu_sync_min: float
            minimum frequency considered for syncrotron radiation

        nu_sync_max: float
            maximum frequency considered for syncrotron radiation

        ic_sync: bool
            if True, include synchrotron radiation as seed field for inverse Compton scattering

        ic_dust: bool
            if True, include dust emission as seed field for inverse Compton scattering

        ic_cmb: bool
            if True, include CMB as seed field for inverse Compton scattering

        dust_radial_dependence: str
            Radial dependence of dust, either 'gauss' or 'const'
            Note: this will be removed soon!

        integration_mode: str
            specify how you want to compute your integrals.
            Options are:
            - "scipy_simps" use scipy implementation of simpsons rule
            - "numba_simps" use custom numba implementation of simpsons rule
            - "numba_piecewise" use custom numba implemetation of piecewise multiplication
        """

        # read in config file
        if isinstance(config, dict):
            self._parameters = config
        else:
            with open(config) as f:
                self._parameters = yaml.safe_load(f)

        self._ic_sync = ic_sync
        self._ic_dust = ic_dust
        self._ic_cmb = ic_cmb

        # TODO: remove
        self._dust_radial_dependence = dust_radial_dependence

        self._nu_sync_min = nu_sync_min
        self._nu_sync_max = nu_sync_max
        self._n_el = n_el
        self._B = B
        self._j_dust = j_dust
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

        # using original formulation with Bessel function of order 5 / 3:
        #for i, s in enumerate(logx):
            #if not i:
                #logx_arr = np.linspace(s, self.__end, steps)
            #else:
                #logx_arr = np.vstack((logx_arr, np.linspace(s, self.__end, steps)))

        #xF = np.exp(logx) * simps(kv(5./3., np.exp(logx_arr)) * np.exp(logx_arr), logx_arr, axis=1)
        # however, one should take into account that the B field is random
        # and hence this needs to be averaged over pitch angle, see, e.g., Appendix D
        # of https://arxiv.org/pdf/1006.1045.pdf
        x = np.exp(logx)
        xF = (8. + 3. * x ** 2.) * kv(1./3., x / 2.) **2.
        xF += x * kv(2./3.,  x / 2.) * (2. * kv(1./3., x / 2.) - 3. * x * kv(2./3., x / 2.))
        xF *= x / 20.
        xF[xF < 1e-40] = np.full(np.sum(xF < 1e-40), 1e-40)
        self.log_xF = interp1d(logx, np.log(xF))
        self._j_sync_interp = None
        self._j_sync_interp_object = None
        self._j_ic_interp = None
        self._phot_dens_sync_interp = None

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
    def parameters(self):
        return self._parameters

    @property
    def ic_sync(self):
        return self._ic_sync

    @property
    def ic_cmb(self):
        return self._ic_cmb

    @property
    def ic_dust(self):
        return self._ic_dust

    @property
    def dust_radial_dependence(self):
        """
        Deprecated, only kept for checks!
        """
        # TODO: remove
        return self._dust_radial_dependence

    @property
    def n_el(self):
        return self._n_el

    @property
    def B(self):
        return self._B

    @property
    def j_dust(self):
        return self._j_dust

    @property
    def d(self):
        return self._d

    @property
    def r0(self):
        if 'wind_size_cm' in self.parameters:
            # 3sigma of a 2D Gaussian contains ~0.997^2 = 99.4% of the flux
            # could also do 4 sigma but the error of the larger interp grid should be similar
            return max(self.parameters['wind_size_cm'],self.parameters['radio_size_cm'])*3
        else: return self._r0

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

    @j_dust.setter
    def j_dust(self, j_dust):
        self._j_dust = j_dust

    @d.setter
    def d(self, d):
        self._d = d

#     @r0.setter
#     def r0(self, r0):
#         self._r0 = r0

    @integration_mode.setter
    def integration_mode(self, integration_mode):
        self._integration_mode = integration_mode
        self._set_integration_mode()

    @use_fast_interp.setter
    def use_fast_interp(self, use_fast_interp):
        self._use_fast_interp = use_fast_interp

    @dust_radial_dependence.setter
    def dust_radial_dependence(self, dust_radial_dependence):
        self._dust_radial_dependence = dust_radial_dependence

    def _set_integration_mode(self):
        """Set the method for integrating multi dimension arrays"""
        if self._integration_mode == "scipy_simps":
            self._integrate_5d = simps
        elif self._integration_mode == "numba_simps":
            self._integrate_5d = multi_5dim_simps
        elif self._integration_mode == "numba_piecewise":
            self._integrate_5d = multi_5dim_piecewise
        elif self._integration_mode == "numba_romb":
            self._integrate_5d = multi_5dim_romb
        elif self._integration_mode == "romb":
            self._integrate_5d = romb
        else:
            raise ValueError("Unknown integration mode chosen")

    @ic_sync.setter
    def ic_sync(self, ic_sync):
        self._ic_sync = ic_sync

    @ic_cmb.setter
    def ic_cmb(self, ic_cmb):
        self._ic_cmb = ic_cmb

    @ic_dust.setter
    def ic_dust(self, ic_dust):
        self._ic_dust = ic_dust

    def j_sync(self, nu, r, g_steps=65, gmin=None, gmax=None, g_axis=2, integration_mode='simps'):
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
            gmin = self._parameters['gmin']
        if gmax is None:
            gmax = self._parameters['gmax']

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
        # nu_c = 3 e B gamma^2 / 4 pi m c in cgs units, see B&G Eq. 4.32
        # With e in Fr this has units Fr G s g^-1 cm^-1
        # Using that Fr G s^2 cm^-1 g^-1 = 1 we can see that nu_c is in s^-1 = Hz
        # This is encoded in pre_factor:
        pre_factor = (c.e.esu * 3. / 4. / c.m_e.cgs / c.c.cgs / np.pi).value
        # pre_factor is equal to 4.199e6 for B in G
        # this is consistent with Longair vol.2 p. 261 who gets 4.199e10 with B in T
        nu_c = pre_factor * self._B(rrr, **self._parameters) * exp(ggg)**2.
        x = nnn / nu_c

        # define a mask for integration
        m = (log(x) > self.__start) & (log(x) < self.__end)

        # and for the maximum extension
        m &= rrr <= self.r0
        result = np.full(x.shape, 1e-80)

        # synchrotron function
        result[m] = exp(self.log_xF(log(x[m])))

        # electron spectrum
        # dN / dV d gamma
        result[m] *= self._n_el(exp(ggg[m]), rrr[m], **self._parameters)

        # integrate over gamma
        if integration_mode == 'romb':
            result = romb(result * exp(ggg), dx=np.diff(log_g)[0], axis=g_axis)
        else:
            result = simps(result * exp(ggg), ggg, axis=g_axis)

        # pre factors: sqrt(3) * e^3 / mc^2 with B in G, see e.g. B&G 4.44
        # this has then units Fr^3 s^2 B g-1 cm-2
        # When you use Fr G s^2 / (cm g) = 1 you get
        # units Fr^2 / cm and with Fr = cm^3/2 g^1/2 s^-1
        # this becomes g cm^2 s^2 = erg = erg / Hz / s.
        # The pre factor is then consistent with Eq. (18.36) in Longair Vol.2
        # since he calculates in W and for B in Tesla.
        result *= ((c.e.esu**3.) / (c.m_e.cgs * c.c.cgs**2.) * sqrt(3.)).value
        # this is equal to 2.344355730864404e-22

        # multiply with magnetic field
        # note that there's no factor sqrt(2/3) anymore,
        # this is taken care in the new sync function, where the averaging
        # over random B field enters correctly.
        # New sync function is in __init__ function
        result *= self._B(rr, **self._parameters)


        # Together with electron spectrum, this has now units
        # erg / Hz / s / cm^3, i.e. is the Volume emissivity
        # Since emission is assumed to be isotropic, divide by 4 pi
        # to get volume emissivity per solid angle
        result /= 4. * pi

        # returns value in unites erg/s/Hz/cm^3/sr
        return result

    def interp_sync_init(self, r_min, r_max,
                         gmin=None,
                         gmax=None,
                         nu_steps=100,
                         g_steps=129,
                         r_steps=80,
                         integration_mode='simps'):
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
        gmin: float or None
            minimum lorentz factor
        gmax: float or None
            maximum lorentz factor
        r_steps: int,
            number of steps in radius
        g_steps: int,
            number of integration steps for gamma
        interpolation_grid_r: str
            either 'log' or 'lin' for interpolation over logarithmic
            or linear grid points
        """
        log_nu_intp, log_nu_intp_steps = np.linspace(np.log(self._nu_sync_min),
                                                     np.log(self._nu_sync_max),
                                                     nu_steps, retstep=True)
        r_intp, r_intp_steps = np.linspace(r_min, r_max, r_steps, retstep=True)

        j_sync = self.j_sync(np.exp(log_nu_intp), r_intp,
                             gmin=gmin,
                             gmax=gmax,
                             g_steps=g_steps,
                             integration_mode=integration_mode)

        if self._use_fast_interp:
            self._j_sync_interp_object = fast_interp2d([log_nu_intp[0], r_intp[0]],
                                                       [log_nu_intp[-1], r_intp[-1]],
                                                       [log_nu_intp_steps, r_intp_steps],
                                                       log(j_sync),
                                                       k=1,
                                                       p=[False, False],
                                                       c=[True, True],
                                                       e=[0, 0]
                                                       )
        else:
            self._j_sync_interp_object = RectBivariateSpline(log_nu_intp, r_intp, log(j_sync), kx=1, ky=1, s=0)

        if self._use_fast_interp:
            self._j_sync_interp = lambda log_nu, r: self._j_sync_interp_object(log_nu, r)
        else:
            self._j_sync_interp = lambda log_nu, r: self._j_sync_interp_object(log_nu, r, grid=False)

    def j_dust_nebula(self, nu, r):
        """
        Return volume emissivity of grey body j_nu erg/s/cm^3/Hz/sr,

        Parameters
        ----------
        nu: array like
            array with frequencies in Hz

        r: array-like
            distance from the nebula center in cm

        Returns
        -------
            array with grey body flux in erg/s/cm^3/Hz/sr
        """

        if len(nu.shape) == len(r.shape) == 1:
            nn, rr = np.meshgrid(nu, r, indexing='ij')
        elif len(nu.shape) > 1 and np.all(np.equal(nu.shape, r.shape)):
            nn = nu
            rr = r
        else:
            raise ValueError("nu and theta have inconsistent shapes")

        # Get the emissivity
        t0 = time.time()
        result = self._j_dust(rr, nn, **self._parameters)
        t1 = time.time()
        self._logger.debug(f"Dust calculation in j_dust_nebula function took {t1 - t0:.3f}s")
        # results in emissivity erg / s / Hz / cm^3 / sr
        return result

    def j_grey_body(self, nu, r):
        """
        Return volume emissivity of grey body j_nu erg/s/cm^3/Hz/sr,
        assumes dust component to scale as radial gaussian from nebula center

        Deprecated, only kept for checks!

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
        # TODO: remove

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
            black_body_nb(result, nn.flatten() / eV2Hz, self._parameters['dust_T'])
            result = result.reshape(nn.shape)
        else:
            result = black_body(nn / eV2Hz, self._parameters['dust_T'])
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
        result *= self._parameters['dust_norm']

        # multiply with gaussian extension
        sigma = tan(self._parameters['dust_extension'] * arcmin2rad) * self._d

        t2 = time.time()

        if self._dust_radial_dependence == 'gauss':
            result *= np.exp(-rr ** 2. / 2. / sigma ** 2.)

        elif self._dust_radial_dependence == 'shell':
            rmin = tan(self._parameters['min_dust_extension'] * arcmin2rad) * self._d
            mask = (rr <= sigma) & (rr >= rmin)
            # divide by dust volume, i.e., normalization parameter dust_norm is unit less in this case
            volume = 4. / 3. * np.pi * (sigma ** 3. - rmin ** 3.)
            result /= volume
            result[~mask] = 0.

        t3 = time.time()
        self._logger.debug(f"extension calculation in grey body function took {t3 - t2:.3f}s")

        # assume isotropic emission
        result /= 4. * np.pi
        # results in emissivity erg / s / Hz / cm^3 / sr
        return result

    def phot_dens(self, eps, r, r1_steps=33):
        """
        Calculate photon number density of Crab nebula according to Hillas et al. (1998)
        for the synchrotron and / or dust compoment

        Parameters
        ----------
        eps: array-like
            n-dim array with energy of photons, in eV
        r: array-like
            angular offset from nebula center in deg
        r1_steps: int
            number of steps in radius for integration


        Returns
        -------
        m x n-dim array with photon densities in photons / eV / cm^3

        Notes
        -----
        See https://arxiv.org/pdf/1008.4524.pdf Eq. (A3)
        """

        t0 = time.time()
        r_max = r.max()*0.99 # move 1% out and in of the edges so the r and r1 array
        r_min = r.min()*1.01 # do not have the same value (divergence in the kernel)

        ee, xx, y, yy = self._get_integration_arrays(eps, r, r1_steps, r_max, r_min)

        # photon emissivity
        #j_nu = np.full(ee.shape, 1e-10, dtype=np.float32)
        j_nu = np.full(ee.shape, 1e-80, dtype=np.float64)
        t1 = time.time()

        if self._ic_sync:

            # for KC model: emissitivity is 0 for r < r0,
            # change r1 integration array to accomodate for this
            if 'r_shock' in self._parameters:
                _, xx, y, yy = self._get_integration_arrays(eps, r, r1_steps, r_max, self._parameters['r_shock'])

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
                    (r_max * yy[m]).flatten()
                    #log(r_max * yy[m]).flatten()
                )).reshape(ee[m].shape)
            else:
                j_nu[m] = np.exp(self._j_sync_interp(log(ee[m] * eV2Hz),
                                                     #log(r_max * yy[m]),
                                                     r_max * yy[m],
                                                     ))

            t2 = time.time()
            # conversion to photon emissivity in photons/s/cm^3/eV/sr
            # Now in units of photons/s/Hz/cm^3/sr
            j_nu[m] /= ee[m] * eV2erg

            # convert in units of photons/eV/cm^3/s/sr
            j_nu[m] *= eV2Hz

            # assume isotropic emissivity
            # and convert to photons/eV/cm^3/s
            j_nu[m] *= 4. * np.pi

            # seed photon density at distance r now calculated
            # through integration over r1, see Eq. 15
            # in Atoyan & Aharonian
            kernel = kernel_r(yy, xx)
            kernel *= j_nu

            # seed photon density in photons/eV/cm^3
            self._logger.debug(kernel.shape)
            self._logger.debug(f"Integrating using {self._integration_mode}")
            if self._integration_mode == 'romb' or not len(yy.shape) == 5:
                phot_dens = romb(kernel, dx=np.diff(y)[0], axis=-1)
            else:
                phot_dens = self._integrate_5d(kernel, yy)
            phot_dens *= 0.5 * r_max / c.c.cgs.value
            t3 = time.time()

            self._logger.debug("phot_dens: time for interpolation of Sync: {0:.3f}s,"
                               " time for integration of SSC component  {1:.3f}s, "
                               " time for filling arrays {2:.3f}s ".format(t2-t1, t3 - t2, t1 - t0))

        else:
            phot_dens = np.full(eps.shape, 1e-80)

        if self._ic_dust:
            # get dust volume emissivity in units of erg/s/cm^3/Hz/sr
            t01 = time.time()
            j_dust = self.j_dust_nebula(ee * eV2Hz, yy * r_max)

            # conversion to photon emissivity in photons/s/cm^3/eV/sr
            # Now in units of photons/s/Hz/cm^3/sr
            j_dust /= ee * eV2erg

            # convert in units of photons/eV/cm^3/s/sr
            j_dust *= eV2Hz

            # assume isotropic emissivity
            # and convert to photons/eV/cm^3/s
            j_dust *= 4. * np.pi

            t02 = time.time()

            # if KC model: change back
            # x and y arrays for dust component,
            # since dust is assumed to be present even in r < r_shock
            if 'r_shock' in self._parameters:
                _, xx, y, yy = self._get_integration_arrays(eps, r, r1_steps, r_max, r_min)

            kernel_dust = j_dust * kernel_r(yy, xx)

            #phot_dens_dust = simps(kernel_dust * yy, log(yy), axis=-1) * r_max * 0.5 / c.c.cgs.value
            if self._integration_mode == 'romb' or not len(yy.shape) == 5:
                phot_dens_dust = romb(kernel_dust, dx=np.diff(y)[0], axis=-1)
            else:
                phot_dens_dust = self._integrate_5d(kernel_dust, yy)
            phot_dens_dust *= r_max * 0.5 / c.c.cgs.value

            phot_dens += phot_dens_dust
            t03 = time.time()
            self._logger.debug("time to calculate grey body {0:.3f}s "
                         " time for integration of dust component  {1:.3f}s".format(t02-t01, t03 - t02))

        return phot_dens

    @staticmethod
    def _get_integration_arrays(eps, r, r1_steps, r_max, r_min):
        """
        Set up integration arrays for integrations performed
        in photon density calculation
        """

        # radius for integration of photon emissivity
#         r1 = np.logspace(np.log10(r_min), np.log10(r_max), r1_steps)
        r1 = np.linspace(r_min, r_max, r1_steps) # linear r grid works much better
        # stack the eps array along new r1 axis

        ee = np.tile(eps[..., np.newaxis], list(r1.shape))
        rr = np.tile(r[..., np.newaxis], list(r1.shape))
        r1r1 = np.tile(r1, list(eps.shape) + [1])
        yy = r1r1 / r_max
        xx = rr / r_max
        y = r1 / r_max
        return ee, xx, y, yy

    def j_ic(self, nu, r, g_steps=129, e_steps=129, r1_steps=33, integration_mode='simps', test=0):
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
        log_g = linspace(log(self._parameters['gmin']), log(self._parameters['gmax']), g_steps)
        gamma = exp(log_g)

        # generate the arrays for observed freq nu, gamma factor, radius, and energy of photon field
        nnn, ggg, _ = meshgrid(nu, log_g, np.arange(e_steps), indexing='ij')
        nnn=nnn[...,None] # ic kernel
        ggg=ggg[...,None] # ic kernel
        rrr=np.tile(r,(nu.size,g_steps,e_steps,1)) # phot dens
        _, gg, rr = meshgrid(nu, log_g, r, indexing='ij') # n_el
        
        x1=log(nu[None,:] / eV2Hz / 4. / gamma[:,None] ** 2.)
        x1 = np.maximum(x1,log(1e-18))
        x2 = log(nu[None,:] / eV2Hz)
        log_eee1=linspace(x1, x2, e_steps).T[...,None] # for ic_kernel
        log_eee=np.tile(log_eee1, (1,1,1,r.size))  # for photon dens
        m = x2 > x1
        
        t1 = time.time()  # lower dimensions and no for loop (3s-->1s)

        # calculate photon densities:
        # these are in photons / eV / cm^3
        # first calculate them on a grid, then interpolate
        if self._ic_sync or self._ic_dust:
            log_EeV_grid, log_EeV_grid_step = np.linspace(log_eee.min(), log_eee.max(), e_steps, retstep=True)
            r_phot_dens, r_phot_dens_step = np.linspace(r.min(), r.max(), r.size, retstep=True)
            log_ee_grid, rr_phot_dens = np.meshgrid(log_EeV_grid, r_phot_dens, indexing='ij')
            logging.debug("Interpolating photon density")

            phot_dens_grid = self.phot_dens(exp(log_ee_grid), rr_phot_dens, r1_steps=r1_steps)
            if test==2:
                return phot_dens_grid, exp(log_ee_grid), rr_phot_dens

            if self._use_fast_interp:
                log_phot_dens_interp = fast_interp2d([log_EeV_grid[0], r_phot_dens[0]],
                                                     [log_EeV_grid[-1], r_phot_dens[-1]],
                                                     [log_EeV_grid_step, r_phot_dens_step],
                                                     log(phot_dens_grid),
                                                     k=1,
                                                     p=[False, False],
                                                     c=[True, True],
                                                     e=[0, 0]
                                                     )
                phot_dens = exp(log_phot_dens_interp(log_eee, rrr))
            else:
                log_phot_dens_interp = RectBivariateSpline(log_EeV_grid, r_phot_dens, log(phot_dens_grid),
                                                           kx=1, ky=1, s=0)
                phot_dens = exp(log_phot_dens_interp(log_eee, rrr, grid=False))

        else:
            phot_dens = np.full(log_eee.shape, 1e-40)

        if self._ic_cmb:
            phot_dens += black_body(exp(log_eee), cosmo.Tcmb0.value)

        t2 = time.time()
        phot_dens[~m.T] = 1e-40
        m_isnan = np.isnan(phot_dens)
        phot_dens[m_isnan] = 1e-40

        # IC scattering kernel
        f = ic_kernel(nnn, exp(ggg), exp(log_eee1))

        # multiply the two in integrate over initial photon energy
        kernel_in = phot_dens * f

        self._logger.debug(f"kernel shape for integration over photon dens energy: {kernel_in.shape}")
        # kernel needs to be divided by exp(log_eee) but
        # cancels since we're integrating over log(energy).
        # only doing this with simps integration since
        # log_eee spacing is not constant
        kernel_out = simps(kernel_in, log_eee, axis=2)
        # now in photons / cm^3 / eV / cm^3  (since n_phot / epsilon in cm^-3 eV^-2)
        kernel_out = kernel_out*self._n_el(exp(gg), rr, **self._parameters) / exp(gg) ** 2.

        # integrate over electron gamma factor
        self._logger.debug(f"kernel shape for integration over gamma factor: {kernel_out.shape}")
        if integration_mode == 'romb':
            result = romb(kernel_out * exp(gg), dx=np.diff(log_g)[0], axis=1)
        else:
            result = simps(kernel_out * exp(gg), gg, axis=1)

        # result of integration is in units of photons/cm^3/eV/cm^3
        # multiplying with Thomson*c*energy converting units gives to
        # units of erg/sec/eV/cm^3
        result *= 3. / 4. * (c.sigma_T.cgs * c.c.cgs).value * nu[:,None] / eV2Hz * eV2erg
        # convert to erg / sec / Hz / cm^3
        # this is the spectral luminosity L_nu
        result /= eV2Hz
        #  divide by 4 pi to get
        # volume emissitivity in units of erg / sec / Hz / cm^3 / sr
        result /= 4. * pi
        t3 = time.time()
        self._logger.debug(f"integration over photon dens and gamma took {t3-t2:.3f}s")

        # interpolate result
        result[result <= 0] = 1e-100
        #self._j_ic_interp = RectBivariateSpline(log(nu), log(r), log(result), kx=3, ky=3, s=0)
        self._j_ic_interp = RectBivariateSpline(log(nu), r, log(result), kx=1, ky=1, s=0)
        if test==1:
            return log_eee, rrr, nnn, ggg, phot_dens ,f, m, m_isnan
        
        return result

    def theta_max_arcmin(self):
        """
        Calculate the maximum angular extension of the nebula
        in arcmin
        """
        theta_max = np.arctan(self.r0 / self._d) / arcmin2rad
        return theta_max

    def intensity(self, nu, theta,
                  which='sync',
                  r_steps=129,
                  r_min=0.,
                  integration_mode='simps',
                  **kwargs):
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
            options passed to either interp_sync_init, j_ic, or j_dust_nebula
            depending on value of 'which'

        Returns
        -------
        2D array with intensity as function of frequency nu and theta
        the theta array, and the volume emissivity array
        """
        if isinstance(theta, int):
            theta_arcmin = np.linspace(0., self.theta_max_arcmin() * 0.99, theta)
        else:
            theta_arcmin = theta

        if log:
            r = np.logspace(np.log10(r_min), np.log10(self.r0), r_steps)
        else:
            r = np.linspace(r_min, self.r0, r_steps)

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

            j = np.exp(self._j_sync_interp(np.log(nnn), xxx))

        elif which == 'dust':
            j = self.j_dust_nebula(nnn, xxx, **kwargs)

        elif which == 'ic':
            if self._j_ic_interp is None:
                self.j_ic(nu, r, **kwargs)
            #j = np.exp(self._j_ic_interp(np.log(nnn), np.log(xxx), grid=False))
            j = np.exp(self._j_ic_interp(np.log(nnn), xxx, grid=False))

        else:
            raise ValueError("Unknown value provided for 'which'. Options are 'sync', 'dust', or 'ic'")

        # integrate
        # Since j_nu is in ergs / s / cm^3 / sr / Hz
        # the result is now in ergs / s / cm^2 / sr /Hz
        # the extra cos factor comes from the substitution from s to r
        if integration_mode == 'romb':
            I_nu = romb(j, dx=np.diff(r)[0], axis=-1) / np.cos(tntn * arcmin2rad)
        else:
            I_nu = simps(j, rrr, axis=-1) / np.cos(tntn * arcmin2rad)
        return I_nu, theta_arcmin, j

    def intensity2(self, nu, theta,
                   which='sync',
                   r_steps=129,
                   r_min=0.,
                   integration_mode='simps',
                   **kwargs):
        """
        Compute the specific intensity
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

        r_min: float
            minimum radius considered for line of sight integration

        which: str
            specify for which radiation you want to calculate the
            intensity. Either 'sync', 'ic', or 'dust'

        integration_mode: str
            specify which integration method to use; either 'simps' or 'romb'

        kwargs: dict
            options passed to either interp_sync_init, j_ic, or j_dust_nebula
            depending on value of 'which'

        Returns
        -------
        2D array with intensity as function of frequency nu and theta
        the theta array, and the volume emissivity array
        """
        if isinstance(theta, int):
            theta_arcmin = np.linspace(0., self.theta_max_arcmin() * 0.999, theta)
        else:
            if np.max(theta) > self.theta_max_arcmin():
                raise ValueError(f"Can't go further than {self.theta_max_arcmin():.2f}arcmin. theta max was requested at {np.max(theta):.2f}arcmin.")
            theta_arcmin = theta

        # build the integration array
        r = np.linspace(r_min, self.r0, r_steps)
        ## s are linearly spaced values along the LoS
        ## x are the radii from s to the centre
        x_min_sq = (np.sin(theta_arcmin*arcmin2rad)*self._d)**2
        with np.errstate(invalid='ignore'):
            s_int_min = np.nan_to_num(np.sqrt(r_min**2 - x_min_sq))
            s_int_max = np.nan_to_num(np.sqrt(self.r0**2 - x_min_sq))
        ss = np.linspace(s_int_min, s_int_max, r_steps).T  # r_steps because s^=r
        xx = np.sqrt(x_min_sq[:,np.newaxis] + ss**2)
                                      
        nnn = np.tile(nu[:, np.newaxis, np.newaxis], list(xx.shape))
        xxx = np.rollaxis(np.tile(xx[..., np.newaxis], list(nu.shape)), -1)

        # get the volume emissivity
        if which == 'sync':
            if self._j_sync_interp is None:
                self.interp_sync_init(r_min=r.min(), r_max=r.max(), **kwargs)

#             xxx[xxx == 0.] = 1e-10
            j = np.exp(self._j_sync_interp(np.log(nnn), xxx))

        elif which == 'dust':
            j = self.j_dust_nebula(nnn, xxx, **kwargs)

        elif which == 'ic':
            if self._j_ic_interp is None:
                self.j_ic(nu, r, **kwargs)
            #j = np.exp(self._j_ic_interp(np.log(nnn), np.log(xxx), grid=False))
            j = np.exp(self._j_ic_interp(np.log(nnn), xxx, grid=False))

        else:
            raise ValueError("Unknown value provided for 'which'. Options are 'sync', 'dust', or 'ic'")

        # integrate
        # Since j_nu is in ergs / s / cm^3 / sr / Hz
        # the result is now in ergs / s / cm^2 / sr /Hz
        if integration_mode == 'romb':
            I_nu = 2. * romb(j, dx=np.diff(ss,axis=1)[:,0], axis=-1) 
        else:
            sss = np.tile(ss[np.newaxis,...], list(nu.shape)+[1,1])
            I_nu = 2. * simps(j, sss, axis=-1) 
        return I_nu, theta_arcmin, j

    def flux(self,
             nu, theta_edges=None,
             which='sync',
             r_steps=129,
             r_min=0.,
             theta_steps=17,
             integration_mode='simps',
             **kwargs):
        """
        Compute the specific flux within some solid angle
        as a line of sight integral over the volume emissivity
        for different angular separations theta

        Parameters
        ----------
        nu: array-like
            Array with frequencies in Hz

        theta_edges: array-like
            Edges for angular separation within which flux is calculated as an integral
            of the specific intensity .
            Gives in arcmin

        r_steps: int
            steps used for integration over r for intensity calculation

        r_min: float
            minimum radius considered for line of sight integration for
            intensity

        theta_steps: int
            steps used for integration over solid angle between edges

        which: str
            specify for which radiation you want to calculate the
            intensity. Either 'sync', 'ic', or 'dust'

        integration_mode: str
            specify which integration method to use; either 'simps' or 'romb'

        kwargs: dict
            options passed to either interp_sync_init, j_ic, or j_dust_nebula
            depending on value of 'which'

        Returns
        -------
        2D array with flux as function of frequency nu within the theta integration edges
        """
        
        if theta_edges is None:
            theta_edges = [0, self.theta_max_arcmin() * 0.999]
        # first, build a theta array for integration over edges
        theta_array = []
        dtheta_array = []

        kernel = []
        for i, t in enumerate(theta_edges[:-1]):
            t_max = self.theta_max_arcmin()
            t_upper = theta_edges[i+1]
            if t > t_max: 
                continue
            elif t_upper > t_max:
                warnings.warn(f"Setting theta_max to {t_max:.2f}arcmin.")
                t_upper = t_max
            x = np.linspace(t,
                            t_upper,
                            theta_steps)

            I_nu, _, _ = self.intensity2(nu, x,
                                         r_min=r_min,
                                         r_steps=r_steps,
                                         integration_mode=integration_mode,
                                         which=which,
                                         **kwargs)

            theta_array.append(x)
            dtheta_array.append(np.diff(x)[0])
            kernel.append(I_nu)

            kernel[-1] *= np.cos(x * arcmin2rad)
            kernel[-1] *= np.sin(x * arcmin2rad)

        # perform integration
        theta_array = np.array(theta_array)
        dtheta_array = np.array(dtheta_array)

        theta_array *= arcmin2rad
        dtheta_array *= arcmin2rad
        kernel = np.array(kernel)

        if integration_mode == 'romb':
            flux = []
            for i, k in enumerate(kernel):
                y = romb(k, dx=dtheta_array[i])
                flux.append(y)
            flux = np.array(flux)
        else:
            # stack theta array to match dimensin of mu
            theta_array = np.tile(theta_array[:, np.newaxis, :], (1, kernel.shape[1], 1))
            flux = simps(kernel, theta_array, axis=-1)

        flux *= 2. * np.pi
        return np.squeeze(flux)

    def ext68(self,
              nu,
              which='sync',
              r_steps=129,
              r_min=0.,
              theta_max=None,
              theta_steps=20,
              theta_steps_interp=500,
              integration_mode='simps',
              test=0,
              **kwargs):
        """
        Compute the 68% extension of the nebula
        model from a 2D interpolation of the intensity.
        as a line of sight integral over the volume emissivity
        for different angular separations theta

        Parameters
        ----------
        nu: array-like
            Array with frequencies in Hz

        r_steps: int
            steps used for integration over r

        r_min: float
            minimum radius considered for line of sight integration

        which: str
            specify for which radiation you want to calculate the
            intensity. Either 'sync', 'ic', or 'dust'

        integration_mode: str
            specify which integration method to use; either 'simps' or 'romb'

        theta_steps: int
            number of theta steps used to calculate intensity

        theta_steps_interp: int
            number of theta steps that interpolation is performed on

        kwargs: dict
            options passed to either interp_sync_init, j_ic, or j_dust_nebula
            depending on value of 'which'

        Returns
        -------
        Two arrays with frequencies and array with 68% extension for each frequency
        """
        
        if theta_max:
            theta_steps = np.linspace(0., theta_max, theta_steps)
        
        # calculate the intensity along the line of sight
        I_nu, theta_arcmin, _ = self.intensity2(nu,
                                                r_min=r_min,
                                                r_steps=r_steps,
                                                theta=theta_steps,
                                                which=which,
                                                integration_mode=integration_mode,
                                                **kwargs)
        if test==1:
            return I_nu, theta_arcmin
        # interpolate the intensity
        # restrict yourself to some values of frequency
        # for numerical accuracy
        if which == 'sync':
            m = nu < 5e23
        elif which == 'ic':
            m = nu < 3e29
        else:
            m = np.ones(nu.size, dtype=bool)

        if not np.sum(m):
            raise ValueError("No valid nu range provided")

        # perform interpolation
        I_nu_interp = RectBivariateSpline(np.log10(nu[m]), theta_arcmin,
                                          np.log10(I_nu[m, :]),
                                          kx=1,
                                          ky=1)

        # compute a fine grid
        t_test = np.linspace(theta_arcmin[0], theta_arcmin[-1], theta_steps_interp)
        I_interp = 10. ** I_nu_interp(np.log10(nu[m]), t_test)

        # calculate the fluxes between theta bounds,
        # assuming I to be constant within bounds
        dtheta = t_test[1:] - t_test[:-1]
        theta_cen = 0.5 * (t_test[1:] + t_test[:-1])
        f_interp = 0.5 * (I_interp[:, 1:] + I_interp[:, :-1]) * dtheta

        # multiply with remaining theta dependence of integrand
        f_interp *= np.cos(theta_cen * arcmin2rad) * np.sin(theta_cen * arcmin2rad)

        # compute CDF from fluxes
        cdf_interp = np.cumsum(f_interp, axis=1)
        cdf_interp = (cdf_interp.T - cdf_interp.min(axis=1)).T
        cdf_interp = (cdf_interp.T / cdf_interp.max(axis=1)).T
        if test == 2:
            return t_test, I_interp
        # compute 68% quantile from nearest index
        idx68 = np.argmin(np.abs(cdf_interp - 0.68), axis=1)

        # return the 68% extension
        return nu[m], t_test[idx68]