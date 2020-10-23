import numpy as np
import iminuit as minuit
import time
import functools
import logging
from scipy.special import ndtri, erf
from collections import OrderedDict
from copy import deepcopy
import numpy as np

# --- Helper functions ------------------------------------- #


def set_default(func = None, passed_kwargs = {}):
    """
    Read in default keywords of the simulation and pass to function
    """
    if func is None:
        return functools.partial(set_default, passed_kwargs = passed_kwargs)

    @functools.wraps(func)
    def init(*args, **kwargs):
        for k in passed_kwargs.keys():
            kwargs.setdefault(k,passed_kwargs[k])
        return func(*args, **kwargs)
    return init
# ---------------------------------------------------------- #

# --- minuit defaults ------------------------------------- #
# The default tolerance is 0.1.
# Minimization will stop when the estimated vertical distance to the minimum (EDM)
# is less than 0.001*tolerance*UP (see SET ERR).


minuit_def = {
    'verbosity': 0,
    'int_steps': 1e-4,
    'strategy': 2,
    'tol': 1e-3,
    'up': 1.,
    'max_tol_increase': 3000.,
    'tol_increase': 1000.,
    'ncall': 5000,
    'pedantic': True,
    'precision': None,

    'pinit': {'norm' : -10.,
              'index': -3.,
              'alphac': 1.,
              'r': 17.},
    'fix': {'norm' : False,
            'index': False,
            'alphac': False,
            'r': False },
    'limits': {'norm' : [-20,-5],
               'index': [-5,5],
               'alphac': [0.1,10.],
               'r': [16.,18.]},
    'islog': {'norm' : True,
              'index': False,
              'alphac': False,
              'r': True},
}
# ---------------------------------------------------------- #


class FitCrab(object):
    def __init__(self, crab_ssc, crab_data, fit_sync=False, fit_ic=False, dsys=0.):

        self._parnames = None
        self._par_islog = None
        self._ssc = crab_ssc
        self._data = crab_data
        self._fit_sync = fit_sync
        self._fit_ic = fit_ic
        self._x = None
        self._y = None
        self._dy = None
        self._y_theo = None
        self._dsys = dsys
        self._minimize_f = None
        self._m = None
        self._fitarg = None
        self._n_pars = 0
        # save the initial electron spectrum parameters
        # and magnetic field for the IC fit
        self._p0 = deepcopy(self._ssc.params_n_el)
        self._p0['B'] = deepcopy(self._ssc.B)
        pass

    @property
    def ssc(self):
        return self._ssc

    @property
    def data(self):
        return self._data

    @property
    def fit_sync(self):
        return self._fit_sync

    @fit_sync.setter
    def fit_sync(self, fit_sync):
        self._fit_sync = fit_sync

    @property
    def fit_ic(self):
        return self._fit_ic

    @fit_ic.setter
    def fit_ic(self, fit_ic):
        self._fit_ic = fit_ic

    @property
    def dsys(self):
        return self._dsys

    @property
    def m(self):
        return self._m

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def dy(self):
        return self._dy

    @property
    def y_theo(self):
        return self._y_theo

    @property
    def n_pars(self):
        return self._n_pars

    @dsys.setter
    def dsys(self, dsys):
        self._dsys= dsys

    def init_values_crab_meyer(self,
                               islog_keys=('Nradio', 'gmin', 'gmax', 'gradio_min',
                                           'gradio_max', 'Nwind', 'gwind_min', 'gwind_max', 'gwind_b', 'dust_norm'),
                               fix_keys=('dust_extension', 'gmin', 'gmax'),
                               exclude=('ic_dust', 'ic_sync', 'ic_cmb')
                               ):
        """
        Generate initial guess dictionaries needed for fitting
        from current SSC model.

        Tuned to the fit parameters of Meyer et al. const B-field model.

        Returns
        -------
        dict with four sub dictionaries:
        - pinit: initial guesses for parameters
        - limits: limits for the parameters
        - fix: whether parameters are fixed or not
        - islog: whether parameters are fitted in log10 space
        """
        pinit, limits, islog, fix = {}, {}, {}, {}

        for d in [self._ssc.params_n_seed, self._ssc.params_n_el]:
            for k, v in d.items():
                if k in exclude:
                    continue
                if k in islog_keys:
                    pinit[k] = np.log10(v)
                    limits[k] = [pinit[k] - 2., pinit[k] + 2.]
                    islog[k] = True
                else:
                    pinit[k] = v
                    limits[k] = np.sort([pinit[k] * 0.3, pinit[k] * 3.])
                    islog[k] = False
                if k in fix_keys:
                    fix[k] = True
                else:
                    fix[k] = False

        pinit['B'] = self._ssc.B
        limits['B'] = [pinit['B'] * 0.1, pinit['B'] * 10.]
        islog['B'] = False
        fix['B'] = self._fit_sync
        result = dict(pinit=pinit, islog=islog, fix=fix, limits=limits)
        return result

    def _fill_chisq(self, *args):
        """
        likelihood function passed to iMinuit
        """
        params = {}
        for i, p in enumerate(self._parnames):
            if self._par_islog[p]:
                params[p] = np.power(10., args[i])
            else:
                params[p] = args[i]
        return self.chisq(params)

    def chisq(self, params):
        """Calculate the chi^2 sq"""
        chi_sq = 0

        # update the model parameters
        # in the ssc module
        for k, v in params.items():
            if k in self._ssc.params_n_el.keys():
                self._ssc.params_n_el[k] = v
            elif k in self._ssc.params_n_seed.keys():
                self._ssc.params_n_seed[k] = v
            elif k == 'B':
                self._ssc.B = v

        # perform fit to synchrotron data only
        if self._fit_sync and not self._fit_ic:
            self._y_theo = self._x * (self._ssc.sync(self._x, g_steps=50) + self._ssc.grey_body(self._x))

        # perform fit to IC data only
        elif self._fit_ic and not self._fit_sync:
            self._ssc.FSyncInterp = None  # init new synchrotron interpolation upon each call

            # rescale gamma factors to keep synchrotron part constant
            self._ssc.params_n_el['gradio_min'] = self._p0['gradio_min'] * np.sqrt(self._p0['B']/ self._ssc.B)
            self._ssc.params_n_el['gradio_max'] = self._p0['gradio_max'] * np.sqrt(self._p0['B']/ self._ssc.B)

            # rescale normalization to keep synchrotron part constant
            self._ssc.params_n_el['Nradio'] = self._p0['Nradio'] * \
                                              (self._p0['B'] / self._ssc.B) ** \
                                              ((-self._ssc.params_n_el['Sradio'] + 1.)/2.)

            # see Blumenthal & Gould (1970), Eq. 4.59
            self._ssc.params_n_el['gwind_min'] = self._p0['gwind_min'] * np.sqrt(self._p0['B']/ self._ssc.B)
            self._ssc.params_n_el['gwind_max'] = self._p0['gwind_max'] * np.sqrt(self._p0['B']/ self._ssc.B)

            # divide since it's the inverse we are fitting
            self._ssc.params_n_el['gwind_b'] = self._p0['gwind_b'] / np.sqrt(self._p0['B']/ self._ssc.B)
            # rescale normalization to keep synchrotron part constant
            # #ssc._ssc.params_n_el['Nwind']	= self._p0['Nwind'] * (self._p0['B']/ self._ssc.B) \
            # ** ((-self._ssc.params_n_el['Swind']+ 1.)/2.)
            # why like this and not as above???
            self._ssc.params_n_el['Nwind'] = self._p0['Nwind'] * (self._p0['B']/ self._ssc.B) ** 0.5

            self._y_theo = self._ssc.ic(self._x, g_steps=100, e_steps=50) * self._x

        # perform fit over both IC and sync
        else:
            m_sync = self._x < 1e22
            self._y_theo = np.zeros_like(self._x)
            self._y_theo[m_sync] = self._x[m_sync] * (self._ssc.sync(self._x[m_sync], g_steps=50) +
                                              self._ssc.grey_body(self._x[m_sync]))

            self._y_theo[~m_sync] = self._x[~m_sync] * (self._ssc.sync(self._x[~m_sync], g_steps=50) +
                                                        self._ssc.ic(self._x[~m_sync], g_steps=100, e_steps=50))

        chi_sq = np.sum((self._y_theo - self._y)**2.
                        / (np.sqrt(self._dy ** 2. + self._dsys**2. * self._y**2.))**2.)

        return chi_sq

    @set_default(passed_kwargs = minuit_def)
    def fill_fitarg(self, **kwargs):
        """
        Helper function to fill the dictionary for minuit fitting
        """
        # set the fit arguments
        fitarg = {}
        fitarg.update(kwargs['pinit'])
        for k in kwargs['limits'].keys():
            fitarg['limit_{0:s}'.format(k)] = kwargs['limits'][k]
            fitarg['fix_{0:s}'.format(k)] = kwargs['fix'][k]
            fitarg['error_{0:s}'.format(k)] = kwargs['pinit'][k] * kwargs['int_steps']

        fitarg = OrderedDict(sorted(fitarg.items()))
        # get the names of the parameters
        self._parnames = kwargs['pinit'].keys()
        self._par_islog = kwargs['islog']
        return fitarg

    @set_default(passed_kwargs=minuit_def)
    def run_migrad(self, fitarg, **kwargs):
        """
        Helper function to initialize migrad and run the fit.
        Initial parameters are estimated with scipy fit.
        """
        self._fitarg = fitarg

        values, bounds = [],[]
        for k in self._parnames:
            values.append(fitarg[k])
            bounds.append(fitarg['limit_{0:s}'.format(k)])

        logging.info(self._parnames)
        logging.info(values)

        logging.info(self._fill_chisq(*values))

        cmd_string = "lambda {0}: self.__fill_chisq({0})".format(
            (", ".join(self._parnames), ", ".join(self._parnames)))

        string_args = ", ".join(self._parnames)
        global f # needs to be global for eval to find it
        f = lambda *args: self._fill_chisq(*args)

        cmd_string = "lambda %s: f(%s)" % (string_args, string_args)
        logging.debug(cmd_string)

        # work around so that the parameters get names for minuit
        self._minimize_f = eval(cmd_string, globals(), locals())

        self._m = minuit.Minuit(self._minimize_f,
                               print_level =kwargs['verbosity'],
                               errordef = kwargs['up'],
                               pedantic = kwargs['pedantic'],
                               **fitarg)

        self._m.tol = kwargs['tol']
        self._m.strategy = kwargs['strategy']

        logging.debug("tol {0:.2e}, strategy: {1:n}".format(
            self._m.tol, self._m.strategy))

        self._m.migrad(ncall=kwargs['ncall']) #, precision = kwargs['precision'])
        return

    def __print_failed_fit(self):
        """print output if migrad failed"""
        if not self._m.migrad_ok():
            fmin = self._m.get_fmin()
            logging.warning(
                '*** migrad minimum not ok! Printing output of get_fmin'
            )
            logging.warning('{0:s}:\t{1}'.format('*** has_accurate_covar',
                                                 fmin.has_accurate_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_covariance',
                                                 fmin.has_covariance))
            logging.warning('{0:s}:\t{1}'.format('*** has_made_posdef_covar',
                                                 fmin.has_made_posdef_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_posdef_covar',
                                                 fmin.has_posdef_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_reached_call_limit',
                                                 fmin.has_reached_call_limit))
            logging.warning('{0:s}:\t{1}'.format('*** has_valid_parameters',
                                                 fmin.has_valid_parameters))
            logging.warning('{0:s}:\t{1}'.format('*** hesse_failed',
                                                 fmin.hesse_failed))
            logging.warning('{0:s}:\t{1}'.format('*** is_above_max_edm',
                                                 fmin.is_above_max_edm))
            logging.warning('{0:s}:\t{1}'.format('*** is_valid',
                                                 fmin.is_valid))
        return

    def __repeat_migrad(self, **kwargs):
        """Repeat fit if fit was above edm"""
        fmin = self._m.get_fmin()
        if not self._m.migrad_ok() and fmin['is_above_max_edm']:
            logging.warning(
                'Migrad did not converge, is above max edm. Increasing tol.'
            )
            tol = self._m.tol
            self._m.tol *= self._m.edm / (self._m.tol * self._m.errordef) * kwargs['tol_increase']

            logging.info('New tolerance : {0}'.format(self._m.tol))
            if self._m.tol >= kwargs['max_tol_increase']:
                logging.warning(
                    'New tolerance to large for required precision'
                )
            else:
                self._m.migrad(
                    ncall=kwargs['ncall'])#,
                #precision = kwargs['precision']
                #)
                logging.info(
                    'Migrad status after second try: {0}'.format(
                        self._m.migrad_ok()
                    )
                )
                self._m.tol = tol
        return

    @set_default(passed_kwargs = minuit_def)
    def fit(self, nu_min, nu_max, fit_sync=True, minos=0., exclude=(), **kwargs):
        """
        Fit the SSC model

        Parameters
        ----------
        exclude: list
            list with data set names that should be excluded
        nu_min: float
            minimum considered frequency of data points in Hz
        nu_max: float
            maximum considered frequency of data points in Hz
        fit_sync: bool
            only fit synchrotron part. If false, only fit IC part

        kwargs
        ------
        pinit: dict
            initial guess for intrinsic spectral parameters

        fix: dict
            booleans for freezing parameters

        bounds: dict
            dict with list for each parameter with min and max value


        Returns
        -------
        tuple with likelihood profile for distance of
        gamma-ray emitting region
        """
        self._fit_sync = fit_sync
        self._x, self._y, self._dy = self._data.build_data_set_for_fitting(exclude=exclude,
                                                                        nu_min=nu_min, nu_max=nu_max,
                                                                        log=False, yunit='flux')

        fitarg = self.fill_fitarg(**kwargs)

        t1 = time.time()
        self.run_migrad(fitarg, **kwargs)

        try:
            self._m.hesse()
            logging.debug("Hesse matrix calculation finished")
        except RuntimeError as e:
            logging.warning(
                "*** Hesse matrix calculation failed: {0}".format(e)
            )

        logging.debug(self._m.fval)
        self.__repeat_migrad(**kwargs)
        logging.debug(self._m.fval)

        fmin = self._m.get_fmin()

        if not fmin.hesse_failed:
            try:
                self._corr = self._m.np_matrix(correlation=True)
            except:
                self._corr = -1

        logging.debug(self._m.values)

        if self._m.migrad_ok():
            if minos:
                for k in self.m.values.keys():
                    if kwargs['fix'][k]:
                        continue
                    self.m.minos(k,minos)
                logging.debug("Minos finished")

        else:
            self.__print_failed_fit()

        logging.info('fit took: {0}s'.format(time.time() - t1))

        self._npars = 0
        for k in self._m.values.keys():
            if kwargs['fix'][k]:
                err = np.nan
            else:
                err = self._m.errors[k]
            self._npars += 1
            logging.info('best fit {0:s}: {1:.5e} +/- {2:.5e}'.format(k, self._m.values[k], err))

    def write_best_fit(self):
        """
        Return the current best-fit parameters as a dictionary
        with which a new CrabSSC module can be initialized
        """
        config = dict(params_n_el={}, params_n_seed={})

        for k, v in self._m.values.items():

            if self._par_islog[k]:
                x = 10.**v
            else:
                x = v

            if k in self._ssc.params_n_el.keys():
                config['params_n_el'][k] = x

            elif k in self._ssc.params_n_seed.keys():
                config['params_n_seed'][k] = x

            elif k == 'B':
                config['B'] = x

        for k in ['ic_sync', 'ic_cmb', 'ic_dust']:
            config['params_n_seed'][k] = self._ssc.params_n_seed[k]

        config['d'] = self._ssc.d

        return config

    def llhscan(self, parname, bounds, steps, log = False):
        """
        Perform a manual scan of the likelihood for one parameter
        (inspired by mnprofile)

        Parameters
        ----------
        parname: str
            parameter that is scanned

        bounds: list or tuple
            scan bounds for parameter

        steps: int
            number of scanning steps

        {options}

        log: bool
            if true, use logarithmic scale

        Returns
        -------
        tuple of 4 lists containing the scan values, likelihood values,
        best fit values at each scanning step, migrad_ok status
        """
        llh, pars, ok = [], [], []
        if log:
            values = np.logscape(np.log10(bounds[0]),np.log10(bounds[1]), steps)
        else:
            values = np.linspace(bounds[0], bounds[1], steps)

        for i,v in enumerate(values):
            fitarg = deepcopy(self.m.fitarg)
            fitarg[parname] = v
            fitarg['fix_{0:s}'.format(parname)] = True

            string_args = ", ".join(self._parnames)
            global f # needs to be global for eval to find it
            f = lambda *args: self._fill_chisq(*args)

            cmd_string = "lambda %s: f(%s)" % (string_args, string_args)

            minimize_f = eval(cmd_string, globals(), locals())

            m = minuit.Minuit(minimize_f,
                              print_level=0, forced_parameters=self.m.parameters,
                              pedantic=False, **fitarg)
            m.migrad()
            llh.append(m.fval)
            pars.append(m.values)
            ok.append(m.migrad_ok())

        return values, np.array(llh), pars, ok
