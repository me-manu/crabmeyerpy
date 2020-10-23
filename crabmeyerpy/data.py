import numpy as np
import copy
from .ssc import kpc2cm


class CrabData(object):
    def __init__(self, file="data/crab_data.npy", d=2.):
        """
        Initialize data class

        Parameters
        ----------
        file: path
            path to npy file containing the data
        d: float
            distance to nebula in kpc
        """

        self._data = np.load(file, allow_pickle=True).flat[0]
        self._d = d
        self._lumi2flux = 1. / 4. / np.pi / (self._d * kpc2cm) ** 2.

    @property
    def data(self):
        return self._data

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, d):
        self._d = d
        self._lumi2flux = 1. / 4. / np.pi / (self._d * kpc2cm) ** 2.

    def get_data_set(self, name, log=False, yunit='flux'):
        """
        Return the data points for one data set

        Parameter
        ---------
        name: str
            name of the data set
        log: bool
            Return x values on log10

        yunit: str
            if "flux" return y values in nu F nu else return in nu L nu

        Returns
        -------
        tuple with x, y, and dy values
        """

        d = copy.deepcopy(self._data[name])
        x = d.T[0]
        if not log:
            x = 10.**x

        if yunit == 'flux':
            y = d.T[1] * self._lumi2flux
            dy = d.T[2] * self._lumi2flux
        else:
            y = d.T[1]
            dy = d.T[2]

        return x, y, dy

    def build_data_set_for_fitting(self, exclude=(), nu_min=1e7, nu_max=1e30,
                                   log=False, yunit='flux'):
        """
        Build a data set for fitting

        Parameters
        ----------

        exclude: list
            list with data set names that should be excluded
        nu_min: float
            minimum considered frequency of data points in Hz
        nu_max: float
            maximum considered frequency of data points in Hz
        log: bool
            x values in log scale
        yunit: str
            if "flux" return y values in nu F nu else return in nu L nu

        Returns
        -------
        tuple with x, y, and dy values
        """
        for i, k in enumerate(self._data.keys()):
            if k in exclude:
                continue
            x, y, dy = self.get_data_set(k, log=log, yunit=yunit)

            try:
                x_all = np.concatenate([x_all, x])
                y_all = np.concatenate([y_all, y])
                dy_all = np.concatenate([dy_all, dy])
            except UnboundLocalError:
                x_all = x
                y_all = y
                dy_all = dy

        # sort the data
        idx = np.argsort(x_all)
        x_all = x_all[idx]
        y_all = y_all[idx]
        dy_all = np.abs(dy_all[idx])

        if log:
            m = (10.**x_all >= nu_min) & (10.**x_all <= nu_max)
        else:
            m = (x_all >= nu_min) & (x_all <= nu_max)
        m &= dy_all > 0.

        return x_all[m], y_all[m], dy_all[m]
