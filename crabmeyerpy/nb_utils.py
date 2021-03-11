import numba as nb
import numpy as np
import math
from scipy.interpolate import RectBivariateSpline
from astropy import constants as c

const_k_B = c.k_B.value
const_e = c.e.value
const_h_bar_c_eV_m = (c.hbar * c.c).to('eV cm').value


@nb.njit(parallel=True,fastmath=True,error_model='numpy')
def multi_5dim_simps(y, x):
    """
    Perform simpson integration over last axis of a 5d array
    using numba assuming non-uniform spacing
    """
    result = np.empty(y.shape[:-1], dtype=y.dtype)

    for i in nb.prange(y.shape[0]):
        for j in range(y.shape[1]):
            for k in range(y.shape[2]):
                for l in range(y.shape[3]):
                    result[i, j, k, l] = simpson_nonuniform(x[i, j, k, l], y[i, j, k, l])
    return result


@nb.njit(parallel=True,fastmath=True,error_model='numpy')
def multi_5dim_simps_even_spacing(y, x):
    """
    Perform simpson integration over last axis of a 5d array
    using numba assuming even spacing
    """
    result = np.empty(y.shape[:-1], dtype=y.dtype)

    for i in nb.prange(y.shape[0]):
        for j in range(y.shape[1]):
            for k in range(y.shape[2]):
                for l in range(y.shape[3]):
                    dx = x[i, j, k, l, -1] - x[i, j, k, l,  0]
                    result[i, j, k, l] = simpson_nb(y[i, j, k, l], dx)

    return result


@nb.njit(parallel=True,fastmath=True,error_model='numpy')
def multi_5dim_piecewise(y, x):
    """
    Perform integration over last axis of a 5d array
    using numba with simple piece-wise mutliplication
    """
    result = np.empty(y.shape[:-1], dtype=y.dtype)

    for i in nb.prange(y.shape[0]):
        for j in range(y.shape[1]):
            for k in range(y.shape[2]):
                for l in range(y.shape[3]):
                    result[i, j, k, l] = sum_piecewise(x[i, j, k, l], y[i, j, k, l])

    return result


@nb.njit(parallel=True,fastmath=True,error_model='numpy')
def multi_3dim_piecewise(y, x):
    """
    Perform integration over last axis of a 3d array
    using numba with simple piece-wise mutliplication
    """
    result = np.empty(y.shape[:-1], dtype=y.dtype)

    for i in nb.prange(y.shape[0]):
        for j in range(y.shape[1]):
            result[i, j] = sum_piecewise(x[i, j], y[i, j])
    return result


@nb.njit(parallel=True,fastmath=True,error_model='numpy')
def multi_4dim_simps_nb(y, x):
    """
    Perform simpson integration over last axis of a 4d array
    using numba
    """
    result = np.empty(y.shape[:-1], dtype=y.dtype)

    for i in nb.prange(y.shape[0]):
        for j in range(y.shape[1]):
            for k in range(y.shape[2]):
                #dx = x[i, j, k, -1] - x[i, j, k, 0]
                #result[i, j, k] = simpson_nb(y[i, j, k], dx)
                result[i, j, k] = simpson_nonuniform(x[i, j, k], y[i, j, k])
    return result


@nb.njit(fastmath=True)
def simpson_nb(y, dx):
    """
    https://stackoverflow.com/questions/50440592/is-there-any-good-way-to-optimize-the-speed-of-this-python-code
    """
    s = y[0]+y[-1]

    n = y.shape[0]//2
    for i in range(n-1):
        s += 4.*y[i*2+1]
        s += 2.*y[i*2+2]

    s += 4*y[(n-1)*2+1]
    return dx / 3. * s / y.shape[0]


@nb.njit(fastmath=True)
def sum_piecewise(x, y):
    result = 0.
    for i in range(x.size - 1):
        result += (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2.

    return result


@nb.njit(fastmath=True)
def simpson_nonuniform(x, f):
    """
    Simpson rule for irregularly spaced data.

        Parameters
        ----------
        x : list or np.array of floats
                Sampling points for the function values
        f : list or np.array of floats
                Function values at the sampling points


        Returns
        -------
        float : approximation for the integral
    """
    N = len(x) - 1
    h = np.diff(x)

    result = 0.0
    for i in range(1, N, 2):
        hph = h[i] + h[i - 1]
        result += f[i] * ( h[i]**3 + h[i - 1]**3
                           + 3. * h[i] * h[i - 1] * hph )\
                     / ( 6 * h[i] * h[i - 1] )
        result += f[i - 1] * ( 2. * h[i - 1]**3 - h[i]**3
                              + 3. * h[i] * h[i - 1]**2)\
                     / ( 6 * h[i - 1] * hph)
        result += f[i + 1] * ( 2. * h[i]**3 - h[i - 1]**3
                              + 3. * h[i - 1] * h[i]**2)\
                     / ( 6 * h[i] * hph )

    if (N + 1) % 2 == 0:
        result += f[N] * ( 2 * h[N - 1]**2
                          + 3. * h[N - 2] * h[N - 1])\
                     / ( 6 * ( h[N - 2] + h[N - 1] ) )
        result += f[N - 1] * ( h[N - 1]**2
                           + 3*h[N - 1]* h[N - 2] )\
                     / ( 6 * h[N - 2] )
        result -= f[N - 2] * h[N - 1]**3\
                     / ( 6 * h[N - 2] * ( h[N - 2] + h[N - 1] ) )
    return result


@nb.njit('void(double[:], double[:], double)', fastmath=True)
def black_body_nb(result, eps, T):
    """
    return photon density for black body in photons / eV / cm^3 for temperature T

    Parameters
    ----------
    result: array-like
        array to store result
    eps: array-like
        energy in eV
    T: float
        black body temperature
    """
    for i in range(len(result)):
        kx = eps[i] / T / (const_k_B / const_e)  # k/e = 8.617e-5, exponent unitless for eps in eV
        result[i] = eps[i] ** 2. / (math.exp(kx) - 1.)
        result[i] /= const_h_bar_c_eV_m ** 3. * math.pi**2.


@nb.njit('void(double[:], double[:], double)', fastmath=True)
def radial_gaussian_nb(result, r, sigma):
    """
    Calculate radial Gaussian for extension sigma

    Parameters
    ----------
    result: array-like
        array to store result
    r: array-like
        radius
    sigma: float
        extension of Gaussian
    """
    for i in range(len(result)):
        result[i] = math.exp(-r[i] ** 2. / 2. / sigma ** 2.)


class interp2d(object):
    """
    Adapted from https://github.com/dbstein/fast_splines/blob/master/fast_splines/fast_splines.py
    """
    def __init__(self, xv, yv, z, k=3, s=0):
        """
        xv are the x-data nodes, in strictly increasing order
        yv are the y-data nodes, in strictly increasing order
            both of these must be equispaced!
        (though x, y spacing need not be the same)
        z is the data
        k is the order of the splines (int)
            order k splines give interp accuracy of order k+1
            only 1, 3, 5, supported
        """
        if k not in [1, 3, 5]:
            raise Exception('k must be 1, 3, or 5')
        self.xv = xv
        self.yv = yv
        self.z = z
        self.k = k
        self._dtype = yv.dtype
        t_erp = RectBivariateSpline(xv, yv, z, kx=k, ky=k, s=0)
        self._tx, self._ty, self._c = t_erp.tck
        self._nx = self._tx.shape[0]
        self._ny = self._ty.shape[0]
        self._hx = self.xv[1] - self.xv[0]
        self._hy = self.yv[1] - self.yv[0]
        self._nnx = self.xv.shape[0]-1
        self._nny = self.yv.shape[0]-1
        self._cr = self._c.reshape(self._nnx+1, self._nny+1)

    def __call__(self, op_x, op_y, out=None):
        """
        out_points are the 1d array of x values to interp to
        out is a place to store the result
        """
        m = int(np.prod(op_x.shape))
        copy_made = False
        if out is None:
            _out = np.empty(m, dtype=self._dtype)
        else:
            # hopefully this doesn't make a copy
            _out = out.ravel()
            if _out.base is None:
                copy_made = True
        _op_x = op_x.ravel()
        _op_y = op_y.ravel()
        splev2(self._tx, self._nx, self._ty, self._ny, self._cr, self.k,
               _op_x, _op_y, m, _out, self._hx, self._hy, self._nnx, self._nny)
        _out = _out.reshape(op_x.shape)
        if copy_made:
            # if we had to make a copy, update the provided output array
            out[:] = _out
        return _out


@nb.njit(parallel=True, fastmath=True)
def splev2(tx, nx, ty, ny, c, k, x, y, m, z, dx, dy, nnx, nny):
    # fill in the h values for x
    k1 = k+1
    hbx = np.empty((m, 6))
    hhbx = np.empty((m, 5))
    lxs = np.empty(m, dtype=np.int64)
    splev_short(tx, nx, k, x, m, dx, nnx, hbx, hhbx, lxs)
    hby = np.empty((m, 6))
    hhby = np.empty((m, 5))
    lys = np.empty(m, dtype=np.int64)
    splev_short(ty, ny, k, y, m, dy, nny, hby, hhby, lys)
    for i in nb.prange(m):
        sp = 0.0
        llx = lxs[i] - k1
        for j in range(k1):
            llx += 1
            lly = lys[i] - k1
            for k in range(k1):
                lly += 1
                sp += c[llx,lly] * hbx[i,j] * hby[i,k]
        z[i] = sp


@nb.njit(parallel=True, fastmath=True)
def splev_short(t, n, k, x, m, dx, nn, hb, hhb, lxs):
    # fetch tb and te, the boundaries of the approximation interval
    k1 = k+1
    nk1 = n-k1
    tb = t[k1-1]
    te = t[nk1+1-1]
    #l = k1
    #l1 = l+1
    adj = int(k/2) + 1
    # main loop for the different points
    for i in nb.prange(m):
        h = hb[i]
        hh = hhb[i]
        # fetch a new x-value arg
        arg = x[i]
        arg = max(tb, arg)
        arg = min(te, arg)
        # search for knot interval t[l] <= arg <= t[l+1]
        l = int(arg/dx) + adj
        l = max(l, k)
        l = min(l, nn)
        lxs[i] = l
        # evaluate the non-zero b-splines at arg.
        h[0] = 1.0
        for j in range(k):
            for ll in range(j+1):
                hh[ll] = h[ll]
            h[0] = 0.0
            for ll in range(j+1):
                li = l + ll + 1
                lj = li - j - 1
                f = hh[ll]/(t[li]-t[lj])
                h[ll] += f*(t[li]-arg)
                h[ll+1] = f*(arg-t[lj])
