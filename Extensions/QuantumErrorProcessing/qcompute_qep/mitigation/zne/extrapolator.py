#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Define Classes and functions that implement various extrapolation methods.
"""

import abc
import warnings
from scipy import optimize
from typing import List, Any, Union, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from inspect import signature


ExtrapolationResult = Tuple[
    Any,  # The extrapolated value
    Any,  # The (estimated) error of the extrapolated value
    Any,  # store the optimal extrapolation parameters
    np.ndarray,  # store the covariance of the fitting parameters
    Callable,  # store the fit function, must be a callable object
]

__SUPPORTED_EXTRAPOLATORS__ = {'linear', 'exponential', 'richardson', 'polynomial', 'exppoly', 'customized'}


class Extrapolator(abc.ABC):
    r"""The Abstract Extrapolation Class.

    Each inherited class must implement the ``extrapolate`` method.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._extra_value: float = None  # store the extrapolated value
        self._extra_error: float = None  # store the (estimated) error of the extrapolated value
        self._popt: List[float] = None  # store the optimal fitting parameters
        self._pcov: np.ndarray = None  # store the covariance of the fitting parameters
        self._data: Tuple[List[float], List[float]] = None  # store the xdata and ydata
        self._order = kwargs.get('order', 1)  # the extrapolation order
        self._asymptote = kwargs.get('asymptote', 1)  # the fitting asymptotic value
        self._func: Callable = None  # the fit function, must be a callable object
        self._has_extrapolated = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.extrapolate(*args, **kwargs)

    @abc.abstractmethod
    def extrapolate(self, *args: Any, **kwargs: Any) -> Any:
        r"""The abstract extrapolate function.

        Each inherited class must implement this method.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        s = 'A(n) {} object. has extrapolation record: {}'.format(self.name, self.has_extrapolated)
        if self.has_extrapolated:
            s += '\nextrapolation result: {:.2f}, error of result: {:.2f}'.format(self._extra_value, self._extra_error)
            s += '\noptimized parameters: {}'.format(np.round(s, 2))
            s += '\ncovariance of optimized parameters:\n{}'.format(np.round(self._pcov, 2))
        return s

    def plot_data(self, save: bool = False, fname: str = None) -> Figure:
        r"""Plot a scatter figure using original x and y data.

        The original x and y data are later used for extrapolation.

        :param save: bool, indicate if to save the figure or not
        :param fname: str, if save=True, save the figure with the given figure name
        :return: Figure, a `matplotlib.figure.Figure` instance
        """
        fig = plt.figure(figsize=(8, 5))

        plt.scatter(self._data[0], self._data[1], edgecolors='k', c='lightblue')

        plt.xlabel('Scaling factor — $\lambda$', fontsize=13)
        plt.ylabel('Expectation — $E(\lambda)$', fontsize=13)
        plt.title('Original Data', fontsize=16)
        plt.show()
        if save:
            fname = 'original_data' if fname is None else fname
            fig.savefig(fname, dpi=300)
        return fig

    def plot_extrapolation_results(self, save: bool = False, fname: str = None) -> Figure:
        r"""Plot the extrapolation result.

        Plot the extrapolation result together with the raw data used for extrapolation.

        :param save: bool, indicate if to save the figure or not
        :param fname: str, if save=True, save the figure with the given figure name
        :return: Figure, a `matplotlib.figure.Figure` instance
        """
        fig = plt.figure(figsize=(8, 5))
        # plot the raw data
        plt.scatter(self._data[0], self._data[1], edgecolors='lightblue', c='lightblue', alpha=0.6)
        # plot the extrapolation value
        plt.scatter(0, self._extra_value, c='red', marker='*', alpha=0.6)
        # plot the fitted curve
        x = np.linspace(0, self._data[0][-1])
        y = self.fitted_func(x)
        plt.plot(x, y, '--', color='k')
        plt.xlabel(r'Scaling Factor — $\lambda$', fontsize=13)
        plt.ylabel(r'Expectation Value — $E(\lambda)$', fontsize=13)
        fig.tight_layout()
        plt.show()
        if save:
            fname = 'extrapolation_result' if fname is None else fname
            fig.savefig(fname, dpi=300)
        return fig

    def fitted_func(self, xdata: Union[float, List[float]]) -> Union[float, List[float]]:
        return self._func(xdata)

    @property
    def extrapolation_result(self) -> ExtrapolationResult:
        return self._extra_value, self._extra_error, self._popt, self._pcov, self._func

    @extrapolation_result.setter
    def extrapolation_result(self, res: ExtrapolationResult):
        self._extra_value, self._extra_error, self._popt, self._pcov, self._func = res

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def has_extrapolated(self):
        return self._has_extrapolated


# Self-defined polynomial extrapolation function
def qep_poly_extrapolate(xdata: List[float],
                         ydata: List[float],
                         order: int) -> ExtrapolationResult:
    r"""Use the ``numpy.polyfit`` function to fit the data.

    :param xdata: List[float], the independent variable where the ydata is measured
    :param ydata: List[float], the dependent data, nominally f(xdata, ...)
    :param order: int, the maximal order of the polynomial function
    :return: ExtrapolationResult, the extrapolation results
    """
    with warnings.catch_warnings(record=True):
        try:
            popt, pcov, *_ = np.polyfit(xdata, ydata, order, full=False, cov=True)
        except (ValueError, np.linalg.LinAlgError):
            popt = np.polyfit(xdata, ydata, order)
            pcov = None

    # The extrapolation result
    result = popt[-1]

    # Record the extrapolation information
    func = np.poly1d(popt)
    res_error = None
    if pcov is not None and pcov.shape == (order + 1, order + 1):
        res_error = np.sqrt(pcov[order, order])  # standard deviation

    return result, res_error, popt, pcov, func


# Self-defined extrapolation function
def qep_curve_extrapolate(ansatz: Callable, xdata: List[float], ydata: List[float], **kwargs) -> ExtrapolationResult:
    r"""Use the ``scipy.optimize.curve_fit`` function to fit the data.

    The optional keyword arguments can be

    + ``p0``: List[float], the initial guess for the parameters,
    + ``sigma``: determines the uncertainty in ydata, and
    + ``bounds``: lower and upper bounds on parameters.

    :param ansatz: Callable, the fit function :math:`f(x)`. It must take the independent variable as
                    the first argument and the parameters to fit as separate remaining arguments
    :param xdata: List[float], the independent variable where the ydata is measured
    :param ydata: List[float], the dependent data, nominally f(xdata, ...)
    :return: ExtrapolationResult, the extrapolation results
    """
    p0 = kwargs.get('p0', None)
    sigma = kwargs.get('sigma', None)
    bounds = kwargs.get('bounds', (-np.inf, np.inf))
    try:
        popt, pcov = optimize.curve_fit(ansatz, xdata, ydata, p0=p0, sigma=sigma, bounds=bounds, maxfev=5000)
    except ValueError:
        print('optimize.curve_fit() fails. Either ydata or xdata contain NaNs. popt set to 0.')
        sig = signature(ansatz)
        popt = np.zeros(len(sig.parameters) - 1)
        pcov = None
    except RuntimeError:
        print('optimize.curve_fit() fails because the least-squares minimization cannot converge. popt set to 0.')
        sig = signature(ansatz)
        popt = np.zeros(len(sig.parameters) - 1)
        pcov = None
    else:
        pass

    result = ansatz(0, *popt)
    res_error = None  # not computed within fitting process

    def func(x: Union[float, List[float]]) -> Union[float, List[float]]:
        return ansatz(x, *popt)

    return result, res_error, popt, pcov, func


class LinearExtrapolator(Extrapolator):
    r"""The Linear Extrapolator Class.
    """
    def extrapolate(self, xdata: List[float], ydata: List[float], **kwargs: Any) -> float:
        r"""Use the `numpy.polyfit` linear function to fit the data.

        The `numpy.polyfit` linear function is mathematically defined as

                .. math:: y(x) = c_1 x + c_0.

        The extrapolated value is obtained by considering the zero limit: :math:`\lim_{x\to 0}y(x) = c_0`.

        :param xdata: List[float], a set of x values for fitting
        :param ydata: List[float], a set of y values for fitting
        :return: float, the extrapolated value to the zero-limit
        """
        self._has_extrapolated = True
        # sort the xdata ascendingly and rearrange the ydata accordingly
        idx_sort = np.argsort(xdata)
        self._data = (np.array(xdata)[idx_sort], np.array(ydata)[idx_sort])
        # set the fit order to 1 and enforce the linear fit function
        self._order = 1
        self.extrapolation_result = qep_poly_extrapolate(self._data[0], self._data[1], order=self._order)
        return self._extra_value


class PolynomialExtrapolator(Extrapolator):
    r"""The Polynomial Extrapolator Class.
    """
    def extrapolate(self, xdata: List[float], ydata: List[float], order: int = 1, **kwargs: Any) -> float:
        r"""Use the `numpy.polyfit` polynomial function to fit the data.

        The `numpy.polyfit` polynomial function is mathematically defined as

                .. math:: y(x) = \sum_{k=0}^{d}c_k x^k,

        where :math:`d` is the given extrapolation order.
        The extrapolated value is obtained by considering the zero limit: :math:`\lim_{x\to 0}y(x) = c_0`.

        :param xdata: List[float], a set of x values for fitting
        :param ydata: List[float], a set of y values for fitting
        :param order: int, the extrapolation order, defaults to 1 (linear fitting)
        :return: float, the extrapolated value to the zero-limit
        """
        self._has_extrapolated = True
        # sort the xdata ascendingly and rearrange the ydata accordingly
        idx_sort = np.argsort(xdata)
        self._data = (np.array(xdata)[idx_sort], np.array(ydata)[idx_sort])
        self._order = order
        self.extrapolation_result = qep_poly_extrapolate(self._data[0], self._data[1], self._order)
        return self._extra_value


class ExpPolyExtrapolator(Extrapolator):
    r"""The Exponential Polynomial Extrapolator Class.
    """
    def extrapolate(self,
                    xdata: List[float],
                    ydata: List[float],
                    order: int = 1,
                    asymptote: float = None,
                    **kwargs: Any) -> float:
        r"""
        Use the `numpy.polyfit` exponential-polynomial function to fit data.

        The `numpy.polyfit` exponential-polynomial function is mathematically defined as

                .. math:: y(x) = a + \exp{\left(\sum_{k=0}^{d}c_k x^k\right)},

        where :math:`d` is the given extrapolation order.
        The extrapolated value is obtained by considering the zero limit: :math:`\lim_{x\to 0}y(x) = a + e^{c_0}`.

        :param xdata: List[float], a set of x values for fitting
        :param ydata: List[float], a set of y values for fitting
        :param order: int, the extrapolation order, defaults to 1 (linear fitting)
        :param asymptote: the expected asymptotic value :math`a` for extrapolation
        :return: float, the extrapolated value to the zero-limit
        """
        self._has_extrapolated = True
        # sort the xdata ascendingly and rearrange the ydata accordingly
        idx_sort = np.argsort(xdata)
        self._data = (np.array(xdata)[idx_sort], np.array(ydata)[idx_sort])

        is_increase = np.count_nonzero(np.gradient(self._data[1], self._data[0]) > 0) > len(xdata) / 2
        sign = 1 if is_increase else -1
        self._order = order

        def _ansatz_unknown(x: float, *coeffs: float) -> float:
            """Ansatz of generic order with unknown asymptote."""
            # Coefficients of the polynomial to be exponentiated
            z_coeffs = np.flip(coeffs[2:])  # reverse
            return coeffs[0] + coeffs[1] * np.exp(x * np.polyval(z_coeffs, x))

        def _ansatz_known(x: float, *coeffs: float) -> float:
            """Ansatz of generic order with known asymptote."""
            # Coefficients of the polynomial to be exponentiated
            z_coeffs = np.flip(coeffs[1:])
            return asymptote + coeffs[0] * np.exp(x * np.polyval(z_coeffs, x))

        if asymptote is None:
            # fit and extrapolate
            p0 = [0, sign, -1] + [0] * (self._order - 1)
            self.extrapolation_result = qep_curve_extrapolate(_ansatz_unknown,
                                                              self._data[0],
                                                              self._data[1],
                                                              p0=p0)
            self._extra_error = np.sqrt(self._pcov[0, 0] + self._pcov[1, 1] + 2 * self._pcov[0, 1])

        else:
            # fit and extrapolate
            p0 = [sign, -1] + [0] * (self._order - 1)
            self.extrapolation_result = qep_curve_extrapolate(_ansatz_known,
                                                              self._data[0],
                                                              self._data[1],
                                                              p0=p0)
            self._extra_error = np.sqrt(self._pcov[0, 0])

        return self._extra_value


class ExponentialExtrapolator(Extrapolator):
    r"""The Exponential Extrapolator Class.
    """
    def extrapolate(self, xdata: List[float], ydata: List[float], **kwargs: Any) -> float:
        r"""Use the `scipy.optimize.curve_fit` exponential function to fit data.

        The `scipy.optimize.curve_fit` exponential function is mathematically defined as

                .. math:: y(x) = a + be^{-cx},  c > 0.

        The extrapolated value is obtained by considering the zero limit: :math:`\lim_{x\to 0}y(x) = a + b`.

        :param xdata: List[float], a set of x values for fitting
        :param ydata: List[float], a set of y values for fitting
        :return: float, the extrapolated value to the zero-limit
        """
        self._has_extrapolated = True
        self._asymptote = kwargs.get('asymptote', 1)
        # sort the xdata ascendingly and rearrange the ydata accordingly
        idx_sort = np.argsort(xdata)
        self._data = (np.array(xdata)[idx_sort], np.array(ydata)[idx_sort])

        is_increase = np.count_nonzero(np.gradient(self._data[1], self._data[0]) > 0) > len(xdata) / 2
        sign = 1 if is_increase else -1

        # Use the first two points to guess the decay param
        if self._asymptote is None:
            # non-linear fitting
            ansatz = lambda x, a, b, c: a + b * np.exp(-c * x)
            self.extrapolation_result = qep_curve_extrapolate(ansatz,
                                                              self._data[0],
                                                              self._data[1],
                                                              p0=[self._data[1][0], sign, -1])
        else:
            ansatz = lambda x, b, c: self._asymptote + b * np.exp(-c * x)
            self.extrapolation_result = qep_curve_extrapolate(ansatz,
                                                              self._data[0],
                                                              self._data[1],
                                                              p0=[sign, -1])

        return self._extra_value


class RichardsonExtrapolator(Extrapolator):
    r"""The Richardson Extrapolator Class.
    """
    def extrapolate(self, xdata: List[float], ydata: List[float], **kwargs: Any) -> float:
        r"""Use the Richardson extrapolation function to fit the data.

        In fact, the Richardson extrapolation is exactly the polynomial extrapolation with order :math:`d=l-1`,
        where :math:`l` is the length of xdata list for extrapolation.
        The fitting function is mathematically defined as

                .. math:: y(x) = \sum_{k=0}^{l-1}c_k x^k.

        The extrapolated value is obtained by considering the zero limit: :math:`\lim_{x\to 0}y(x) = c_0`.

        :param xdata: List[float], a set of x values for fitting
        :param ydata: List[float], a set of y values for fitting
        :return: float, the extrapolated value to the zero-limit
        """
        self._has_extrapolated = True
        # sort the xdata ascendingly and rearrange the ydata accordingly
        idx_sort = np.argsort(xdata)
        self._data = (np.array(xdata)[idx_sort], np.array(ydata)[idx_sort])

        # set the polynomial fit order
        self._order = len(xdata) - 1

        # fit and extrapolate
        self.extrapolation_result = qep_poly_extrapolate(self._data[0], self._data[1], self._order)
        self._func = np.poly1d(self._popt)

        return self._extra_value


class CustomizedExtrapolator(Extrapolator):
    r"""The Customized Extrapolator Class.
    """
    def __init__(self, ansatz: Callable = None, *args: Any, **kwargs: Any) -> None:
        super(CustomizedExtrapolator, self).__init__()
        self._func = ansatz

    def extrapolate(self, xdata: List[float], ydata: List[float], ansatz: Callable = None, **kwargs: Any) -> Any:
        r"""Use the user customized function :math:`f(x)` to fit the data.

        The extrapolated value is obtained by considering the zero limit: :math:`\lim_{x\to 0}f(0)`.

        :param xdata: List[float], the independent variable where the ydata is measured
        :param ydata: List[float], the dependent data, nominally f(xdata, ...)
        :param ansatz: Callable, the fit function :math:`f(x)`. It must take the independent variable as
                        the first argument and the parameters to fit as separate remaining arguments
        :param p0: List[float], initial guess for the parameters
        :param kwargs: other optional key word arguments
        :return: corresponding y-axis data when data of x-axis is zero
        """
        self._has_extrapolated = True
        # sort the xdata ascendingly and rearrange the ydata accordingly
        idx_sort = np.argsort(xdata)
        self._data = (np.array(xdata)[idx_sort], np.array(ydata)[idx_sort])
        # determine the fit function
        if self._func is None and ansatz is None:
            raise TypeError("The ansatz function has not been provided!")
        if ansatz is not None:
            self._func = ansatz

        self.extrapolation_result = qep_curve_extrapolate(self._func, self._data[0], self._data[1], **kwargs)

        return self._extra_value
