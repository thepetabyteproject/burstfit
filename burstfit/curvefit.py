#!/usr/bin/env python3

import logging

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class CurveFit:
    """
    Simple wrapper class to handle curve fitting. It can also retry
    the fitting with modified bounds if errors are encountered
    or if the fitting errors are not finite.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    for details of inputs.

    Args:
        function: Fitting function
        xdata: Input x data array (independent variable)
        ydata: Input y data array to fit (dependent variable)
        bounds: Lower and upper bounds on parameters
        p0: Initial guess for the parameters
        retry: To retry the fitting in case of RunTimeError or infinite fit errors
        retry_frac_runtimeerror: To set the bounds using p0 in case of RuntimeError
        retry_frac_infinite_err: To set the bounds using p0 in case of infinite fit errors
    """

    def __init__(
        self,
        function,
        xdata,
        ydata,
        bounds=[-np.inf, np.inf],
        p0=None,
        retry=False,
        retry_frac_runtimeerror=0.9,
        retry_frac_infinite_err=0.2,
    ):
        self.function = function
        self.xdata = xdata
        self.ydata = ydata
        self.bounds = bounds
        self.retry = retry
        self.retry_frac_runtimeerror = retry_frac_runtimeerror
        self.retry_frac_infinite_err = retry_frac_infinite_err
        self.p0 = p0

    def run_fit(self):
        """
        Runs the fitting function and checks for errors and retries.

        Returns:
            popt: List of converged parameters
            err: Errors on the parameters

        """
        if self.retry:
            try:
                popt, err = self.cf()
            except RuntimeError as e:
                logger.warning(f"{e}")
                logger.warning(
                    f"Retrying with p0+-({self.retry_frac_runtimeerror}*p0) bounds"
                )
                p0_1 = np.array(self.p0) * (1 - self.retry_frac_runtimeerror)
                p0_2 = np.array(self.p0) * (1 + self.retry_frac_runtimeerror)
                self.bounds = (
                    np.min([p0_1, p0_2], axis=0),
                    np.max([p0_1, p0_2], axis=0),
                )
                popt, err = self.cf()

            if np.isinf(err).sum() > 0:
                logger.warning(
                    f"Fit errors are not finite. Retrying with p0+-({self.retry_frac_infinite_err}*p0) bounds"
                )
                p0_1 = np.array(self.p0) * (1 - self.retry_frac_infinite_err)
                p0_2 = np.array(self.p0) * (1 + self.retry_frac_infinite_err)
                self.bounds = (
                    np.min([p0_1, p0_2], axis=0),
                    np.max([p0_1, p0_2], axis=0),
                )
                popt, err = self.cf()
                assert (
                    np.isinf(err).sum() == 0
                ), "Errors are still not finite. Terminating."
        else:
            popt, err = self.cf()
            assert np.isinf(err).sum() == 0, "Fit errors are not finite. Terminating."
        return popt, err

    def cf(self):
        """
        Do the actual curve fitting using curve_fit

        Returns:
            popt: List of converged parameters
            err: Errors on the parameters

        """
        logger.debug(f"Bounds for the fit are: {self.bounds}")
        popt, pcov = curve_fit(
            self.function,
            xdata=self.xdata,
            ydata=self.ydata,
            p0=self.p0,
            bounds=self.bounds,
        )
        err = np.sqrt(np.diag(pcov))
        return popt, err
