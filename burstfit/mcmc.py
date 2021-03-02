#!/usr/bin/env python3

import logging
from multiprocessing import Pool

import emcee
import numpy as np

from burstfit.utils.plotter import plot_mcmc_results

logger = logging.getLogger(__name__)


class MCMC:
    """
    Class to run MCMC on the burst model.

    Args:
        model_function:
        sgram:
        initial_guess:
        param_names:
        nwalkers:
        nsteps:
        skip:
        start_pos_dev:
        prior_range:
        ncores:
        outname:
    """

    def __init__(
        self,
        model_function,
        sgram,
        initial_guess,
        param_names=None,
        nwalkers=30,
        nsteps=1000,
        skip=3000,
        start_pos_dev=0.01,
        prior_range=0.2,
        ncores=10,
        outname="mcmc_res",
    ):
        self.model_function = model_function
        if isinstance(initial_guess, dict):
            cf_params = []
            cf_errors = []
            for i, value in initial_guess.items():
                cf_params += value["popt"]
                cf_errors += list(value["perr"])
            initial_guess = np.array(cf_params)
        elif isinstance(initial_guess, list):
            initial_guess = np.array(initial_guess)
            cf_errors = None
        else:
            cf_errors = None

        self.cf_errors = cf_errors
        self.initial_guess = np.array(initial_guess)
        self.prior_range = prior_range
        self.sgram = sgram
        self.std = np.std(sgram)
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.skip = skip
        self.start_pos_dev = start_pos_dev
        self.ncores = ncores
        self.sampler = None
        self.samples = None
        self.param_names = param_names
        self.outname = outname
        self.set_initial_pos()

    @property
    def ndim(self):
        """

        Returns:

        """
        return len(self.initial_guess)

    def lnprior(self, params):
        """

        Args:
            params:

        Returns:

        """
        m1 = (params) < (1 + self.prior_range) * self.initial_guess
        m2 = (params) > (1 - self.prior_range) * self.initial_guess

        if m1.sum() + m2.sum() == 2 * len(self.initial_guess):
            return 0
        else:
            return -np.inf

    def lnprob(self, params):
        """

        Args:
            params:

        Returns:

        """
        lp = self.lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlk(inps=params)

    def lnlk(self, inps):
        """

        Args:
            inps:

        Returns:

        """
        model = self.model_function([0], *inps)
        return -0.5 * np.sum(((self.sgram.ravel() - model) / self.std) ** 2)

    def set_initial_pos(self):
        """

        Returns:

        """
        pos = [
            np.array(self.initial_guess)
            * np.random.uniform(
                1 - self.start_pos_dev, 1 + self.start_pos_dev, size=self.ndim
            )
            for i in range(self.nwalkers)
        ]
        self.pos = np.array(pos)
        return self

    def run_mcmc(self):
        """

        Returns:

        """
        with Pool(self.ncores) as pool:
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                self.lnprob,
                pool=pool,
            )
            sampler.run_mcmc(self.pos, self.nsteps, progress=True)
        self.sampler = sampler
        return self.sampler

    def get_chain(self, skip=None):
        """

        Args:
            skip:

        Returns:

        """
        if not skip:
            skip = self.skip
        self.samples = self.sampler.get_chain(flat=True)[skip:, :]
        return self.samples

    def print_results(self):
        """

        Returns:

        """
        if not np.any(self.samples):
            self.get_chain()
        qs = np.quantile(self.samples, [0.16, 0.5, 0.84], axis=0)
        logger.info(f"MCMC fit results are:")
        e1 = qs[1] - qs[0]
        e2 = qs[2] - qs[1]
        p = qs[1]
        for i, param in enumerate(self.param_names):
            logger.info(f"{self.param_names[i]}: {p[i]} + {e2[i]:.2f} - {e1[i]:.2f}")

    def plot(self, save=False):
        """

        Args:
            save:

        Returns:

        """
        plot_mcmc_results(
            self.samples, self.outname, self.initial_guess, self.param_names, save
        )
