#!/usr/bin/env python3

import logging
from contextlib import closing
from multiprocessing import Pool

import emcee
import numpy as np

from burstfit.utils.plotter import plot_mcmc_results

logger = logging.getLogger(__name__)


class MCMC:
    """
    Class to run MCMC on the burst model.

    Args:
        model_function: Function to create the model
        sgram: 2D spectrogram data
        initial_guess: Initial guess of parameters for MCMC (can be a dictionary or list)
        param_names: Names of parameters
        nwalkers: Number of walkers to use in MCMC
        nsteps: Number of iterations to use in MCMC
        skip: Number of samples to skip for burn-in
        start_pos_dev: Percent deviation for start position of the samples
        prior_range: Percent of initial guess to set as prior range
        ncores: Number of CPUs to use
        outname: Name of output files
    """

    def __init__(
        self,
        model_function,
        sgram,
        initial_guess,
        param_names,
        nwalkers=30,
        nsteps=1000,
        skip=3000,
        start_pos_dev=0.01,
        prior_range=0.2,
        ncores=10,
        outname="mcmc_res",
        save_results=True,
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
        assert len(param_names) == len(initial_guess)
        self.param_names = param_names
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
        self.outname = outname
        self.save_results=save_results
        self.autocorr=None
        self.set_initial_pos_and_priors()

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
        m1 = params <= self.max_prior
        m2 = params >= self.min_prior

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

    def set_initial_pos_and_priors(self):
        """

        Returns:

        """
        logger.info(f'Initial guess for MCMC is: {self.initial_guess}')
        pos = [
            np.array(self.initial_guess)
            * np.random.uniform(
                1 - self.start_pos_dev, 1 + self.start_pos_dev, size=self.ndim
            )
            for i in range(self.nwalkers)
        ]
        self.pos = np.array(pos)

        self.max_prior = (1 + self.prior_range) * self.initial_guess
        self.min_prior = (1 - self.prior_range) * self.initial_guess

        tau_idx = [i for i, t in enumerate(self.param_names) if 'tau' in t]
        for idx in tau_idx:
            self.min_prior[idx] = 0

        return self

    def run_mcmc(self):
        """

        Returns:

        """
        logger.debug(f'Range of initial positions of walkers (min, max): ({self.pos.min(0)}, {self.pos.max(0)})')
        logger.debug(f'Range of priors (min, max): ({(1 - self.prior_range) * self.initial_guess},'
                     f'{(1 + self.prior_range) * self.initial_guess})')
        if self.save_results:
            backend = emcee.backends.HDFBackend(f'{self.outname}.h5')
            backend.reset(self.nwalkers, self.ndim)
        else:
            backend = None

        index = 0
        autocorr = np.empty(self.nsteps)
        old_tau = np.inf
        with closing(Pool(self.ncores)) as pool:
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                self.lnprob,
                pool=pool,
                backend=backend
            )
            for sample in sampler.sample(self.pos, iterations=self.nsteps, progress=True, store=True):
                if sampler.iteration % 100:
                    continue

                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau

            pool.terminate()
        self.autocorr = autocorr
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
        if self.samples.shape[0] == 0:
            logger.warning(f'Not enough samples in chain to skip. Not removing burn-in.')
            self.samples = self.sampler.get_chain(flat=True)
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
