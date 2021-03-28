#!/usr/bin/env python3

import logging
from contextlib import closing
from multiprocessing import Pool

import emcee
import numpy as np

from burstfit.utils.plotter import plot_mcmc_results, autocorr_plot

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
        save_results: Save MCMC samples to a file
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
        self.save_results = save_results
        self.autocorr = None
        self.pos = None
        self.max_prior = None
        self.min_prior = None
        self.set_initial_pos()
        self.set_priors()

    @property
    def ndim(self):
        """
        Returns the number of dimensions.

        Returns:
            number of dimensions

        """
        return len(self.initial_guess)

    def lnprior(self, params):
        """
        Prior function. Priors are uniform from (1-prior_range)*initial_guess to (1+prior_range)*initial_guess.
        Minimum prior for tau is set to 0.

        Args:
            params: Parameters to check.

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
        Log probability function.

        Args:
            params: Parameters to evaluate at.

        Returns:
            Prior + log likelihood at the inputs.

        """
        lp = self.lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlk(inps=params)

    def lnlk(self, inps):
        """
        Log likelihood function. Uses the model_function to generate the model.

        Args:
            inps: Parameters to evaluate at.

        Returns:
            Log likelihood.

        """
        model = self.model_function([0], *inps)
        return -0.5 * np.sum(((self.sgram.ravel() - model) / self.std) ** 2)

    def set_initial_pos(self):
        """
        Function to set the initial values of walkers and prior ranges.
        Minimum prior for tau is set to 0.

        Returns:

        """
        logging.info("Setting initial positions for MCMC.")
        logger.info(f"Initial guess for MCMC is: {self.initial_guess}")

        if self.nwalkers < 2 * self.ndim:
            logger.warning(
                "Number of walkers is less than 2*ndim. Setting nwalkers to 2*ndim+1."
            )
            self.nwalkers = 2 * self.ndim + 1

        pos = [
            np.array(self.initial_guess)
            * np.random.uniform(
                1 - self.start_pos_dev, 1 + self.start_pos_dev, size=self.ndim
            )
            for i in range(self.nwalkers)
        ]
        self.pos = np.array(pos)
        return self

    def set_priors(self):
        """
        Set priors for MCMC

        Returns:

        """
        logger.info("Setting priors for MCMC.")
        self.max_prior = (1 + self.prior_range) * self.initial_guess
        self.min_prior = (1 - self.prior_range) * self.initial_guess

        tau_idx = [i for i, t in enumerate(self.param_names) if "tau" in t]
        if len(tau_idx):
            max_tau = np.max(np.take(self.max_prior, tau_idx))

        sig_t_idx = [i for i, t in enumerate(self.param_names) if "sigma_t" in t]
        if len(sig_t_idx):
            max_sigma_t = np.max(np.take(self.max_prior, sig_t_idx))

        S_idx = [i for i, t in enumerate(self.param_names) if "S" in t]

        mu_f_idx = [i for i, t in enumerate(self.param_names) if "mu_f" in t]
        sigma_f_idx = [i for i, t in enumerate(self.param_names) if "sigma_f" in t]

        nf, nt = self.sgram.shape

        if len(tau_idx) > 0:
            logger.info(
                "Found tau in param_names. Setting its min value of prior to 0."
            )
            self.min_prior[tau_idx] = 0

        if len(sig_t_idx) > 0:
            logger.info(
                "Found sigma_t in param_names. Setting its min value of prior to 0."
            )
            self.min_prior[sig_t_idx] = 0

        if len(sig_t_idx) > 0 and len(tau_idx) > 0:
            logger.info(
                f"Found sigma_t and tau in param_names. Setting its max value of prior to "
                f"2*(max_tau_prior({max_tau}) + max_sigma_t_prior({max_sigma_t}))"
            )
            self.max_prior[tau_idx] = 2 * (max_sigma_t + max_tau)
            self.max_prior[sig_t_idx] = 2 * (max_sigma_t + max_tau)

        if len(S_idx) > 0 and len(sig_t_idx) > 0:
            logger.info(
                f"Found S and sigma_t in param_names. Setting its max value of prior to "
                f"500*max(ts)*max_sigma_t_prior. Setting its min value of prior to 0."
            )
            self.max_prior[S_idx] = 500 * np.max(self.sgram.sum(0)) * max_sigma_t
            self.min_prior[S_idx] = 0

        if len(mu_f_idx) > 0:
            logger.info(
                f"Found mu_f in param_names. Setting its priors to (-2*nf, 3*nf)"
            )
            self.min_prior[mu_f_idx] = -2 * nf
            self.max_prior[mu_f_idx] = 3 * nf

        if len(sigma_f_idx) > 0:
            logger.info(
                f"Found sigma_f in param_names. Setting its priors to (0, 5*nf)"
            )
            self.min_prior[sigma_f_idx] = 0
            self.max_prior[sigma_f_idx] = 5 * nf

        return self

    def run_mcmc(self):
        """
        Runs the MCMC.

        Returns:
            Sampler object

        """
        logger.debug(
            f"Range of initial positions of walkers (min, max): ({self.pos.min(0)}, {self.pos.max(0)})"
        )
        logger.debug(
            f"Range of priors (min, max): ({(1 - self.prior_range) * self.initial_guess},"
            f"{(1 + self.prior_range) * self.initial_guess})"
        )

        if self.save_results:
            backend = emcee.backends.HDFBackend(f"{self.outname}_samples.h5")
            backend.reset(self.nwalkers, self.ndim)
        else:
            backend = None

        index = 0
        autocorr = np.zeros(self.nsteps)
        old_tau = np.inf

        logger.info(
            f"Running MCMC with the following parameters: nwalkers={self.nwalkers}, "
            f"nsteps={self.nsteps}, start_pos_dev={self.start_pos_dev}, ncores={self.ncores}, "
            f"skip={self.skip}"
        )

        logger.info("Priors used in MCMC are:")
        for j, p in enumerate(self.param_names):
            logger.info(f"{p}: [{self.min_prior[j]}, {self.max_prior[j]}]")

        with closing(Pool(self.ncores)) as pool:
            sampler = emcee.EnsembleSampler(
                self.nwalkers, self.ndim, self.lnprob, pool=pool, backend=backend
            )
            for sample in sampler.sample(
                self.pos, iterations=self.nsteps, progress=True, store=True
            ):
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
        Returns the chanins from sampler object after removing some samples for burn-in.

        Args:
            skip: Number of steps to skip for burn-in.

        Returns:
            Sample chain.

        """
        if not skip:
            skip = self.skip
        tau = self.sampler.get_autocorr_time(tol=0)
        if np.isnan(tau).sum() == 0:
            burnin = int(2 * np.max(tau))
            thin = int(0.5 * np.min(tau))
            if burnin < skip:
                skip = burnin
        else:
            thin = 0
            logger.warning(
                "Autocorrelation time is nan. Not using tau for burn-in calculation."
            )

        logger.info(f"Discarding {skip} steps/iterations.")
        if skip > self.sampler.iteration:
            logger.warning(f"Not enough steps in chain to skip. Not removing burn-in.")
            skip = 0
        if thin == 0:
            self.samples = self.sampler.get_chain(flat=True, discard=skip)
        else:
            self.samples = self.sampler.get_chain(flat=True, discard=skip, thin=thin)
        return self.samples

    def print_results(self):
        """
        Prints the results of MCMC analysis. It uses median values with 1-sigma errors based on MCMC posteriors.

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
            logger.info(f"{self.param_names[i]}: {p[i]} + {e2[i]:.3f} - {e1[i]:.3f}")

    def plot(self, save=False):
        """
        Plot the samples and corner plot of MCMC posteriors.

        Args:
            save: To save the corner plot.

        Returns:

        """
        logger.info("Plotting MCMC results.")
        plot_mcmc_results(
            self.samples, self.outname, self.initial_guess, self.param_names, save
        )

    def make_autocorr_plot(self, save=False):
        """
        Make autocorrelation plot for MCMC (i.e autocorrelation  time scale vs iteration)
        see https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

        Args:
            save: To save the plot

        Returns:

        """
        index = (self.autocorr > 0).sum()
        n = 100 * np.arange(1, index + 1)
        y = self.autocorr[:index]
        autocorr_plot(n, y, self.outname, save)
