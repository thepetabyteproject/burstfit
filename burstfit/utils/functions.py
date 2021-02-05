import logging

import numpy as np
from scipy import special

from burstfit.utils.astro import dedisperse, finer_dispersion_correction

logger = logging.getLogger(__name__)


def gauss(x, S, mu, sigma):
    """
    Gaussian function with area S

    Args:
        x: input array to evaluate the function
        S: Area of the gaussian
        mu: mean of the gaussian
        sigma: sigma of the gaussian

    Returns:

    """
    if (np.array([S, mu, sigma]) < 0).sum() > 0:
        return np.zeros(len(x))
    return (S / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -(1 / 2) * ((x - mu) / sigma) ** 2
    )


def gauss_norm(x, mu, sig):
    """
    Gaussian function of unit area

    Args:
        x: input array
        mu: center of the gaussian
        sig: sigma of gaussian

    Returns:

    """
    return (1 / (np.sqrt(2 * np.pi) * sig)) * np.exp(-(1 / 2) * ((x - mu) / sig) ** 2)


def gauss_norm2(x, mu1, sig1, mu2, sig2, amp1):
    """
    Two gaussian functions of unit total area

    Args:
        x: input array
        mu1: mean of gaussian1
        sig1: sigma of gaussian1
        mu2: mean of gaussian2
        sig2: sigma of gaussian2
        amp1: amplitude of gaussian1

    Returns:

    """
    return amp1 * gauss_norm(x, mu1, sig1) + (1 - amp1) * gauss_norm(x, mu2, sig2)


def gauss_norm3(x, mu1, sig1, mu2, sig2, mu3, sig3, amp1, amp2):
    """
    Three gaussian functions of unit total area

    Args:
        x: input array
        mu1: mean of gaussian1
        sig1: sigma of gaussian1
        mu2: mean of gaussian2
        sig2: sigma of gaussian2
        mu3: mean of gaussian3
        sig3: sigma of gaussian3
        amp1: amplitude of gaussian1
        amp2: amplitude of gaussian2

    Returns:

    """
    return (
            amp1 * gauss_norm(x, mu1, sig1)
            + amp2 * gauss_norm(x, mu2, sig2)
            + (1 - amp1 - amp2) * gauss_norm(x, mu3, sig3)
    )


def pulse_fn(t, S, mu, sigma, tau):
    """

    Function of the pulse profile: Gaussian convolved with an exponential tail
    (see https://arxiv.org/pdf/1404.6593.pdf, equation 4, for more details)

    Args:
        t: input array
        S: Area of the pulse (fluence)
        mu: mean of gaussian
        sigma: sigma of gaussian
        tau: scattering timescale

    Returns:

    """
    if (np.array([S, mu, sigma, tau]) < 0).sum() > 0:
        return np.zeros(len(t))
    if sigma / tau > 6:
        p = gauss(t, S, mu, sigma)
    else:
        A = S / (2 * tau)
        B = np.exp((1 / 2) * (sigma / tau) ** 2)
        ln_C = -1 * (t - mu) / tau
        D = 1 + special.erf((t - (mu + (sigma ** 2) / tau)) / (sigma * np.sqrt(2)))
        m0 = D == 0
        ln_C[m0] = 0
        p = A * D * B * np.exp(ln_C)
    return p


def sgram_fn(
        metadata,
        pulse_function,
        spectra_function,
        spectra_params,
        pulse_params,
        other_params,
):
    """
    Spectrogram function

    Args:
        metadata: Some useful metadata (nt, nf, dispersed_at_dm, tsamp, fstart, foff, clip_fac)
        pulse_function: Function to model pulse
        spectra_function: Function to model spectra
        spectra_params: Dictionary with spectra parameters
        pulse_params: Dictionary with pulse parameters
        other_params: list of other params needed for this function (eg: [dm])

    Returns:

    """
    nt, nf, dispersed_at_dm, tsamp, fstart, foff, clip_fac = metadata
    #     dm, tau_idx = other_params
    [dm] = other_params
    tau_idx = 4
    nt = int(nt)
    nf = int(nf)
    freqs = fstart + foff * np.linspace(0, nf - 1, nf)
    chans = np.arange(nf)
    times = np.arange(nt)
    spectra_from_fit = spectra_function(chans, **spectra_params)  # nu_0, nu_sig)

    model = np.zeros(shape=(nf, nt))
    if "tau" in pulse_params.keys():
        tau = pulse_params["tau"]
        p_params = pulse_params
        for i, freq in enumerate(freqs):
            tau_f = tau * (freq / fstart) ** (-1 * tau_idx)  # tau is defined at fstart
            p_params["tau"] = tau_f
            p = pulse_function(times, **p_params)
            model[i, :] += p
    else:
        for i, freq in enumerate(freqs):
            p = pulse_function(times, **pulse_params)
            model[i, :] += p

    model_dm = dispersed_at_dm - dm

    dedispersed_model, delay_bins, delay_time = dedisperse(
        model, model_dm, tsamp, freqs
    )

    dedispersed_model_corrected = finer_dispersion_correction(
        dedispersed_model, delay_time, delay_bins, tsamp
    )
    model_final = dedispersed_model_corrected * spectra_from_fit[:, None]
    if clip_fac != 0:
        model_final = np.clip(model_final, 0, clip_fac)
    return model_final
