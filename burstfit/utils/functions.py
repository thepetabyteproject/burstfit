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


def spectra_fn(nu, nu_0, nu_sig):
    """

    Gaussian spectra function

    Args:
        nu:
        nu_0:
        nu_sig:

    Returns:

    """
    return (1 / (np.sqrt(2 * np.pi) * nu_sig)) * np.exp(
        -(1 / 2) * ((nu - nu_0) / nu_sig) ** 2
    )


def sgram_fn(
        metadata,
        pulse_function,
        spectra_function,
        spectra_params,
        pulse_params,
        dm,
):
    """
    Spectrogram function

    Args:
        metadata: Some useful metadata (nt, nf, dispersed_at_dm, tsamp, fstart, foff, clip_fac)
        pulse_function: Function to model pulse
        spectra_function: Function to model spectra
        spectra_params: Dictionary with spectra parameters
        pulse_params: Dictionary with pulse parameters
        dm:

    Returns:

    """
    nt, nf, dispersed_at_dm, tsamp, fstart, foff, clip_fac = metadata
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
            tau_f = tau * (freq / freqs[0]) ** (-4)
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
    return model_final  # model_final.ravel()

# def model_all_components(params, *popts):
#    """
#
#    Args:
#        params:
#        *popts:
#
#    Returns:
#
#    """
#    nt, nf, dispersed_at_dm, tsamp, fstart, foff = params
#    nt = int(nt)
#    nf = int(nf)
#    m = np.zeros(shape=(nf, nt))
#    popts = list(popts)
#    for i in range(len(popts) // 7):
#        popt = popts[(i) * 7 : (i + 1) * 7]
#        m += model_sgram(xdata_params, *popt).reshape(nf, nt)
#    return m.ravel()
