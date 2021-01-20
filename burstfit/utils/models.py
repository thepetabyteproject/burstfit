import numpy as np
from scipy import special
from burstfit.utils.astro import dedisperse, finer_dispersion_correction


def gauss(x, S, mu, sigma):
    if (np.array([S, mu, sigma]) < 0).sum() > 0:
        return np.zeros(len(x))
    return (S / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -(1 / 2) * ((x - mu) / sigma) ** 2
    )


def pulse_model(t, S, mu, sigma, tau):
    # https://arxiv.org/pdf/1404.6593.pdf, equation 4
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
    #     assert np.abs(np.trapz(p) - S) < 0.10 * S
    #     except AssertionError:
    #         print('assertion error')
    #         print(S, mu, sigma, tau)
    return p


def spectra_model(nu, nu_0, nu_sig):
    return (1 / (np.sqrt(2 * np.pi) * nu_sig)) * np.exp(
        -(1 / 2) * ((nu - nu_0) / nu_sig) ** 2
    )


def sgram_model(
    metadata,
    nu_0,
    nu_sig,
    S_t,
    t_mu,
    t_sigma,
    tau,
    dm,
):
    nt, nf, dispersed_at_dm, tsamp, fstart, foff, clip_fac = metadata
    nt = int(nt)
    nf = int(nf)
    freqs = fstart + foff * np.linspace(0, nf - 1, nf)
    chans = np.arange(nf)
    times = np.arange(nt)
    spectra_from_fit = spectra_model(chans, nu_0, nu_sig)
    #     pulse_from_fit = pulse21(times, S_t, t_mu, t_sigma, tau)

    model = np.zeros(shape=(nf, nt))
    for i, freq in enumerate(freqs):
        tau_f = tau * (freq / freqs[0]) ** (-4)
        p = pulse_model(times, S_t, t_mu, t_sigma, tau_f)
        model[i, :] += p

    model_dm = dm - dispersed_at_dm

    dedispersed_model, delay_bins, delay_time = dedisperse(
        model, model_dm, tsamp, freqs
    )

    #     dedispersed_model_corrected = dedispersed_model
    dedispersed_model_corrected = finer_dispersion_correction(
        dedispersed_model, delay_time, delay_bins, tsamp
    )
    model_final = dedispersed_model_corrected * spectra_from_fit[:, None]
    if clip_fac != 0:
        model_final = np.clip(model_final, 0, clip_fac)
    return model_final.ravel()


def model_all_components(params, *popts):
    nt, nf, dispersed_at_dm, tsamp, fstart, foff = params
    nt = int(nt)
    nf = int(nf)
    m = np.zeros(shape=(nf, nt))
    popts = list(popts)
    for i in range(len(popts) // 7):
        popt = popts[(i) * 7 : (i + 1) * 7]
        m += model_sgram(xdata_params, *popt).reshape(nf, nt)
    return m.ravel()
