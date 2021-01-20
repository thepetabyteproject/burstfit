import numpy as np


def dedisperse(data, dm, tsamp, freqs):
    nf, nt = data.shape
    assert nf == len(freqs)
    delay_time = 4148808.0 * dm * (1 / (freqs[0]) ** 2 - 1 / (freqs) ** 2) / 1000
    delay_bins = np.round(delay_time / tsamp).astype("int64")
    dedispersed = np.zeros(data.shape, dtype=np.float32)
    for ii in range(nf):
        dedispersed[ii, :] = np.concatenate(
            [
                data[ii, -delay_bins[ii] :],
                data[ii, : -delay_bins[ii]],
            ]
        )
    return dedispersed, delay_bins, delay_time


def finer_dispersion_correction(dedispersed_model, delay_time, delay_bins, tsamp):
    delay_remaining = delay_time / tsamp - delay_bins
    dedispersed_model_corrected = np.zeros(dedispersed_model.shape)
    for i in range(dedispersed_model_corrected.shape[0]):
        r = delay_remaining[i]
        assert np.abs(r) < 1
        if r < 0:
            l = np.correlate(dedispersed_model[i, :], [1 + r, -1 * r], mode="same")
            l = np.roll(l, -1)
        else:
            l = np.correlate(dedispersed_model[i, :], [r, 1 - r], mode="same")
        dedispersed_model_corrected[i] = l
    return dedispersed_model_corrected
