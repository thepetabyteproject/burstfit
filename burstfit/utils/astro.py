import numpy as np


def dedisperse(data, dm, tsamp, freqs):
    """
    Function to dedisperse the data

    Args:
        data: Frequency-time array to dedisperse
        dm: Dispersion Measure to dedisperse at
        tsamp: Sampling time in seconds
        freqs: Frequencies array (MHz)

    Returns:
        dedispersed: Dedispersed array
        delay_bins: Delay in number of bins
        delay_time: Delay times (s)

    """
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
    """
    Function to correct for dispersion within a time sample.

    Args:
        dedispersed_model: Dedispersed FT array
        delay_time: Delay times in seconds
        delay_bins: Delays in number of bins
        tsamp: Sampling time (s)

    Returns:
        dedispersed_model_corrected: Dedispersed and corrected array

    """
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


def radiometer(tsys, gain, bw, w):
    """
    Calculates the radiometer noise

    Args:
        tsys: Receiver Tsys in Kelvin
        gain: Receiver gain in K/Jy
        bw: Bandwidth of the data or burst (in Hz)
        w: Tsamp (s)

    Returns:

    """
    return tsys / (gain * (2 * w * bw) ** (1 / 2))
