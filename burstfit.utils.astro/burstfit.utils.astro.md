<a name="burstfit.utils.astro"></a>
# burstfit.utils.astro

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/utils/astro.py#L1)

<a name="burstfit.utils.astro.dedisperse"></a>
#### dedisperse

```python
def dedisperse(data, dm, tsamp, freqs)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/utils/astro.py#L4)

Function to dedisperse the data

**Arguments**:

- `data` - Frequency-time array to dedisperse
- `dm` - Dispersion Measure to dedisperse at
- `tsamp` - Sampling time in seconds
- `freqs` - Frequencies array (MHz)
  

**Returns**:

- `dedispersed` - Dedispersed array
- `delay_bins` - Delay in number of bins
- `delay_time` - Delay times (s)

<a name="burstfit.utils.astro.finer_dispersion_correction"></a>
#### finer\_dispersion\_correction

```python
def finer_dispersion_correction(dedispersed_model, delay_time, delay_bins, tsamp)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/utils/astro.py#L35)

Function to correct for dispersion within a time sample.

**Arguments**:

- `dedispersed_model` - Dedispersed FT array
- `delay_time` - Delay times in seconds
- `delay_bins` - Delays in number of bins
- `tsamp` - Sampling time (s)
  

**Returns**:

- `dedispersed_model_corrected` - Dedispersed and corrected array

<a name="burstfit.utils.astro.radiometer"></a>
#### radiometer

```python
def radiometer(tsys, gain, bw, w)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/utils/astro.py#L63)

Calculates the radiometer noise

**Arguments**:

- `tsys` - Receiver Tsys in Kelvin
- `gain` - Receiver gain in K/Jy
- `bw` - Bandwidth of the data or burst (in Hz)
- `w` - Tsamp (s)
  

**Returns**:


