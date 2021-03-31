<a name="burstfit.data"></a>
# burstfit.data

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/data.py#L3)

<a name="burstfit.data.BurstData"></a>
## BurstData Objects

```python
class BurstData(Candidate)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/data.py#L11)

Class to handle burst data

**Arguments**:

- `fp` _Union[str, list]_ - String or a list of files. It can either filterbank or psrfits files.
- `dm` _float_ - Dispersion Measure of the candidate
- `tcand` _float_ - start time of the candidate in seconds at the highest frequency channel
- `width` _int_ - pulse width of the candidate in samples
- `snr` _float_ - Signal to Noise Ratio
- `min_samp` _int_ - Minimum number of time samples
- `kill_mask` _numpy.ndarray_ - Boolean mask of channels to kill
- `spectral_kurtosis_sigma` _float_ - Sigma for spectral kurtosis filter
- `savgol_frequency_window` _float_ - Filter window for savgol filter
- `savgol_sigma` _float_ - Sigma for savgol filter
- `flag_rfi` _bool_ - To turn on RFI flagging

<a name="burstfit.data.BurstData.prepare_data"></a>
#### prepare\_data

```python
 | def prepare_data(mask_chans=[], time_window=200e-3, normalise=True)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/data.py#L66)

Prepares data for burst fitting

**Arguments**:

- `mask_chans` - list with tuples (start_freq, end_freq) and channel numbers to mask
- `time_window` - time window (s) around the burst to use for burst fitting
- `normalise` - To normalise the mean and std of the data using an off pulse region
  

**Returns**:


<a name="burstfit.data.BurstData.nstart"></a>
#### nstart

```python
 | @property
 | def nstart()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/data.py#L103)

Returns: start sample number of the spectrogram

<a name="burstfit.data.BurstData.mask"></a>
#### mask

```python
 | @property
 | def mask()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/data.py#L113)

Returns: Channel mask array using all the available masks

<a name="burstfit.data.BurstData.prepare_input_mask"></a>
#### prepare\_input\_mask

```python
 | def prepare_input_mask(mask_chans=[])
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/data.py#L126)

Function to mask some frequency channels using input_mask, kill_mask and rfi_mask

**Arguments**:

- `mask_chans` - list with tuples (start_freq, end_freq) and channel numbers to mask
  

**Returns**:


<a name="burstfit.data.BurstData.mask_channels"></a>
#### mask\_channels

```python
 | def mask_channels()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/data.py#L152)

Apply channel  mask to the dedispersed data

**Returns**:


<a name="burstfit.data.BurstData.normalise_data"></a>
#### normalise\_data

```python
 | def normalise_data(on_pulse_data, off_pulse_data, return_clip_fac=True)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/data.py#L163)

Function to normalise data

**Arguments**:

- `on_pulse_data` - Data to normalise
- `off_pulse_data` - Data to use to estimate mean and std
- `return_clip_fac` - To return the clipping factor, decided using nbits of data
  

**Returns**:


<a name="burstfit.data.BurstData.crop_dedispersed_data"></a>
#### crop\_dedispersed\_data

```python
 | def crop_dedispersed_data(time_window)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/d2a59b6cca4f8d56113504e79416bde1ab64ffea/burstfit/data.py#L188)

To get a cutout of data from only around the burst

**Arguments**:

- `time_window` - time length to use on both sides of burst for the cutout
  

**Returns**:


