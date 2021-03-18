<a name="burstfit.utils.functions"></a>
# burstfit.utils.functions

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/functions.py#L1)

<a name="burstfit.utils.functions.gauss"></a>
#### gauss

```python
def gauss(x, S, mu, sigma)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/functions.py#L10)

Gaussian function with area S

**Arguments**:

- `x` - input array to evaluate the function
- `S` - Area of the gaussian
- `mu` - mean of the gaussian
- `sigma` - sigma of the gaussian
  

**Returns**:


<a name="burstfit.utils.functions.gauss_norm"></a>
#### gauss\_norm

```python
def gauss_norm(x, mu, sig)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/functions.py#L28)

Gaussian function of unit area

**Arguments**:

- `x` - input array
- `mu` - center of the gaussian
- `sig` - sigma of gaussian
  

**Returns**:


<a name="burstfit.utils.functions.gauss_norm2"></a>
#### gauss\_norm2

```python
def gauss_norm2(x, mu1, sig1, mu2, sig2, amp1)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/functions.py#L43)

Two gaussian functions of unit total area

**Arguments**:

- `x` - input array
- `mu1` - mean of gaussian1
- `sig1` - sigma of gaussian1
- `mu2` - mean of gaussian2
- `sig2` - sigma of gaussian2
- `amp1` - amplitude of gaussian1
  

**Returns**:


<a name="burstfit.utils.functions.gauss_norm3"></a>
#### gauss\_norm3

```python
def gauss_norm3(x, mu1, sig1, mu2, sig2, mu3, sig3, amp1, amp2)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/functions.py#L61)

Three gaussian functions of unit total area

**Arguments**:

- `x` - input array
- `mu1` - mean of gaussian1
- `sig1` - sigma of gaussian1
- `mu2` - mean of gaussian2
- `sig2` - sigma of gaussian2
- `mu3` - mean of gaussian3
- `sig3` - sigma of gaussian3
- `amp1` - amplitude of gaussian1
- `amp2` - amplitude of gaussian2
  

**Returns**:


<a name="burstfit.utils.functions.pulse_fn"></a>
#### pulse\_fn

```python
def pulse_fn(t, S, mu, sigma, tau)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/functions.py#L86)

Function of the pulse profile: Gaussian convolved with an exponential tail
(see https://arxiv.org/pdf/1404.6593.pdf, equation 4, for more details)

**Arguments**:

- `t` - input array
- `S` - Area of the pulse (fluence)
- `mu` - mean of gaussian
- `sigma` - sigma of gaussian
- `tau` - scattering timescale
  

**Returns**:


<a name="burstfit.utils.functions.pulse_fn_vec"></a>
#### pulse\_fn\_vec

```python
def pulse_fn_vec(t, S, mu, sigma, tau)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/functions.py#L117)

Vectorized implementation of pulse profile function: Gaussian convolved with an exponential tail
(see https://arxiv.org/pdf/1404.6593.pdf, equation 4, for more details)

**Arguments**:

- `t` - input array
- `S` - Area of the pulse (fluence)
- `mu` - means of gaussians for each channel
- `sigma` - sigma of gaussian
- `tau` - scattering timescale for each channel
  

**Returns**:

  2D spectrogram with pulse profiles

<a name="burstfit.utils.functions.sgram_fn"></a>
#### sgram\_fn

```python
def sgram_fn(metadata, pulse_function, spectra_function, spectra_params, pulse_params, other_params)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/functions.py#L164)

Spectrogram function

**Arguments**:

- `metadata` - Some useful metadata (nt, nf, dispersed_at_dm, tsamp, fstart, foff)
- `pulse_function` - Function to model pulse
- `spectra_function` - Function to model spectra
- `spectra_params` - Dictionary with spectra parameters
- `pulse_params` - Dictionary with pulse parameters
- `other_params` - list of other params needed for this function (eg: [dm])
  

**Returns**:


<a name="burstfit.utils.functions.sgram_fn_vec"></a>
#### sgram\_fn\_vec

```python
def sgram_fn_vec(metadata, pulse_function, spectra_function, spectra_params, pulse_params, other_params)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/functions.py#L224)

Vectorized implementation of spectrogram function. Assumes the following input names for pulse_function:
S, mu, sigma, tau

**Arguments**:

- `metadata` - Some useful metadata (nt, nf, dispersed_at_dm, tsamp, fstart, foff)
- `pulse_function` - Function to model pulse
- `spectra_function` - Function to model spectra
- `spectra_params` - Dictionary with spectra parameters
- `pulse_params` - Dictionary with pulse parameters
- `other_params` - list of other params needed for this function (eg: [dm])
  

**Returns**:


