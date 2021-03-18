<a name="burstfit.utils.math"></a>
# burstfit.utils.math

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/math.py#L1)

<a name="burstfit.utils.math.f_test"></a>
#### f\_test

```python
def f_test(x, y)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/math.py#L9)

F-Test

**Arguments**:

- `x` - Input array 1
- `y` - Input array 2
  

**Returns**:


<a name="burstfit.utils.math.tests"></a>
#### tests

```python
def tests(off_pulse, on_pulse_res, pth=0.05, ntest=1)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/math.py#L28)

Run statistical tests to compare the two inputs

**Arguments**:

- `off_pulse` - input array to compare
- `on_pulse_res` - input array to compare
- `pth` - threshold on p value to consider the distributions similar
- `ntest` - minimum number of tests to pass
  

**Returns**:


<a name="burstfit.utils.math.fmae"></a>
#### fmae

```python
def fmae(param, m, a, param_err=0, m_err=0, a_err=0)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/math.py#L59)

**Arguments**:

- `param` - parameter value
- `m` - number to multiply to param
- `a` - number to add
- `param_err` - error on the fitted parameter
- `m_err` - error on multiplier
- `a_err` - error on adder
  

**Returns**:


<a name="burstfit.utils.math.transform_parameters"></a>
#### transform\_parameters

```python
def transform_parameters(params, mapping, param_names)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/utils/math.py#L80)

Transform parameters and their errors using mapping

**Arguments**:

- `params` - dictionary with parameter values and errors in "popt" and "perr"
- `mapping` - mapping for all parameters
- `param_names` - names of parameters to be used to identify the correct mapping
  

**Returns**:


