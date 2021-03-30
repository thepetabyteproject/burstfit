<a name="burstfit.curvefit"></a>
# burstfit.curvefit

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/curvefit.py#L3)

<a name="burstfit.curvefit.CurveFit"></a>
## CurveFit Objects

```python
class CurveFit()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/curvefit.py#L11)

Simple wrapper class to handle curve fitting. It can also retry
the fitting with modified bounds if errors are encountered
or if the fitting errors are not finite.
See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
for details of inputs.

**Arguments**:

- `function` - Fitting function
- `xdata` - Input x data array (independent variable)
- `ydata` - Input y data array to fit (dependent variable)
- `bounds` - Lower and upper bounds on parameters
- `p0` - Initial guess for the parameters
- `retry` - To retry the fitting in case of RunTimeError or infinite fit errors
- `retry_frac_runtimeerror` - To set the bounds using p0 in case of RuntimeError
- `retry_frac_infinite_err` - To set the bounds using p0 in case of infinite fit errors

<a name="burstfit.curvefit.CurveFit.run_fit"></a>
#### run\_fit

```python
 | def run_fit()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/curvefit.py#L50)

Runs the fitting function and checks for errors and retries.

**Returns**:

- `popt` - List of converged parameters
- `err` - Errors on the parameters

<a name="burstfit.curvefit.CurveFit.cf"></a>
#### cf

```python
 | def cf()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/a40a655954316c842352e8fe5db91d9fb90fb38f/burstfit/curvefit.py#L94)

Do the actual curve fitting using curve_fit

**Returns**:

- `popt` - List of converged parameters
- `err` - Errors on the parameters

