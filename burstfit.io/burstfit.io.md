<a name="burstfit.io"></a>
# burstfit.io

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/io.py#L1)

<a name="burstfit.io.BurstIO"></a>
## BurstIO Objects

```python
class BurstIO()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/io.py#L23)

I/O class to save the fitting results and read results to reproduce model.

**Arguments**:

- `burstfit_obj` - Instance of burstfit class with fitting parameters
- `burstdata_obj` - Instance of burstdata class with burst data
- `dictionary` - dictionary with fitting results
- `jsonfile` - JSON file with the fitting results
  outname:
  outdir:

<a name="burstfit.io.BurstIO.set_attributes_to_save"></a>
#### set\_attributes\_to\_save

```python
 | def set_attributes_to_save()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/io.py#L68)

Sets required attributes to be saved

**Returns**:


<a name="burstfit.io.BurstIO.save_results"></a>
#### save\_results

```python
 | def save_results(outname=None, outdir=None)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/io.py#L117)

Saves results of parameter fitting

**Arguments**:

- `outname` - name of the output json file
  

**Returns**:


<a name="burstfit.io.BurstIO.read_json_and_precalc"></a>
#### read\_json\_and\_precalc

```python
 | def read_json_and_precalc(file=None)
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/io.py#L162)

Read the result json file and calculate required parameters.

**Arguments**:

- `file` - results file to read
  

**Returns**:


<a name="burstfit.io.BurstIO.set_metadata"></a>
#### set\_metadata

```python
 | def set_metadata()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/io.py#L190)

Sets the metadata tuple

**Returns**:


<a name="burstfit.io.BurstIO.set_classes_from_dict"></a>
#### set\_classes\_from\_dict

```python
 | def set_classes_from_dict()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/io.py#L206)

Sets models and required classes

**Returns**:


<a name="burstfit.io.BurstIO.model"></a>
#### model

```python
 | @property
 | def model()
```

[[view_source]](https://github.com/thepetabyteproject/burstfit/blob/dc85c0cff44e1449b8d9cf13ea1a6d76604d258f/burstfit/io.py#L251)

Function to make the model

**Returns**:

  2D array of model

