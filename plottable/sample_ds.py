import xarray as xr
import numpy as np
temps = np.random.normal(size=(4,4))
temps += np.arange(40, -40, -20)
temperature = xr.Dataset({"lat": [0, 30, 60, 90], "long": [-90, 0, 90, 180], "temperature": (("long", "lat"), temps)})
