import os
import numpy as np

f = np.load(os.path.expanduser("~/Downloads/bts_input.npz"))
inet = f["interior_net"]
lsun = f["lsun"]

print (inet.shape, inet.dtype, inet.min(), inet.max())
print (lsun.shape, lsun.dtype, lsun.min(), lsun.max())

