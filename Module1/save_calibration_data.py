import numpy as np

# Paste the matrix and distortion values from calibdb.net here:
mtx = np.array([[775.9125930005705, 0, 644.9595455757583],
                [0, 775.5281385019309, 369.0593071154218],
                [0, 0, 1]])

dist = np.array([[ -0.057788470141806526,
        0.1439962214518821,
        0.0011126329715790586,
        0.0009468680648103644,
        -0.21482517635492418]])

np.savez("calibration_data_mac.npz", mtx=mtx, dist=dist)
print("Saved calibration_data_mac.npz")
