import numpy as np
data = np.load('x_test.npz')

for key, value in data.items():
    value= value[:4:]
    np.savetxt("somepath" + key + ".csv", value)