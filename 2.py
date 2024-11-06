import numpy as np

a = np.array([[1, 8], [0, -1], [1, 999], [1, -999]])
b = np.array([[88], [0], [-1], [9]])
ma = -99999
mi = 99999
for i in b:
    if max(i) > ma:
        ma = max(i)
    if min(i) < mi:
        mi = min(i)
stepp = (ma - mi) / 10000