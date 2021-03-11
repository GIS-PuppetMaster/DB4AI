import numpy as np

a = np.zeros((2, 3, 4), dtype='str')
a[0][2][3] = '1'

print(type(a))
print(a)

