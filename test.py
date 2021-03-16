import numpy as np
import time
a = np.ones((30000,30000))
b = np.ones((30000,30000))
s = time.time()
c = a+b
print(time.time()-s)

