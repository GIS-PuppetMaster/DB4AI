import re

import torch
import numpy as np
from time import time

variable_name_reg = '[a-zA-Z_]+[a-zA-Z0-9_]*'
data_shape_reg = f'^[(]([1-9][0-9]*,|-1,)+([1-9][0-9]*|-1)?[)]'

test = 'xxx(4,3)'
match = re.search(data_shape_reg, test)
if match:
    print(match.group())
else:
    print()
