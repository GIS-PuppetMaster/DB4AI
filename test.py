import re
import sys
import numpy
import torch
import pandas as pd
import numpy as np
from time import time
class MixedTensor(object):
    def __init__(self, data, relational=False, **kwargs):
        if relational:
            self._t = None
            self._mixed_data = data
        else:
            self._t = torch.as_tensor(data, **kwargs)
            self._mixed_data = self._t
        self._relational = relational
        self.init_kwargs = kwargs

    def __repr__(self):
        return "relational:\n{}\n\ndata:\n{}\n\nmixed_data:{}\n".format(self._relational, self._t, self._mixed_data)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if self._relational:
            self._t = torch.as_tensor(self._mixed_data.values, **self.init_kwargs)
        tmp_args= []
        for a in args:
            if not isinstance(a, MixedTensor):
                tmp_args.append(a)
            else:
                tmp_args.append(torch.as_tensor(a._mixed_data.values, **a.init_kwargs))
        args = tmp_args
        ret = func(*args, **kwargs)
        if self._relational:
            self._t = None
        return MixedTensor(ret)


# noinspection PyMissingConstructor
class MixedSubTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x, extra_data, *args, **kwargs):
        return object.__new__(cls, x, *args, **kwargs)

    def __init__(self, x, extra_data):
        self.extra_data = extra_data
        self.requires_grad=False
        self.grad_fn=None
        self.grad=None
        self.shape=None

    def clone(self, *args, **kwargs):
        return MixedSubTensor(super().clone(*args, **kwargs), self.extra_data)

    def to(self, *args, **kwargs):
        new_obj = MixedSubTensor([], self.extra_data)
        tempTensor = super().to(*args, **kwargs)
        new_obj.data = tempTensor.data
        new_obj.requires_grad = tempTensor.requires_grad
        return new_obj

    def release_data(self):
        self.T = None

d = MixedSubTensor(torch.ones((1000,2)), extra_data='info')
# print(sys.getsizeof(d))
# d.release_data()
# print(sys.getsizeof(d))
d.requires_grad=True
a = torch.ones((10, 2))
# b = MixedTensor(pd.DataFrame(numpy.ones((2,1))),relational=True,dtype=torch.float32)
b = torch.ones((2,1))
c = torch.sum(torch.matmul(d, b))
c.grad_fn = None
torch.autograd.backward(c)

print(d.grad)

# variable_name_reg = '[a-zA-Z_]+[a-zA-Z0-9_]*'
# data_shape_reg = f'^[(]([1-9][0-9]*,|-1,)+([1-9][0-9]*|-1)?[)]'
#
# test = 'xxx(4,3)'
# match = re.search(data_shape_reg, test)
# if match:
#     print(match.group())
# else:
#     print()
