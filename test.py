import torch
a = torch.ones((3,4))
b=a[0,:]
# class mytensor:
#     def __init__(self):
#         pass
#     def __getitem__(self, item):
#         pass
# c = mytensor()
# c[0,0:3]
if b._base is not None:
    print(b._base)

    a.__setitem__((0,0),2)
    print(b)
    print(b._base)
else:
    print('none')