import torch
x = torch.ones((100, 4))
w = torch.randn((4, ), requires_grad=True)
for _ in range(2):
    loss = torch.matmul(x, w).mean()
    loss.backward()
    print(w.grad)
    g = w.grad
    with torch.no_grad():
        w = w + w.grad*0.1
    w.requires_grad=True
