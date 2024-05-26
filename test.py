import torch

y = torch.rand((2,3,16))
z = torch.rand((2,4,16))

y_shape = list(y.shape)
y_shape[-2] = 0
x = torch.empty(y_shape)

a = torch.cat([x,y,z], dim=-2)

print(a.shape)