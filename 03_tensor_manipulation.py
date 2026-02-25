import torch
"""

view() => Works only if tensor is contiguous (continuous memory)
          After transpose / permute → becomes non-contiguous → view()
          view always shares memory so for y = x.view() changes in y will also reflect on x  

reshape() => y = x.reshape(3,4) makes only when copy non-contiguous and works for all tensors (slightly slower) 
             If contiguous → NO copy → same memory (acts like view)
             If non-contiguous → makes a copy → new memory 
             Slightly slower only when copy happens     

stack([x,y],dim=0) => join tensors along a NEW dimension / puts tensors on top of each other + does increase dimensions

cat([x,y],dim=0) => extends the same line + does NOT increase dimensions

unsqueeze() => We only wrapped the whole data inside one extra outer bracket.

squeeze() => We removed the extra bracket.

"""
#--------------------------------VIEW()--------------------------------#

x = torch.arange(12)
print(x)
print(x.shape)          # torch.Size([12])

y = x.view(3, 4)
print(y)
print(y.shape)          # torch.Size([3, 4])

y = x.view(4, -1)       # PyTorch calculates -1 automatically
print(y)
print(y.shape)          # torch.Size([4, 3])

y[0,0] =101
print(x)
#--------------------------------RESHAPE()--------------------------------#

a = torch.arange(0,10)
b = a.reshape(2,5)
print(b)

#--------------------------------STACK()--------------------------------#
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

y = torch.stack([a, b]) #same as torch.stack([a, b],dim=0)
print(y)
print(y.shape)
y = torch.stack([a, b], dim=1)
print(y)
print(y.shape)

#--------------------------------CAT()--------------------------------#
c = torch.cat([a, b], dim=0)
print(c)                        # => tensor([1, 2, 3, 4, 5, 6])

#--------------------------------unsqueeze()--------------------------------#

x = torch.tensor([10, 20, 30])
x.unsqueeze(0)  # => [[10, 20, 30]]     1 row, 3 columns
x.unsqueeze(1)  # => [[10],[20],[30]]   3 rows, 1 column


#--------------------------------squeeze()--------------------------------#

x = torch.tensor([[10, 20, 30]])
x.squeeze()