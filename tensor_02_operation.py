import torch

tensor = torch.tensor([1,2,3],
                      dtype=None,           # default int64/float32; can be float32, float64...
                      device=None,          # cpu/gpu/cuda
                      requires_grad=False   # tensor will learn or not
                      )

print("Tensor => ",tensor)
print("Data type => ",tensor.dtype)
print("shape of tensor => ",tensor.shape)
print("Device => ",tensor.device)


# Tensor Operations

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Addition
a + b 
torch.add(a, b)             #tensor([5, 7, 9])

# Subtraction
a - b 
torch.sub(a, b)             #tensor([-3, -3, -3])

# Multiplication
a * b 
torch.mul(a, b)             #tensor([4, 10, 18])

# Matrix Multiplication
a @ b          
print(torch.matmul(a, b))          #tensor(32) 


# Transpose 
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

A_Transpse = A.t()                  #only for 2D works 

print(A,A.shape)
print(A_Transpse,A_Transpse.shape)

# Transpose - For any dimention
A.transpose(0,1)                    # swap axis 0 with axis 1


# Finding the min, max, mean, sum, etc (tensor aggregation)

x = torch.arange(0, 100, 10)
x, x.dtype                  # (tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]), torch.int64)

# Find the min value 
torch.min(x), x.min()           # (tensor(0), tensor(0))

# Find the max 
torch.max(x), x.max()           # (tensor(90), tensor(90))

# Find the mean => note: torch.mean() requires a float32 tensor
torch.mean(x.type(torch.float32)), x.type(torch.float32).mean()     # (tensor(45.), tensor(45.))

# Find the sum
torch.sum(x), x.sum()           # (tensor(450), tensor(450))


# min, max position
t = torch.tensor([10, 5, 20, 8])

t.argmax()        # tensor(2) (index = 2)
torch.argmax(t)   # tensor(2)

t.argmin()        # tensor(1) (index = 1)
torch.argmin(t)   # tensor(1)

print(t[0],t[1],t[2])       # value => tensor(10) tensor(5) tensor(20)