import torch
print(torch.__version__,"\n")

data =torch.tensor(7)
print("python number:", data.item(),"\n")        #.item() convert => tensor → Python number

x = torch.tensor([1, 2, 3])
print("Tensor:",x)                  


# zeros / ones / random     [rows,cols]
a = torch.zeros(2,3)
b = torch.ones(2,4)
c = torch.rand(2,3)                     # Uniform distribution
d = torch.randn(2, 3)                   # Normal (Gaussian) distribution  (Mean = 0, Std = 1)
e = torch.tensor([[1, 2], [3, 4]])
f = torch.zeros_like(e)                 # Creates a new tensor full of 0s same size as tensor e


data =torch.arange(0, 10, 2)
print(data,"\n")        # Creates tensor withvalues in a range with a step

print(a,"\n")
print(b,"\n")
print(c,"\n")

print(c.ndim)               #all three same a.ndim && a.ndimension && a.dim()
print(c.shape)              # (rows, columns, depth, ...)
#print(c.dim())
#print(c.ndimension())

r = torch.tensor([                              # => 2 blocks + 3 rows + 4 cols
    [[1, 2, 3,2],[4,5,0,3],[2,34,4,5]],         #[Matrix 1 (3×4), Matrix 2 (3×4)]               
    [[1, 2, 3,2],[4,5,0,3],[2,34,4,5]]
    ])

print(r.ndim)             
print(r.shape)

a = torch.tensor([10, 20, 30, 40])
print(a.shape)                          #torch.Size([4])

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(a.shape)                          #torch.Size([2, 3])

a = torch.zeros(2, 3, 4)
print(a.shape)                          #torch.Size([2, 3, 4]) => 2 blocks + 3 rows + 4 cols

a = torch.zeros(32, 3, 224, 224)
print(a.shape)                          #torch.Size([32, 3, 224, 224])

# tensor datatypes
print(a.dtype)                          #[1,2,3] =>  int64 
print(b.dtype)                          #[1.0,2.0] => float32
print(c.dtype)
print(e.dtype)

