import torch 
from torch import nn 
import matplotlib.pyplot as plt
from tensor_05_pytorch_model import LinearRegressionModel 

#Take proper x & y data set 
X_train = Y_train = X_test = Y_test = torch.rand(1)

model = LinearRegressionModel

# Create the loss function
loss_fn = nn.L1Loss()           # MAE loss is same as L1Loss

# Create the optimizer - Stochastic Gradient Descent (SGD).
# params => learnable parameters to optimize (weights & bias)
# lr => learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# An epoch is one loop through the data... (this is a hyperparameter because we've set it ourselves)
epochs = 10

### Training
# 0. Loop through the data
for epoch in range(epochs):
    # Set the model to training mode
    model.train()  # train mode in PyTorch sets all parameters that require gradients to require gradients

    # 1. Forward pass
    y_pred = model(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, Y_train)

    # 3. Optimizer zero grad => set weight.grad = 0, bias.grad = 0 for fresh compute 
    optimizer.zero_grad()

    # 4. Perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step()  # by default how the optimizer changes will accumulate through the loop so... we have to zero them above in step 3

    model.eval()  # turns off gradient tracking
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, Y_test)

"""
FLOW:
    optimizer.zero_grad()   # clear old gradients
    loss.backward()         # compute new gradients
    optimizer.step()        # update weights 

HOW IT WORKS:
    When we do loss.backward() new gradients are added to old ones

    1st backword =>     weight.grad = 0.5
    2nd backword =>     weight.grad = 0.5 + 0.3 = 0.8  

    Each batch should compute a fresh gradient.
    Thats why we do     optimizer.zero_grad() then loss.backword()

    optimizer.step() => update weights by using => new_weight = old_weight - learning_rate * gradient
"""