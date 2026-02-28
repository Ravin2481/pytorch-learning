import torch 
from torch import nn 
import matplotlib.pyplot as plt
from tensor_04_pytorch_linear_regression import X_test,plot_predictions

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True,dtype=torch.float))       # Learnable parameter

        self.bias = nn.Parameter(torch.randn(1, requires_grad=True,dtype=torch.float))          # Learnable parameter

     # Forward defines the computation in the model - forward is fix name
    def forward(self, x: torch.Tensor) -> torch.Tensor:     # "x" is the input data 
        return self.weights * x + self.bias                 # linear regression formula (y = m*x + b)
    
model = LinearRegressionModel()

#Both are used during inference (prediction time) to stop PyTorch from tracking gradients
with torch.inference_mode():        # preferred over "with torch.no_grad():"=> Bcuz more optimized version of no_grad() for inference.
    y_pred = model(X_test)

print(y_pred)
plot_predictions(predictions=y_pred)
plt.show()


"""

input → prediction → error → gradient → weight update → better prediction

Flow:
    model = LinearRegressionModel ()

    y_pred = model(x)        # internally call model.forward(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()    # clear old grads
    loss.backward()          # compute new grads
    optimizer.step()         # update weights

    model.state_dict()       # returns a dictionary of all learnable parameters and buffers of the model.

"""