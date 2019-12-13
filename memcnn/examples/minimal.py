import torch
import torch.nn as nn
import memcnn


# define a new torch Module with a sequence of operations: Relu o BatchNorm2d o Conv2d
class ExampleOperation(nn.Module):
    def __init__(self, channels):
        super(ExampleOperation, self).__init__()
        self.seq = nn.Sequential(
                                    nn.Conv2d(in_channels=channels, out_channels=channels,
                                              kernel_size=(3, 3), padding=1),
                                    nn.BatchNorm2d(num_features=channels),
                                    nn.ReLU(inplace=True)
                                )

    def forward(self, x):
        return self.seq(x)


# generate some random input data (batch_size, num_channels, y_elements, x_elements)
X = torch.rand(2, 10, 8, 8)

# application of the operation(s) the normal way
model_normal = ExampleOperation(channels=10)
model_normal.eval()

Y = model_normal(X)

# application of the ExampleOperation turned invertible using an additive coupling
F = ExampleOperation(channels=10 // 2)
invertible_function = memcnn.create_coupling(Fm=F, coupling='additive')

# the ReversibleModule can wrap invertible functions and
# can free memory using the invertible property during training and inference
model_invertible = memcnn.ReversibleModule(fn=invertible_function, keep_input=True, keep_input_inverse=True)
model_invertible.eval()  # This is required for any input tensor to the model with requires_grad=True (inference only)
Y2 = model_invertible(X)

# The input (X) can be approximated (X2) by applying the inverse method of the reversible block on Y2
X2 = model_invertible.inverse(Y2)
assert torch.allclose(X, X2, atol=1e-06)

# Output of the reversible block is unlikely to match the normal output of F
assert not torch.allclose(Y2, Y)
