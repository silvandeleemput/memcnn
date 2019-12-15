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

# turn the ExampleOperation invertible using an additive coupling
invertible_module = memcnn.AdditiveCoupling(
    Fm=ExampleOperation(channels=10 // 2),
    Gm=ExampleOperation(channels=10 // 2)
)

# test that it is actually a valid invertible module (has a valid inverse method)
assert memcnn.is_invertible_module(invertible_module, test_input_shape=X.shape)

# wrap our invertible_module using the InvertibleModuleWrapper and benefit from memory savings during training
invertible_module_wrapper = memcnn.InvertibleModuleWrapper(fn=invertible_module, keep_input=True, keep_input_inverse=True)

# by default the module is set to training, the following sets this to evaluation
# note that this is required to pass input tensors to the model with requires_grad=False (inference only)
invertible_module_wrapper.eval()

# test that the wrapped module is also a valid invertible module
assert memcnn.is_invertible_module(invertible_module_wrapper, test_input_shape=X.shape)

# compute the forward pass using the wrapper
Y2 = invertible_module_wrapper.forward(X)

# the input (X) can be approximated (X2) by applying the inverse method of the wrapper on Y2
X2 = invertible_module_wrapper.inverse(Y2)

# test that the input and approximation are similar
assert torch.allclose(X, X2, atol=1e-06)
