# MemCNN
a [PyTorch](http://pytorch.org/) Framework for Developing Memory Efficient Deep Invertible Networks

Reference: Sil C. van de Leemput, Jonas Teuwen, Rashindra Manniesing. [MemCNN: a Framework for Developing Memory Efficient Deep Invertible Networks](https://openreview.net/forum?id=r1KzqK1wz). *International Conference on Learning Representations (ICLR) 2018 Workshop Track. (https://iclr.cc/)*

## Licencing

This repository comes with the MIT license, which implies everyone has the right to use, copy, distribute and/or modify this work. If you do, please cite our work.

## Installation

### Using NVIDIA docker
#### Requirements
* NVIDIA graphics card and the proper NVIDIA-drivers on your system
* [nvidia-docker](https://github.com/nvidia/nvidia-docker) installed on your system

The following bash commands will clone this repository and do a one-time build of the docker image with the right environment installed:
```bash
git clone https://github.com/silvandeleemput/memcnn.git
docker build ./memcnn/docker --tag=memcnn-docker
```

After the one-time install on your machine, the docker can be invoked by:
```bash
docker run --shm-size=4g --runtime=nvidia -it memcnn-docker
```
This will open a preconfigured bash shell, which is correctly configured to run the experiments from the next section.

The datasets and experimental results will be put inside the created docker container under:
`\home\user\data` and `\home\user\experiments` respectively


### Using a Custom environment
#### Requirements
* [PyTorch](http://pytorch.org/) 0.4 (0.3 downwards compatible, CUDA support recommended)
* [torchvision](https://github.com/pytorch/vision) 0.1.9
* [TensorboardX](https://github.com/lanpa/tensorboard-pytorch) 0.9

Clone the repository and navigate to the right folder to execute the experiments:
```bash
git clone https://github.com/silvandeleemput/memcnn.git
cd ./memcnn/memcnn
```
Note that the location of the cloned repository has to be added to your Python path.

## Example usage: ReversibleBlock

```python
# some required imports
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import memcnn.models.revop


# define a new class of operation(s) PyTorch style
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


# generate some random input data (b, c, y, x)
data = np.random.random((2, 10, 8, 8)).astype(np.float32)
X = Variable(torch.from_numpy(data))

# application of the operation(s) the normal way
Y = ExampleOperation(channels=10)(X)

# application of the operation(s) using the reversible block
F, G = ExampleOperation(channels=10 // 2), ExampleOperation(channels=10 // 2)
Y = memcnn.models.revop.ReversibleBlock(F, G)(X)

````


## Run PyTorch Experiments
```bash
./train.py [MODEL] [DATASET] --fresh
```
Available values for `DATASET` are `cifar10` and `cifar100`.

Available values for `MODEL` are `resnet32`, `resnet110`, `resnet164`, `revnet38`, `revnet110`, `revnet164`


If not available datasets are automatically downloaded.

## Results
TensorFlow results were obtained from [the reversible residual network](https://arxiv.org/abs/1707.04585)
running the code from their [GitHub](https://github.com/renmengye/revnet-public).

<table>
<tr><th>            </th><th colspan="4"> TensorFlow        </th><th colspan="4"> PyTorch     </th></tr>
<tr><th>            </th><th colspan="2"> Cifar-10        </th><th th colspan="2"> Cifar-100        </th><th th colspan="2"> Cifar-10       </th><th th colspan="2"> Cifar-100          </th></tr>
<tr><th> Model      </th><th> acc.      </th><th> time  </th><th> acc.      </th><th> time   </th><th> acc.      </th><th> time    </th><th> acc.      </th><th> time    </th></tr>
<tr><td> resnet-32  </td><td> 92.74     </td><td> 2:04  </td><td> 69.10     </td><td> 1:58   </td><td> 92.86     </td><td> 1:51    </td><td> 69.81     </td><td> 1:51    </td></tr>
<tr><td> resnet-110 </td><td> 93.99     </td><td> 4:11  </td><td> 73.30     </td><td> 6:44   </td><td> 93.55     </td><td> 2:51    </td><td> 72.40     </td><td> 2:39    </td></tr>
<tr><td> resnet-164 </td><td> 94.57     </td><td> 11:05 </td><td> 76.79     </td><td> 10:59  </td><td> 94.80     </td><td> 4:59    </td><td> 76.47     </td><td> 3:45    </td></tr>
<tr><td> revnet-38  </td><td> 93.14     </td><td> 2:17  </td><td> 71.17     </td><td> 2:20   </td><td> 92.8     </td><td> 2:09    </td><td> 69.9     </td><td> 2:16    </td></tr>
<tr><td> revnet-110 </td><td> 94.02     </td><td> 6:59  </td><td> 74.00     </td><td> 7:03   </td><td> 94.1     </td><td> 3:42    </td><td> 73.3     </td><td> 3:50    </td></tr>
<tr><td> revnet-164 </td><td> 94.56     </td><td> 13:09 </td><td> 76.39     </td><td> 13:12  </td><td> 94.9     </td><td> 7:21    </td><td> 76.9     </td><td> 7:17    </td></tr>
</table>

The PyTorch results listed were recomputed on June 11th 2018, and differ from the results in the paper.
The Tensorflow results are still the same.

## Future Releases

* Support for other reversible networks
* Better support for non volume preserving mappings

## Citation

If you use our code, please cite:

```bibtex
@inproceedings{
  leemput2018memcnn,
  title={MemCNN: a Framework for Developing Memory Efficient Deep Invertible Networks},
  author={Sil C. van de Leemput, Jonas Teuwen, Rashindra Manniesing},
  booktitle={ICLR 2018 Workshop Track},
  year={2018},
  url={https://openreview.net/forum?id=r1KzqK1wz},
}
```
