# MemCNN 
a [PyTorch](http://pytorch.org/) Framework for Developing Memory Efficient Deep Invertible Networks

## Installation

### Using NVIDIA docker
#### Requirements
* NVIDIA graphics card and the proper NVIDIA-drivers on your system
* [nvidia-docker](https://github.com/nvidia/nvidia-docker) installed on your system

The following bash commands will clone this repository and do a one-time build of the docker image with the right environment installed:
```bash
git clone git@github.com:silvandeleemput/memcnn.git
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
* [PyTorch](http://pytorch.org/) 0.3 (CUDA support recommended)
* [torchvision](https://github.com/pytorch/vision) 0.1.9
* [TensorboardX](https://github.com/lanpa/tensorboard-pytorch) 0.9

Clone the repository and navigate to the right folder to execute the experiments:
```bash
git clone git@github.com:silvandeleemput/memcnn.git
cd ./memcnn/memcnn
```
Note that the location of the cloned repository has to be added to your Python path.

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
<tr><td> revnet-38  </td><td> 93.14     </td><td> 2:17  </td><td> 71.17     </td><td> 2:20   </td><td> 92.54     </td><td> 1:10    </td><td> 69.33     </td><td> 1:40    </td></tr>
<tr><td> revnet-110 </td><td> 94.02     </td><td> 6:59  </td><td> 74.00     </td><td> 7:03   </td><td> 93.25     </td><td> 3:43    </td><td> 72.24     </td><td> 3:44    </td></tr>
<tr><td> revnet-164 </td><td> 94.56     </td><td> 13:09 </td><td> 76.39     </td><td> 13:12  </td><td> 93.40     </td><td> 7:19    </td><td> 74.63     </td><td> 7:21    </td></tr>
</table>

## Future Releases

* Support for other reversible networks
* Better support for non volume preserving mappings

## Citation

If you use our code, please refer to this GitHub:

https://github.com/silvandeleemput/memcnn/
