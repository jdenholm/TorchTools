# TorchTools
Handy PyTorch models and utilities which should save you time.

---

## Description
This Python library contains a bunch of neural networks, implemented in PyTorch, and a generic ``Dataset`` class with useful features to help you get started with your deep-learning problem more quickly, and spend less time engineering and testing—because researchers *always* test their code—the PyTorch-specific software you need. There is also the hope that you will write and duplicate less code by being able to easily set and modify these models' architectures at instantiation.

Please find the documentation [here](https://jdenholm.github.io/TorchTools/).

---

## Authors and Acknowledgement
This project was created by Jim Denholm. Any contributions, criticisms or suggestions are welcome.


---


## License
Distributed under the MIT License. See LICENSE.md for more information.


---


## Getting Started

---

### Installation

To clone the repository:
```bash
git clone https://github.com/jdenholm/TorchTools.git
```

To create the necessary conda environment:
```bash
cd /path/to/TorchTools/
conda env create -f requirements.conda.yaml
```

To install TorchTools:
```bash
conda activate torch-tools
pip install -e .
```

---

### Run tests
To run the tests:
```bash
pytest tests/
```

---

## Contents

---

### Models

All of the models inherit, either directly or through `torch.nn.Sequential`, from `torch.nn.Module`, and therefore function like standard PyTorch models.

---

#### Dense Network
Often people share code with simple perceptron-style networks where they have hard-coded the architecture—the number of layers, dropout probabilities and the number of input/output features, etc. Hard-coding these features is inelegant and doesn't allow one to easily modify the architecture.

`DenseNetwork` is a straightforward perceptron-style neural network that can be used for classification and regression. It is super-simple to use and instantiate.

For example:


```python
from torch_tools import DenseNetwork

DenseNetwork(
    in_feats=256,
    out_feats=2,
    hidden_sizes=(128, 64, 32),
    input_bnorm=True,
    input_dropout=0.1,
    hidden_dropout=0.25,
    hidden_bnorm=True,
    negative_slope=0.2,
)
```




    DenseNetwork(
      (0): InputBlock(
        (0): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Dropout(p=0.1, inplace=False)
      )
      (1): DenseBlock(
        (0): Linear(in_features=256, out_features=128, bias=True)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Dropout(p=0.25, inplace=False)
        (3): LeakyReLU(negative_slope=0.2)
      )
      (2): DenseBlock(
        (0): Linear(in_features=128, out_features=64, bias=True)
        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Dropout(p=0.25, inplace=False)
        (3): LeakyReLU(negative_slope=0.2)
      )
      (3): DenseBlock(
        (0): Linear(in_features=64, out_features=32, bias=True)
        (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Dropout(p=0.25, inplace=False)
        (3): LeakyReLU(negative_slope=0.2)
      )
      (4): DenseBlock(
        (0): Linear(in_features=32, out_features=2, bias=True)
      )
    )



---

#### Convolutional Network 2D


Torchvision's default models assume three-channel (RGB) images. To use a different number of channels, one has to overwrite the first convolutional layer, which results in ugly code that must be modified if you want to want to change architectures from, say, ResNet to VGG, etc. Furthermore, you may wish to experiment with the fully connected layers' architecture, which requires more ugly code to modify the default models. To circumvent this, you can use `ConvNet2d`.

`ConvNet2d` is a convolution neural network made of three parts: an encoder, an adaptive pooling layer and a `DenseNetwork` (which serves as a classification/regression head). The model can be customised in a modular fashion:

* Encoder: the encoder can be chosen with the optional argument `encoder_style`, which can be any of Torchvision's ResNet or VGG models (i.e. `"resnet18"`, `"vgg11"`, `"vgg_11_bn"` etc.).
* Pool: the adaptive pooling layer can be chosen with the `pool_style` optional argument. The options are `"avg"`, `"max"` and `"avg-max-concat"`. The former two options are simply adaptive average and max pooling layers, and the latter is a layer which concatenates the former two (thus doubling the number of features).
* `DenseNetwork`: the final part of the model is simply the dense network introduced in the previous section. The keyword arguments can be supplied by passing a dictionary to `dense_net_kwargs` optional argument.


For example, suppose we want an encoder in the style of Torchvision's VGG11 with batch normalisation, initialised with Torchvision's ImageNet pretrained weights, a pooling layer which concatenates the outputs of both the average and maximum adaptive pooling layers, a classifier with two hidden layers and a dropout probability of 0.25:


```python
from torch_tools import ConvNet2d

ConvNet2d(
    out_feats=512,
    in_channels=3,
    encoder_style="vgg11_bn",
    pretrained=True,
    pool_style="avg-max-concat",
    dense_net_kwargs={"hidden_sizes": (1024, 1024), "hidden_dropout": 0.25},
)
```




    ConvNet2d(
      (backbone): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (8): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (9): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (10): ReLU(inplace=True)
        (11): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (13): ReLU(inplace=True)
        (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (15): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (16): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (17): ReLU(inplace=True)
        (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (19): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (20): ReLU(inplace=True)
        (21): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (22): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (23): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (24): ReLU(inplace=True)
        (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (26): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (27): ReLU(inplace=True)
        (28): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (pool): Sequential(
        (0): _ConcatMaxAvgPool2d(
          (_avg_pool): AdaptiveAvgPool2d(output_size=(7, 7))
          (_max_pool): AdaptiveMaxPool2d(output_size=(7, 7))
        )
        (1): Flatten(start_dim=1, end_dim=-1)
      )
      (dense_layers): DenseNetwork(
        (0): DenseBlock(
          (0): Linear(in_features=50176, out_features=1024, bias=True)
          (1): Dropout(p=0.25, inplace=False)
          (2): LeakyReLU(negative_slope=0.2)
        )
        (1): DenseBlock(
          (0): Linear(in_features=1024, out_features=1024, bias=True)
          (1): Dropout(p=0.25, inplace=False)
          (2): LeakyReLU(negative_slope=0.2)
        )
        (2): DenseBlock(
          (0): Linear(in_features=1024, out_features=512, bias=True)
        )
      )
    )



Another useful feature of `ConvNet2d` is the ability to *freeze* the encoder: PyTorch have made available weights from pretrained models which you can leverage for a totally different problem.

By choosing `pretrained=True`, the encoder part of the model will be loaded with the most
up-to-date ImageNet pretrained weights available from Torchvision. You can then simply train the dense layers, leaving the weights of the enoder untouched.

For example:



```python
from torch import rand
from torch_tools import ConvNet2d

model = ConvNet2d(out_feats=10)

# Batch of 10 fake three-channel images of 256x256 pixels
mini_batch = rand(10, 3, 256, 256)

# With the encoder frozen
preds = model(mini_batch, frozen_encoder=True)

# Without the encoder frozen (default behaviour)
preds = model(mini_batch, frozen_encoder=False)
```

Note:
* Even if you load pretrained weights, but *don't* freeze the encoder, you will likely end up finding better performance than you would by randomly
initialising the model—even if it doesn't make sense. Welcome to deep learning.
* If you change the number of input channels, don't bother freezing the encoder—the first convolutional layer is overloaded and randomly initialised.
* See `torch_tools.models._conv_net_2d.ConvNet2d` for more info.

---

#### UNet—Semantic Segmentation
The `UNet` has become a classic model which, again, is often implemented with the architecture hard-coded. Having an easy-to-instantiate `UNet` with a readily-modifiable architecture is always handy, so we include one here.

Suppose we want a `UNet` that takes three-channel inputs, produces 16 output channels, has an initial convolutional block which produces 64 features, has three layers in the U, uses max pooling (rather than average), uses `ConvTranspose2d` layers to upsample (rather than bilinear interpolation) and has `LeakyReLU` layers with a slope of 0.2.

While this is quite a mouthful, it is incredibly easy to instantiate:


```python
from torch_tools import UNet

UNet(
    in_chans=3,
    out_chans=16,
    features_start=64,
    num_layers=3,
    pool_style="max",
    bilinear=False,
    lr_slope=0.2,
)
```




    UNet(
      (in_conv): DoubleConvBlock(
        (0): ConvBlock(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.2)
        )
        (1): ConvBlock(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.2)
        )
      )
      (down_blocks): ModuleList(
        (0): DownBlock(
          (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (1): DoubleConvBlock(
            (0): ConvBlock(
              (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.2)
            )
            (1): ConvBlock(
              (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.2)
            )
          )
        )
        (1): DownBlock(
          (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (1): DoubleConvBlock(
            (0): ConvBlock(
              (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.2)
            )
            (1): ConvBlock(
              (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.2)
            )
          )
        )
      )
      (up_blocks): ModuleList(
        (0): UNetUpBlock(
          (upsample): ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
          (double_conv): DoubleConvBlock(
            (0): ConvBlock(
              (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.2)
            )
            (1): ConvBlock(
              (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.2)
            )
          )
        )
        (1): UNetUpBlock(
          (upsample): ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
          (double_conv): DoubleConvBlock(
            (0): ConvBlock(
              (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.2)
            )
            (1): ConvBlock(
              (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.2)
            )
          )
        )
      )
      (out_conv): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
    )




---


#### Encoder


We also include a simple encoder model for image-like inputs. This model is effectively a `torch.nn.Sequential` of down-sampling blocks which half the size of the image-like inputs and double the number of channels.

For example:


```python
from torch_tools import Encoder2d

Encoder2d(
    in_chans=3,
    start_features=64,
    num_blocks=4,
    pool_style="max",
    lr_slope=0.123,
)
```




    Encoder2d(
      (0): DoubleConvBlock(
        (0): ConvBlock(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.123)
        )
        (1): ConvBlock(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.123)
        )
      )
      (1): DownBlock(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): DoubleConvBlock(
          (0): ConvBlock(
            (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
          (1): ConvBlock(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
        )
      )
      (2): DownBlock(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): DoubleConvBlock(
          (0): ConvBlock(
            (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
          (1): ConvBlock(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
        )
      )
      (3): DownBlock(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): DoubleConvBlock(
          (0): ConvBlock(
            (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
          (1): ConvBlock(
            (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
        )
      )
    )



---

#### Decoder

We also include a simple decoder model for image-like inputs. The decoder model essentially a series of up-sampling blocks which halve the number of channels in the outputs and double the height and width. It can therefore be used to "decode" images that have been encoded by, say, the "encoder".

By setting `bilinear=False`, the upsampling is done using a `torch.nn.ConvTranspose2d`,
and bilinear interpolation otherwise.

For example:


```python
from torch_tools import Decoder2d

Decoder2d(
    in_chans=128,
    out_chans=3,
    num_blocks=4,
    bilinear=False,
    lr_slope=0.123,
)
```




    Decoder2d(
      (0): UpBlock(
        (0): ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
        (1): DoubleConvBlock(
          (0): ConvBlock(
            (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
          (1): ConvBlock(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
        )
      )
      (1): UpBlock(
        (0): ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2))
        (1): DoubleConvBlock(
          (0): ConvBlock(
            (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
          (1): ConvBlock(
            (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
        )
      )
      (2): UpBlock(
        (0): ConvTranspose2d(32, 32, kernel_size=(2, 2), stride=(2, 2))
        (1): DoubleConvBlock(
          (0): ConvBlock(
            (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
          (1): ConvBlock(
            (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
        )
      )
      (3): Conv2d(16, 3, kernel_size=(1, 1), stride=(1, 1))
    )




---


#### Encoder–Decoder model
This model is effectively a UNet without the skip connections. It can be used for segmentation and style transfer in the same way, however a roughly equivalent UNet will likely outperform it (depending on the problem, of course). That said, it carries the advantage that it requires less memory during training and inference, and is faster computationally.

The spatial dimensions of the image-like inputs (height and width) are necesseraliy preserved.



```python
from torch_tools import EncoderDecoder2d

EncoderDecoder2d(in_chans=3, out_chans=3)
```




    EncoderDecoder2d(
      (encoder): Encoder2d(
        (0): DoubleConvBlock(
          (0): ConvBlock(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1)
          )
          (1): ConvBlock(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1)
          )
        )
        (1): DownBlock(
          (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (1): DoubleConvBlock(
            (0): ConvBlock(
              (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.1)
            )
            (1): ConvBlock(
              (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.1)
            )
          )
        )
        (2): DownBlock(
          (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (1): DoubleConvBlock(
            (0): ConvBlock(
              (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.1)
            )
            (1): ConvBlock(
              (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.1)
            )
          )
        )
        (3): DownBlock(
          (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (1): DoubleConvBlock(
            (0): ConvBlock(
              (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.1)
            )
            (1): ConvBlock(
              (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.1)
            )
          )
        )
      )
      (decoder): Decoder2d(
        (0): UpBlock(
          (0): ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
          (1): DoubleConvBlock(
            (0): ConvBlock(
              (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.1)
            )
            (1): ConvBlock(
              (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.1)
            )
          )
        )
        (1): UpBlock(
          (0): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
          (1): DoubleConvBlock(
            (0): ConvBlock(
              (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.1)
            )
            (1): ConvBlock(
              (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.1)
            )
          )
        )
        (2): UpBlock(
          (0): ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
          (1): DoubleConvBlock(
            (0): ConvBlock(
              (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.1)
            )
            (1): ConvBlock(
              (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.1)
            )
          )
        )
        (3): Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
      )
    )



---



#### Simple 2D convolution network
While we've already included a two-dimensional convolutional model, it uses default architectures from Torchvision's available VGG and ResNet models, which are pretty hefty and might be too "overpowered" for certain applications (say, to be used as a discriminator when training a GAN). To address this, we also include a simpler two-dimensional convolutional neural network which is a lot more lightweight and customisable.

For example:


```python
from torch_tools import SimpleConvNet2d

SimpleConvNet2d(
    in_chans=3,
    out_feats=128,
    features_start=64,
    num_blocks=4,
    downsample_pool="max",
    adaptive_pool="avg-max-concat",
    lr_slope=0.123,
)
```




    SimpleConvNet2d(
      (0): Encoder2d(
        (0): DoubleConvBlock(
          (0): ConvBlock(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
          (1): ConvBlock(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.123)
          )
        )
        (1): DownBlock(
          (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (1): DoubleConvBlock(
            (0): ConvBlock(
              (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.123)
            )
            (1): ConvBlock(
              (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.123)
            )
          )
        )
        (2): DownBlock(
          (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (1): DoubleConvBlock(
            (0): ConvBlock(
              (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.123)
            )
            (1): ConvBlock(
              (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.123)
            )
          )
        )
        (3): DownBlock(
          (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (1): DoubleConvBlock(
            (0): ConvBlock(
              (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.123)
            )
            (1): ConvBlock(
              (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.123)
            )
          )
        )
      )
      (1): _ConcatMaxAvgPool2d(
        (_avg_pool): AdaptiveAvgPool2d(output_size=(1, 1))
        (_max_pool): AdaptiveMaxPool2d(output_size=(1, 1))
      )
      (2): Flatten(start_dim=1, end_dim=-1)
      (3): Linear(in_features=1024, out_features=128, bias=True)
    )




---

### Datasets
Now that we have these fancy neural networks to play with, we need a good way of supplying them with data. The standard way to do this in PyTorch to to create a Dataset class—specifically a class which inherits from `torch.utils.data.Dataset`, whose `__getitem__` method yields the inputs for your model.

Add stuff here

---

#### DataSet



