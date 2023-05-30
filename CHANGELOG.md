## Version 0.2.0


### New Features

#### Variable kernel size
For all of the models using 2D convolutional layers (except ``ConvNet2d``, which uses predetermined architectures), the kernel size argument is now optional. Spoiler alert: it has to be an odd, positive, int.
