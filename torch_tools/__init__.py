"""Init for `torch_tools.`."""
from pathlib import Path as _path

from .datasets import DataSet
from .models import FCNet
from .models import ConvNet2d
from .models import UNet
from .models import Encoder2d
from .models import Decoder2d
from .models import AutoEncoder2d
from .models import SimpleConvNet2d


# pylint: disable=unspecified-encoding
with open(_path(__file__).parent.parent / "VERSION.txt", "r") as ver:
    __version__ = ver.read().strip()
