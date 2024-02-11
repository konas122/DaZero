from dazero.core import Variable
from dazero.core import Parameter
from dazero.core import Function
from dazero.core import using_config
from dazero.core import no_grad
from dazero.core import as_array
from dazero.core import as_variable
from dazero.core import setup_variable
from dazero.core import Config
from dazero.layers import Layer
from dazero.models import Model

import dazero.utils
import dazero.functions
import dazero.optimizers
import dazero.datasets

setup_variable()

__version__ = '0.0.1'
