import torch

# for backwards compatibility
use_context_mans = True

try:
    pytorch_version_one_and_above = int(torch.__version__[0]) > 0
except TypeError:
    pytorch_version_one_and_above = True
