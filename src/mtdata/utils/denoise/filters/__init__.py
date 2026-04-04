"""Denoising filters subpackage."""

# Import all filter modules to register filters
from . import moving_average
from . import spectral
from . import polynomial
from . import specialized
from . import wavelet
from . import decomposition
from . import adaptive
from . import trend

__all__ = []
