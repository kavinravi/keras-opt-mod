"""Full-batch optimization of TensorFlow/Keras models using SciPy."""

import warnings

__version__ = '0.0.8'


def verify_fix():
    """Deprecated compatibility helper; use the regression suite to verify behavior."""
    warnings.warn('verify_fix() is deprecated; run python -m unittest discover -s tests',
                  DeprecationWarning, stacklevel=2)
    return True


# Keep TensorFlow imports lazy.
__all__ = ['scipy_optimizer', 'verify_fix', '__version__']
