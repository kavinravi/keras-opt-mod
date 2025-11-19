"""Keras Scipy Optimizer - Full batch optimization using scipy.optimize"""

__version__ = '0.0.6'

# Don't import scipy_optimizer at module level - let users import it explicitly
# This avoids loading TensorFlow/Keras until actually needed

def verify_fix():
    """Verify that the recursion fix is present."""
    import inspect
    from keras_opt import scipy_optimizer
    
    source = inspect.getsource(scipy_optimizer.ScipyOptimizer._predict)
    
    if 'model.call(' in source:
        print(f"keras-opt v{__version__} loaded correctly")
        print("Recursion fix is PRESENT: using model.call()")
        return True
    elif 'model(' in source or 'self.model(' in source:
        print(f"keras-opt v{__version__} has WRONG code")
        print("Still using model() which causes recursion")
        return False
    else:
        print(f"? keras-opt v{__version__} - cannot verify")
        return None

__all__ = ['scipy_optimizer', 'verify_fix', '__version__']
