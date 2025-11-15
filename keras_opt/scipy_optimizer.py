""" Optimize a keras model using scipy.optimize
"""
import numpy as np
from scipy.optimize import minimize
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K  # pylint: disable=import-error

# Try to import data_adapter from different locations for compatibility
try:
    from tensorflow.python.keras.engine import data_adapter
except (ImportError, ModuleNotFoundError):
    try:
        from keras.src.trainers import data_adapter
    except (ImportError, ModuleNotFoundError):
        # Fallback for very new versions
        from keras.trainers import data_adapter


class ScipyOptimizer():
    """ Implements a training function that uses scipy optimize in order
        to determine the weights for the model.

        The minimize function expects to be able to attempt multiple solutions
        over the model. It calls a function which collects all gradients for
        all steps and then returns the gradient information to the optimizer.
    """

    # Methods that require Hessian-vector products
    HESSIAN_METHODS = {'trust-ncg', 'trust-krylov', 'trust-exact', 'newton-cg'}

    def __init__(self, model, method='cg', verbose=1, maxiter=1, **optimizer_kwargs):
        self.model = model
        self.method = method
        self.verbose = verbose
        self.maxiter = maxiter
        self.optimizer_kwargs = optimizer_kwargs  # Additional scipy.optimize options
        
        # Cache for Hessian computation
        self._cached_iterator = None
        self._cached_grads = None
    
    def _predict(self, x_data, training=True):
        """Call model for prediction without recursion issues.
        
        We don't store a tf.function reference at init time because that can
        cause circular dependencies when model.train_function is assigned later.
        Instead, we call the model directly each time.
        """
        # Use the model's layers directly to avoid any train_function references
        return self.model(x_data, training=training)

    def _update_weights(self, x):
        x_offset = 0
        for var in self.model.trainable_variables:
            shape = var.shape
            w_size = np.prod(shape)
            value = np.array(x[x_offset:x_offset+w_size]).reshape(shape)
            var.assign(value)
            x_offset += w_size
        assert x_offset == len(x)

    def _fun_generator(self, x, iterator):
        """ Function optimized by scipy minimize.

            Returns function cost and gradients for all trainable variables.
        """
        model = self.model
        self._update_weights(x)
        losses = []

        dataset = iterator._dataset  # pylint:disable=protected-access
        assert dataset is not None
        
        # Cache the iterator for Hessian computation
        self._cached_iterator = iterator
        
        iterator = iter(dataset)

        # Determine progress bar steps (just for display)
        try:
            size = dataset.cardinality().numpy()
            if size > 0:
                n_steps = size
            else:
                n_steps = None
        except:
            n_steps = None

        progbar = keras.utils.Progbar(n_steps, verbose=self.verbose)

        with tf.GradientTape() as tape:
            for step, data in enumerate(iterator):
                # Handle data unpacking - use direct calls like original code
                data = data_adapter.expand_1d(data)
                x_data, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
                
                y_pred = self._predict(x_data, training=True)
                loss = model.compiled_loss(y, y_pred, sample_weight,
                                           regularization_losses=model.losses)
                progbar.update(step, [('loss', loss.numpy())])
                losses.append(loss)
            xloss = tf.reduce_mean(tf.stack(losses))
            grads = tape.gradient(xloss, model.trainable_variables)

        cost = xloss.numpy()
        
        # Cache gradients for Hessian computation
        self._cached_grads = grads

        if all(isinstance(x, tf.Tensor) for x in grads):
            xgrads = np.concatenate([x.numpy().reshape(-1) for x in grads])
            return cost, xgrads

        if all(isinstance(x, tf.IndexedSlices) for x in grads):
            xgrad_list = []
            for var, grad in zip(model.trainable_variables, grads):
                value = tf.Variable(np.zeros(var.shape), dtype=var.dtype)
                value.assign_add(grad)
                xgrad_list.append(value.numpy())
            xgrads = np.concatenate([x.reshape(-1) for x in xgrad_list])
            return cost, xgrads

        raise NotImplementedError()
        return -1, np.array([])  # pylint:disable=unreachable

    def _hessp_generator(self, x, p, iterator):
        """ Compute Hessian-vector product for second-order optimization methods.
        
            The Hessian-vector product H*p is computed as the gradient of (grad^T * p)
            with respect to the model parameters.
            
            Args:
                x: Current parameter vector (not used, weights already set by _fun_generator)
                p: Direction vector for Hessian-vector product
                iterator: Data iterator (not directly used, uses cached iterator from _fun_generator)
                
            Returns:
                Hessian-vector product as a flattened numpy array
        """
        model = self.model
        
        if self._cached_iterator is None or self._cached_grads is None:
            raise RuntimeError("Hessian computation requires cached gradients from function evaluation")
        
        dataset = self._cached_iterator._dataset  # pylint:disable=protected-access
        iterator = iter(dataset)
        
        # Convert the direction vector p into the same structure as model variables
        p_vars = []
        p_offset = 0
        for var in model.trainable_variables:
            shape = var.shape
            p_size = np.prod(shape)
            p_var = tf.constant(p[p_offset:p_offset+p_size].reshape(shape), dtype=var.dtype)
            p_vars.append(p_var)
            p_offset += p_size
        
        # Compute Hessian-vector product using nested gradient tape
        hvp_list = []
        
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as outer_tape:
            with tf.GradientTape(watch_accessed_variables=True) as inner_tape:
                losses = []
                for data in iterator:
                    data = data_adapter.expand_1d(data)
                    x_data, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
                    y_pred = self._predict(x_data, training=True)
                    loss = model.compiled_loss(y, y_pred, sample_weight,
                                             regularization_losses=model.losses)
                    losses.append(loss)
                xloss = tf.reduce_mean(tf.stack(losses))
            
            # Compute gradients with respect to trainable variables
            grads = inner_tape.gradient(xloss, model.trainable_variables)
            
            # Compute the dot product of gradients with direction vector p
            grad_dot_p = tf.add_n([
                tf.reduce_sum(g * p_v) 
                for g, p_v in zip(grads, p_vars)
                if g is not None
            ])
        
        # Compute gradient of (grad^T * p) - this is the Hessian-vector product
        hvp = outer_tape.gradient(grad_dot_p, model.trainable_variables)
        
        # Clean up the persistent tape
        del outer_tape
        
        # Flatten the Hessian-vector product
        if hvp is None or any(h is None for h in hvp):
            # If Hessian is not available, return zero vector
            return np.zeros_like(p)
        
        hvp_flat = np.concatenate([h.numpy().reshape(-1) for h in hvp])
        return hvp_flat

    def train_function(self, iterator):
        """ Called by model fit.
        """
        min_options = {
            'maxiter': self.maxiter,
            'disp': bool(self.verbose),
        }
        # Merge any additional optimizer options (e.g., gtol, xtol)
        min_options.update(self.optimizer_kwargs)

        var_list = self.model.trainable_variables
        x0 = np.concatenate([x.numpy().reshape(-1) for x in var_list])

        # Use Hessian-vector product for second-order methods
        if self.method in self.HESSIAN_METHODS:
            result = minimize(
                self._fun_generator, x0, method=self.method, jac=True,
                hessp=self._hessp_generator,
                options=min_options, args=(iterator,))
        else:
            result = minimize(
                self._fun_generator, x0, method=self.method, jac=True,
                options=min_options, args=(iterator,))

        self._update_weights(result['x'])
        
        # Compute metrics on final optimized model
        model = self.model
        dataset = iterator._dataset  # pylint:disable=protected-access
        iterator_final = iter(dataset)
        
        # Reset all metrics to ensure clean state
        for metric in model.metrics:
            metric.reset_state()
        
        # Build return dictionary with loss from scipy
        return_dict = {'loss': result['fun']}
        
        # Compute final loss and metrics by doing a pass through the data
        losses = []
        for data in iterator_final:
            data = data_adapter.expand_1d(data)
            x_data, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
            y_pred = self._predict(x_data, training=False)
            
            # Compute loss for this batch
            loss = model.compiled_loss(y, y_pred, sample_weight,
                                      regularization_losses=model.losses)
            losses.append(loss.numpy())
            
            # Update compiled metrics
            model.compiled_metrics.update_state(y, y_pred, sample_weight)
        
        # Use the recomputed loss (more accurate than scipy's final value)
        if losses:
            return_dict['loss'] = float(np.mean(losses))
        
        # Get metric results (excluding loss since we already have it)
        for metric in model.metrics:
            if metric.name != 'loss':  # Skip loss to avoid overwriting
                result_value = metric.result()
                if result_value is not None:
                    return_dict[metric.name] = result_value
        
        return return_dict


def make_train_function(model, **kwargs):
    """ Returns a function that will be called to train the model.

        model._steps_per_execution must be set in order for train function to
        be called once per epoch.
    """
    # Check if model is compiled (compatible with different Keras versions)
    if hasattr(model, '_assert_compile_was_called'):
        model._assert_compile_was_called()  # pylint:disable=protected-access
    elif not model.compiled_loss:
        raise RuntimeError('Model must be compiled before creating train function')
    
    # Configure steps per execution (compatible with different Keras versions)
    if hasattr(model, '_configure_steps_per_execution'):
        model._configure_steps_per_execution(tf.int64.max)  # pylint:disable=protected-access
    
    opt = ScipyOptimizer(model, **kwargs)
    return opt.train_function
