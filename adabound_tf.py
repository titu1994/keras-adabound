import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2


class AdaBound(OptimizerV2):
    """AdaBound optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        final_learning_rate: float >= 0. Final learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        gamma: float >= 0. Convergence speed of the bound function.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: Weight decay weight.
        amsbound: boolean. Whether to apply the AMSBound variant of this
            algorithm.

    # References
        - [Adaptive Gradient Methods with Dynamic Bound of Learning Rate]
          (https://openreview.net/forum?id=Bkg3g2R9FX)
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond]
          (https://openreview.net/forum?id=ryQu7f-RZ)
    """
    def __init__(self,
                 learning_rate=0.001,
                 final_learning_rate=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 gamma=1e-3,
                 epsilon=None,
                 weight_decay=0.0,
                 amsbound=False,
                 name='AdaBound', **kwargs):
        super(AdaBound, self).__init__(name, **kwargs)

        self._set_hyper('learning_rate', kwargs.get('learning_rate', learning_rate))
        self._set_hyper('final_learning_rate', kwargs.get('final_learning_rate', final_learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('gamma', gamma)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsbound = amsbound
        self.weight_decay = weight_decay
        self.base_lr = learning_rate

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')
            self.add_slot(var, 'vhat')

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        vhat = self.get_slot(var, 'vhat')

        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)

        gamma = self._get_hyper('gamma')
        final_lr = self._get_hyper('final_learning_rate')

        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        base_lr_t = tf.convert_to_tensor(self.base_lr)
        t = tf.cast(self.iterations + 1, var_dtype)

        # Applies bounds on actual learning rate
        step_size = lr_t * (tf.math.sqrt(1. - tf.math.pow(beta_2_t, t)) /
                          (1. - tf.math.pow(beta_1_t, t)))

        final_lr = final_lr * lr_t / base_lr_t
        lower_bound = final_lr * (1. - 1. / (gamma * t + 1.))
        upper_bound = final_lr * (1. + 1. / (gamma * t))

        # apply weight decay
        if self.weight_decay != 0.:
            grad += self.weight_decay * var

        # Compute moments
        m_t = (beta_1_t * m) + (1. - beta_1_t) * grad
        v_t = (beta_2_t * v) + (1. - beta_2_t) * tf.math.square(grad)

        if self.amsbound:
            vhat_t = tf.math.maximum(vhat, v_t)
            denom = (tf.math.sqrt(vhat_t) + epsilon_t)
        else:
            vhat_t = vhat
            denom = (tf.math.sqrt(v_t) + self.epsilon)

        # Compute the bounds
        step_size_p = step_size * tf.ones_like(denom)
        step_size_p_bound = step_size_p / denom
        bounded_lr_t = m_t * tf.math.minimum(tf.math.maximum(step_size_p_bound,
                                             lower_bound), upper_bound)

        # Setup updates
        m_t = tf.compat.v1.assign(m, m_t)
        vhat_t = tf.compat.v1.assign(vhat, vhat_t)

        with tf.control_dependencies([m_t, v_t, vhat_t]):
            p_t = var - bounded_lr_t
            param_update = tf.compat.v1.assign(var, p_t)

            return tf.group(*[param_update, m_t, v_t, vhat_t])

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError("Sparse data is not supported yet")

    def get_config(self):
        config = super(AdaBound, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'final_learning_rate': self._serialize_hyperparameter('final_learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'gamma': self._serialize_hyperparameter('gamma'),
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'amsbound': self.amsbound,
        })
        return config
