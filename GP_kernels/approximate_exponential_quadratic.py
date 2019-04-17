import tensorflow as tf 
from tensorflow_probability import positive_semidefinite_kernels as tfk
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util

class ApproximateExponentialQuadraticHessian(tfk.ExponentiatedQuadratic):
    def _apply(self, x1, x2, param_expansion_ndims=0):
        exponent = -0.5 * util.sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(x1, x2), self.feature_ndims)
        if self.length_scale is not None:
            length_scale = util.pad_shape_right_with_ones(
                            self.length_scale, param_expansion_ndims)
            exponent /= length_scale**2

        if self.amplitude is not None:
            amplitude = util.pad_shape_right_with_ones(
                         self.amplitude, param_expansion_ndims)
            exponent += 2. * tf.math.log(amplitude/length_scale)

        return tf.exp(exponent)