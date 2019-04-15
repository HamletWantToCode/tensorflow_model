import tensorflow as tf
from tensorflow_probability.python.positive_semidefinite_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    'ExponentiatedQuadraticHessian',
]


def _validate_arg_if_not_none(arg, assertion, validate_args):
    if arg is None:
        return arg
    with tf.control_dependencies([assertion(arg)] if validate_args else []):
        result = tf.identity(arg)
        return result


class ExponentiatedQuadraticHessian(psd_kernel.PositiveSemidefiniteKernel):
    def __init__(self,
                amplitude=None,
                length_scale=None,
                feature_ndims=1,
                validate_args=False,
                name='ExponentiatedQuadraticHessian'):

        with tf.compat.v1.name_scope(
            name, values=[amplitude, length_scale]) as name:
            dtype = dtype_util.common_dtype(
                [amplitude, length_scale])
            if amplitude is not None:
                amplitude = tf.convert_to_tensor(
                    value=amplitude, name='amplitude', dtype=dtype)
                self._amplitude = _validate_arg_if_not_none(
                    amplitude, tf.compat.v1.assert_positive, validate_args)
            if length_scale is not None:
                length_scale = tf.convert_to_tensor(
                    value=length_scale, name='length_scale', dtype=dtype)
                self._length_scale = _validate_arg_if_not_none(
                    length_scale, tf.compat.v1.assert_positive, validate_args)
        super(ExponentiatedQuadraticHessian, self).__init__(
            feature_ndims, dtype=dtype, name=name)

    @property
    def amplitude(self):
        """Amplitude parameter."""
        return self._amplitude

    @property
    def length_scale(self):
        """Length scale parameter."""
        return self._length_scale

    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return tf.broadcast_static_shape(
            scalar_shape if self.amplitude is None else self.amplitude.shape,
            scalar_shape if self.length_scale is None else self.length_scale.shape)

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(
            [] if self.amplitude is None else tf.shape(input=self.amplitude),
            [] if self.length_scale is None else tf.shape(input=self.length_scale))

    def matrix(self, x1, x2):
        with self._name_scope(self._name, values=[x1, x2]):
            x1 = tf.convert_to_tensor(value=x1, name='x1')
            x2 = tf.convert_to_tensor(value=x2, name='x2')
            x1 = tf.expand_dims(x1, -(self.feature_ndims + 1))
            x2 = tf.expand_dims(x2, -(self.feature_ndims + 2))
        return self._apply(x1, x2, param_expansion_ndims=2)

    def _apply(self, x1, x2, param_expansion_ndims=0):
        n_row = x1.shape[0].value
        n_col = x2.shape[1].value
        n_features = x1.shape[2].value

        exponent = -0.5 * util.sum_rightmost_ndims_preserving_shape(
                    tf.math.squared_difference(x1, x2), self.feature_ndims)
        if self.length_scale is not None:
            length_scale = util.pad_shape_right_with_ones(
                           self.length_scale, param_expansion_ndims)
            exponent /= length_scale**2

        if self.amplitude is not None:
            amplitude = util.pad_shape_right_with_ones(
                        self.amplitude, param_expansion_ndims)
            exponent += 2. * tf.math.log(amplitude)

        kernel = tf.exp(exponent)

        row_matrix_list = []
        for i in range(n_row):
            col_matrix_list = []
            for j in range(n_col):
                d = x1[i, 0] - x2[0, j]
                kernel_element = util.pad_shape_right_with_ones(kernel[:, i, j], param_expansion_ndims)
                M = kernel_element*(tf.eye(n_features, dtype=tf.float64) -
                                   (1./length_scale**2)*tf.matmul(tf.expand_dims(d, axis=1),
                                                                  tf.expand_dims(d, axis=0)))
                col_matrix_list.append(M)
            col_matrix = tf.stack(col_matrix_list, axis=2)
            row_matrix_list.append(col_matrix)
        Hessian = tf.stack(row_matrix_list, axis=1)
        Hessian = tf.reshape(Hessian, shape=(-1, n_row*n_features, n_col*n_features))
        Hessian *= (amplitude / length_scale)**2
        return Hessian

        # Auto differential
        # def kernel(x, y):
        #     exponent = -0.5*tf.reduce_sum(tf.squared_difference(x, y))
        #     exponent /= self.length_scale**2
        #     exponent += 2.0*tf.log(self.amplitude)
        #     return tf.exp(exponent)
        
        # n_row = len(x1)
        # n_col = len(x2)
        # n_dimension = x1[0].shape.num_elements()
        # Hessian = tf.Variable([], dtype=np.float64)
        # for i in range(n_row):
        #     x = x1[i]
        #     for j in range(n_col):
        #         y = x2[j]
        #         z = kernel(x, y)
        #         dz_dx = tf.gradients(z, y)[0]
        #         for k in range(n_dimension):
        #             Hessian = tf.concat([Hessian, tf.gradients(dz_dx, x)[0]], 0)
        # return tf.reshape(Hessian, shape=(n_row*n_dimension, n_col*n_dimension))
        