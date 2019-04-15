import tensorflow as tf
import numpy as np
from exp_quadr_kernel_hessian import ExponentiatedQuadraticHessian

np.random.seed(343)

X = np.random.rand(5, 3)
Y = np.random.rand(2, 3)
amplitude = tf.constant([1.0, 1.0, 4.0], dtype=np.float64)
length_scale = tf.constant([5.0, 2.0, 7.0], dtype=np.float64)
kernel = ExponentiatedQuadraticHessian(amplitude, length_scale)
K = kernel.matrix(X, Y)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('../summary/kernel', sess.graph)
    K_mat = sess.run(K)[1]
    K_ref = np.load('kernel.npy')
    np.testing.assert_array_almost_equal(K_ref, K_mat, 8)
    writer.close()
