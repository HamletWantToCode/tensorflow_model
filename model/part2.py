import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import positive_semidefinite_kernels as tfk
import bunch
import numpy as np

from utils import DataGenerator
from GP_kernels import ApproximateExponentialQuadraticHessian
from transform import PCA

config = bunch.Bunch(
    {'n_features': 500,
     'n_components': 10}
)

train_file = '../data/new_train.npy'
test_file = '../data/new_test.npy'
observed_data = DataGenerator(train_file, config.n_features)
new_data = DataGenerator(test_file, config.n_features)

with tf.name_scope('data'):
    with tf.name_scope('observed_data'):
        observed_X = tf.placeholder(dtype=tf.float64, shape=[None, config.n_features], name='feature')
        observed_y = tf.placeholder(dtype=tf.float64, shape=[None], name='target1')
        observed_dz_dx = tf.placeholder(dtype=tf.float64, shape=[None, config.n_features], name='target2')
    with tf.name_scope('new_data'):
        new_X = tf.placeholder(dtype=tf.float64, shape=[None, config.n_features], name='feature')
        new_y = tf.placeholder(dtype=tf.float64, shape=[None], name='target1')
        # new_dz_dx = tf.placeholder(dtype=tf.float64, shape=[None, config.n_features], name='target2')

with tf.name_scope('hyperparameter'):
    sigma = tf.Variable(0.0, name='amplitude', dtype=tf.float64)
    lambda_ = tf.Variable(0.0, name='length_scale', dtype=tf.float64)
    amplitude = (np.finfo(np.float64).tiny +
                    tf.nn.softplus(sigma))
    length_scale = (np.finfo(np.float64).tiny +
                        tf.nn.softplus(lambda_))

with tf.name_scope('PCA'):
    pca = PCA(config.n_components)
    observed_Xt = pca.fit_transform(observed_X)
    observed_dz_dxt = pca.transform_gradient(observed_dz_dx)
    new_Xt = pca.transform(new_X)

with tf.name_scope('prediction'):
    with tf.name_scope('function_process'):
        with tf.name_scope('kernel'):
            kernel1 = tfk.ExponentiatedQuadratic(amplitude, length_scale)
        with tf.name_scope('posterior'):
            posterior1 = tfd.GaussianProcessRegressionModel(
                kernel=kernel1,
                mean_fn=lambda x: tf.reduce_mean(observed_y, keep_dims=True),
                index_points=new_Xt,
                observation_index_points=observed_Xt,
                observations=observed_y,
                observation_noise_variance=0.,
                predictive_noise_variance=0.
            )
            pred_y = posterior1.mean()
            y_variance = posterior1.variance()

    with tf.name_scope('derivative_process'):
        _init_index = tf.constant(0)
        _pred_dzt = tf.constant([], dtype=tf.float64)
        with tf.name_scope('kernel'):
            kernel2 = ApproximateExponentialQuadraticHessian(amplitude, length_scale)
        with tf.name_scope('posterior'):
            def cond(i, ys):
                return i < config.n_components
            def body(i, ys):
                y = tf.gather(observed_dz_dxt, i, axis=1)
                posterior2 = tfd.GaussianProcessRegressionModel(
                    kernel=kernel2,
                    mean_fn=lambda x: tf.reduce_mean(y, keep_dims=True),
                    index_points=new_Xt,
                    observation_index_points=observed_Xt,
                    observations=y,
                    observation_noise_variance=0.,
                    predictive_noise_variance=0.
                )
                ys = tf.concat([ys, posterior2.mean()], axis=0)
                return i+1, ys
            _, pred_dzt = tf.while_loop(cond, body, [_init_index, _pred_dzt],
                shape_invariants=[_init_index.get_shape(), tf.TensorShape([None])]
            )
            pred_dzt = tf.transpose(tf.reshape(pred_dzt, shape=(config.n_components, -1)))
        with tf.name_scope('inverse_PCA'):
            pred_dz_dx = pca.inverse_transform_gradient(pred_dzt)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, '../summary/checkpoint/model.ckpt')
    test_writer = tf.summary.FileWriter('../summary/test', sess.graph)
    # Run funciton process, predict function value
    pred_target1 = sess.run(pred_y, feed_dict={
        observed_X: observed_data.feature,
        observed_y: observed_data.target1,
        new_X: new_data.feature,
    })
    # Run derivative process, predict function derivative
    pred_target2 = sess.run(pred_dz_dx, feed_dict={
        observed_X: observed_data.feature,
        observed_dz_dx: observed_data.target2,
        new_X: new_data.feature,
    })

    import matplotlib.pyplot as plt 
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(new_data.target1, new_data.target1, 'r')
    ax1.scatter(new_data.target1, pred_target1, c='b')
    X = np.linspace(0, 1, 500)
    ax2.plot(X, new_data.target2[13], 'r')
    ax2.plot(X, pred_target2[13], 'b--')
    plt.show()



