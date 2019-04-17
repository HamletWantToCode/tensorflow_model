import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import positive_semidefinite_kernels as tfk 
import numpy as np
import bunch

from transform import PCA
from utils import DataGenerator

config = bunch.Bunch(
    {'amplitude': 1.0,
     'length_scale': 10.0,
     'n_features': 500,
     'n_components': 10,
     'n_iterations': 1000,
     'learning_rate': 0.05}
)

train_file = '../data/new_train.npy'
observed_data = DataGenerator(train_file, config.n_features)

with tf.name_scope('data'):
    X = tf.placeholder(dtype=tf.float64, shape=[None, config.n_features], name='features')
    y = tf.placeholder(dtype=tf.float64, shape=[None], name='targets')

with tf.name_scope('PCA'):
    pca = PCA(config.n_components)
    Xt = pca.fit_transform(X)

with tf.name_scope('hyperparameters'):
    sigma = tf.Variable(initial_value=config.amplitude,
                        name='sigma',
                        dtype=np.float64)
    lambda_ = tf.Variable(initial_value=config.length_scale,
                          name='lambda',
                          dtype=np.float64)
    amplitude = (np.finfo(np.float64).tiny +
                tf.nn.softplus(sigma))
    length_scale = (np.finfo(np.float64).tiny +
                    tf.nn.softplus(lambda_))

with tf.name_scope('function_process'):
    with tf.name_scope('kernel'):
        kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)
    with tf.name_scope('neg-loglike'):
        function_GP = tfd.GaussianProcess(
                kernel=kernel,
                mean_fn=lambda x: tf.reduce_mean(y, keep_dims=True),
                index_points=Xt,
                observation_noise_variance=0.,
                jitter=1e-10
            )
        log_ll = function_GP.log_prob(y)
        tf.summary.scalar('log_likelihood', log_ll)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        train_step = optimizer.minimize(-log_ll)

merged = tf.summary.merge_all()
saver = tf.train.Saver({'hyperparameter/amplitude': sigma, 'hyperparameter/length_scale': lambda_})

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('../summary/train', sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(1, config.n_iterations+1):
        _, summary = sess.run([train_step, merged],
            feed_dict={X: observed_data.feature, y: observed_data.target1}
        )
        if (i==1) or (i%100 == 0):
            loglik_v, s, l = sess.run([log_ll, amplitude, length_scale],
                feed_dict={X: observed_data.feature, y: observed_data.target1}
            )
            print('step:%d, loglik=%.4e, sigma=%.4e, lamba=%.4f' %(i, loglik_v, s, l))
        train_writer.add_summary(summary, i)
    saver.save(sess, '../summary/checkpoint/model.ckpt')
    train_writer.close()
