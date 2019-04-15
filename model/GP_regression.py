import tensorflow as tf 
from module import GaussianProcess
from tensorflow_probability import distributions as tfd 
from tensorflow_probability import positive_semidefinite_kernels as tfk
import numpy as np 

__all__ = ['GaussProcessRegression']

class GaussProcessRegression(object):
    def __init__(self, config, sess):
        self.config = config
        self.hyperparameters = {}
        self.kernel = None
        self.sess = sess

    def _loglike(self, X, y):
        with tf.name_scope('hyperparameters'):
            amplitude = (np.finfo(np.float64).tiny +
             tf.nn.softplus(tf.Variable(initial_value=self.config.amplitude,
                                        name='amplitude',
                                        dtype=np.float64)))
            length_scale = (np.finfo(np.float64).tiny +
                            tf.nn.softplus(tf.Variable(initial_value=self.config.length_scale,
                                                    name='length_scale',
                                                    dtype=np.float64)))
            observation_noise_variance = (
                np.finfo(np.float64).tiny +
                tf.nn.softplus(tf.Variable(initial_value=self.config.obs_noise,
                                        name='observation_noise_variance',
                                        dtype=np.float64)))
            self.hyperparameters['amplitude'] = amplitude
            self.hyperparameters['length_scale'] = length_scale
            self.hyperparameters['noise'] = observation_noise_variance
        with tf.name_scope('Gauss_prior'):
            with tf.name_scope('Kernel'):
                self.kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)
            GP_prior = tfd.GaussianProcess(
                kernel=self.kernel,
                index_points=X,
                observation_noise_variance=observation_noise_variance
            )
        self.log_ll = GP_prior.log_prob(y)

    def _posterior(self, X):
        posterior = tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=X,
            observation_index_points=self._X_fit,
            observations=self._y_fit,
            observation_noise_variance=self.hyperparameters['noise'],
            predictive_noise_variance=0.)
        self.mean = posterior.mean()
        self.variance = posterior.variance()

    def fit(self, X, y):
        self._X_fit = X
        self._y_fit = y
        self._loglike(X, y)
        with tf.name_scope('Train'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            train_step = optimizer.minimize(-self.log_ll)
        
        log_ll = tf.summary.scalar('log_likelihood', self.log_ll)
        train_writer = tf.summary.FileWriter('tmp/summary/GP', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        for i in range(self.config.steps):
            _, summary = self.sess.run([train_step, log_ll])
            if i%100 == 0:
                neg_loglik = -1*self.sess.run(self.log_ll)
                print('step:%d, neg_loglik==%.4e' %(i, neg_loglik))
            train_writer.add_summary(summary, i)
    
    def predict(self, X):
        self._posterior(X)
        pred_y = self.sess.run(self.mean)
        pred_var = self.sess.run(self.variance)
        return (pred_y, pred_var)