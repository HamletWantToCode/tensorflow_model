import tensorflow as tf 

class PCA(object):
    def __init__(self, n_components):
        self.n_cmps = n_components

    def _fit(self, X):
        self._mean = tf.reduce_mean(X, axis=0)
        X -= self._mean
        _, _, V = tf.linalg.svd(X, name='SVD')
        self._tr_mat = V[:, :self.n_cmps]

    def transform(self, X):
        X -= self._mean
        X_tr = tf.matmul(X, self._tr_mat)
        return X_tr

    def fit_transform(self, X):
        self._fit(X)
        return self.transform(X)

    def transform_gradient(self, dz_dx):
        return tf.matmul(dz_dx, self._tr_mat)

    def inverse_transform_gradient(self, dz_dy):
        return tf.matmul(dz_dy, tf.transpose(self._tr_mat))