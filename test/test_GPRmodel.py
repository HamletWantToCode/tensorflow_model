import bunch
import matplotlib.pyplot as plt 
# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
from model import GaussProcessRegression
import numpy as np 
import tensorflow as tf

def f(x):
    return np.sin(3*np.pi*x)

np.random.seed(45)
# X, y = make_regression(n_samples=500, n_features=5, random_state=43)
# y_ = y - np.mean(y)
# train_X, test_X, train_y, test_y = train_test_split(X, y_, test_size=0.2, random_state=32)
train_X = np.random.uniform(-1, 1, size=(100, 1))
train_y = f(train_X[:, 0]) + np.random.normal(0, np.sqrt(0.1), size=100)
config = {'amplitude': 1.0, 'length_scale': 1.0, 'obs_noise': 1e-10, 'learning_rate': 0.01, 'steps': 1000}
config = bunch.Bunch(config)
sess = tf.Session()
model = GaussProcessRegression(config, sess)
model.fit(train_X, train_y)

test_X = np.linspace(-1.2, 1.2, 100)
test_y = f(test_X)
pred_y, pred_var = model.predict(test_X[:, np.newaxis])
# plt.plot(test_y, test_y, 'r')
# plt.scatter(test_y, pred_y, c='b')
print(np.mean((pred_y-test_y)**2))
plt.plot(test_X, test_y, 'r')
plt.scatter(train_X, train_y, c='b')
plt.plot(test_X, pred_y, 'b')
plt.fill_between(test_X, pred_y-2*pred_var, pred_y+2*pred_var)
plt.show()