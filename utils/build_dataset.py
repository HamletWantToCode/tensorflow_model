import numpy as np 
import pickle
from sklearn.model_selection import train_test_split

with open('../data/density_in_k_2x', 'rb') as f:
    data = pickle.load(f)

with open('../data/potential_in_k_2x', 'rb') as f1:
    p_data = pickle.load(f1)

total_data = np.c_[data, -p_data[:, 1:]]
train_data, test_data = train_test_split(total_data, test_size=0.2, random_state=39493)

np.save('../data/train', train_data)
np.save('../data/test', test_data)