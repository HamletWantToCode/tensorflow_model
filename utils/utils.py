import tensorflow as tf 

def jacobi(ys, x1s, x2s):
    """
    ys: list of tensors, the output of a function
    x1s: list of tensors, points to take partial derivative
    x2s: list of tensors, points to maintain constant
    return: Jacobi, list of tensors
    """
    n_points = len(ys)
    n_col = len(x2s)
    Jacobi = []
    for i in range(n_points):
        j = i // n_col
        Jacobi.append(tf.gradients(ys[i], x1s[j])[0])
    return Jacobi