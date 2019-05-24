import numpy as np

def proxOWL_part(v1, v2):
    """
    Stack-based algorithm for FastProxSL1. c.f. Bogdan's SLOPE paper, Algorithm 4
    """
    # Stack element
    class PTR(object):
        """
        stack element object: (i, j, s, w)
        """
        def __init__(self, i_, j_, s_, w_):
            self.i = i_
            self.j = j_
            self.s = s_
            self.w = w_

    # Stack-based algorithm
    v = v1 - v2
    stk = []

    for i in range(v.shape[0]):
        ptr = PTR(i, i, v[i], v[i])
        stk.append(ptr)
        while True:
            if len(stk) > 1 and stk[-2].w <= stk[-1].w:
                ptr = stk.pop()
                stk[-1].j = i
                stk[-1].s += ptr.s
                stk[-1].w = stk[-1].s / (i - stk[-1].i + 1)
            else:
                break
    
    x = np.zeros_like(v)
    for idx, ptr in enumerate(stk):
        for i in range(ptr.i, ptr.j+1):
            x[i] = max(0, ptr.w)

    return x

def proxOWL(z, mu):
    """
    Args: 
        z:  z = x_t - lr * Gradient (f(x_t)) with lr being the learning rate
        mu: mu = lr * w, where \eta is the learning rate, and w are the OWL params.
            It must be nonnegative and in non-increasing order. 
    """
    
    # Cache the signs of z
    sgn = np.sign(z)
    z = abs(z)
    idx = z.argsort()[::-1]
    z = z[idx]
    
    # Find the index of the last positive entry in vector z - mu  
    flag = 0
    n = z.size
    x = np.zeros_like(z)
    diff = (z - mu)[::-1]
    indc = np.argmax(diff>0)
    flag = diff[indc]

    # Apply prox on non-negative subsequence of z - mu
    if flag > 0:
        k = n - indc
        v1 = z[:k]
        v2 = mu[:k]
        v = proxOWL_part(v1,v2)
        x[idx[:k]] = v
    else:
        pass

    # Restore signs
    x = sgn * x

    return x

def proxLASSO(z, thresh):
    sgn = np.sign(z)
    x = np.maximum(0, np.abs(z) - thresh)
    x = sgn * x
    return x

def costOWL(beta, X, y, weight):
    """
    Cost =  1/N * (y - X * beta)^2      --------> MSE loss in data-fitting
            + (weight * beta_sort)      --------> OWL regularization term
    """

    y = y.flatten()
    cost_fit = float(np.mean((y - X.dot(beta))**2))
    cost_reg = float(np.sum(weight * np.sort(np.abs(beta))[::-1]))
    grad_fit = 2 * X.T.dot(X.dot(beta) - y) / X.shape[0]
    return cost_fit + cost_reg, grad_fit

