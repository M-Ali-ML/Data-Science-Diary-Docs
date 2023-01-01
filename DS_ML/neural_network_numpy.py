import numpy as np
import pandas as pd

def init_params(layer_dim):
    np.random.seed(2)
    params = {}
    L = len(layer_dim)
    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dim[l], layer_dim[l - 1]) / np.sqrt(layer_dim[l-1]) # * 0.01
        params["b" + str(l)] = np.zeros((layer_dim[l], 1))
    return params


def ReLU(Z):
    A = np.maximum(0, Z)
    act_cache = Z
    return A, act_cache

def d_ReLU(dA, act_cache):
    Z = act_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    A = np.exp(Z - np.max(Z))
    A = A / A.sum(axis=0, keepdims=True)
    act_cache = Z
    return A, act_cache


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size,len(np.unique(Y))))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def for_prop(X, params):
    caches = []
    A = X
    L = len(params) //2 
    
    for l in range(1, L):
        A_prev = A
        Z = params["W" + str(l)].dot(A_prev) + params["b" + str(l)]
        A, act_cache = ReLU(Z)
        lin_cache = (A_prev, params["W" + str(l)], params["b" + str(l)])
        cache = (lin_cache, act_cache)
        caches.append(cache)

    # last layers softmax
    Z = params["W" + str(L)].dot(A) + params["b" + str(L)]
    lin_cache = (A, params["W" + str(L)], params["b" + str(L)])
    AL, act_cache = softmax(Z)
    cache = (lin_cache, act_cache)
    caches.append(cache)

    return AL, caches


def calc_cost(AL, Y):
    m = Y.shape[1]
    Y = one_hot(Y)
    cost = np.mean(-(1. / m) * (np.dot(Y, np.log(AL).T) + np.dot((1 - Y), np.log(1 - AL).T)))
    return cost
    

def back_prop(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]

    # last layer backward
    curr_cache = caches[L - 1]
    lin_cache, act_cache = curr_cache
    one_hot_Y = one_hot(Y)
    dAL = AL - one_hot_Y
    grads["dW" + str(L)] = 1. / m * np.dot(dAL, lin_cache[0].T)
    grads["db" + str(L)] = 1. / m * np.sum(dAL, axis=1, keepdims=True)
    grads["dA" + str(L - 1)] = np.dot(lin_cache[1].T, dAL)

    for l in reversed(range(L - 1)):
        curr_cache = caches[l]
        lin_cache, act_cache = curr_cache
        dZ = d_ReLU(grads["dA" + str(l + 1)], act_cache)
        grads["dW" + str(l + 1)] = 1. / m * np.dot(dZ, lin_cache[0].T)
        grads["db" + str(l + 1)] = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        grads["dA" + str(l)] = np.dot(lin_cache[1].T, dZ)
    
    return grads

def update(params, grads, lr):
    L = len(params) // 2

    for l in range(L):
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - lr * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - lr * grads["db" + str(l + 1)]

    return params

def predict(X, Y, params, subset='data'):
    m = X.shape[1]
    probas, _ = for_prop(X, params)
    probas = np.argmax(probas, 0)
    acc = np.sum(probas == Y) / m
    print(f"Accuracy of {subset} is = {acc * 100:.4f}%")

    return probas


def load_train():
    orig_train = pd.read_csv("datasets/MNIST/train.zip", compression='zip')

    orig_train = np.array(orig_train)
    np.random.shuffle(orig_train)

    train = orig_train[5000:].T
    X_train = train[1:train.shape[1]]
    X_train = X_train / 255.
    Y_train = train[0:1]

    val = orig_train[:5000].T
    X_val = val[1:val.shape[1]]
    X_val = X_val / 255.
    Y_val = val[0:1]

    print(f"X_train shape is {X_train.shape}") 
    print(f"Y_train shape is {Y_train.shape}")
    print(f"X_val shape is {X_val.shape}")
    print(f"Y_val shape is {Y_val.shape}")

    return X_train, Y_train, X_val, Y_val

def NN(X, Y, layer_dim, lr=0.01, num_iter=500, verbose=True):
    costs = []
    params = init_params(layer_dim)

    for i in range(0, num_iter):
        AL, cashes = for_prop(X, params)
        cost = calc_cost(AL,Y)
        grads = back_prop(AL, Y, cashes)
        params = update(params, grads, lr)

        if verbose and i % 100 == 0 or i == num_iter - 1:
            print(f"Cost after iteration {i}: {cost}")
        if i % 100 or i == num_iter:
            costs.append(cost)

        
    return params, costs


# train model

layer_dim = [784,24,18,10]  
X_train, Y_train, X_val, Y_val = load_train()
params, costs = NN(X_train, Y_train, layer_dim, lr=0.03, num_iter=500, verbose=True)
pred_train = predict(X_train, Y_train, params, 'train')
pred_val = predict(X_val, Y_val, params, 'val')




