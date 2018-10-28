#all implementation goes here

import numpy as np
import datetime
from helpers import *

import matplotlib.pyplot as plt



########## The six methods ##########
def least_squares_GD(y, tx, initial_w, max_iters, gamma):

	"""Gradient descent algorithm."""
	# Define parameters to store w and loss
	ws = [initial_w]
	losses = []
	w = initial_w

	for n_iter in range(max_iters):
		# compute loss, gradient
		grad, err = compute_gradient(y, tx, w)
		loss = 1/2*np.mean(err**2)

		# gradient w by descent update
		w = w - gamma * grad
		
		# store w and loss
		ws.append(w)
		losses.append(loss)

	#finds best parameters
	min_ind = np.argmin(losses)
	loss = losses[min_ind]
	w = ws[min_ind][:]

	return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):

	"""Stochastic gradient descent algorithm."""
	ws = [initial_w]
	losses = []
	w = initial_w
	batch_size = 1

	#iterate max_iters times, where a small batch is picked on each iteration.
	#Don't understant whyyy we do this?
	for n_iter in range(max_iters):
		for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
			grad, err = compute_gradient(minibatch_y, minibatch_tx, w)
			loss = 1/2*np.mean(err**2)

			w = w - gamma*grad

			# store w and loss
			ws.append(w)
			losses.append(loss)
			print(loss)

	#finds best parameters
	min_ind = np.argmin(losses)
	loss = losses[min_ind]
	w = ws[min_ind][:]

	return w, loss


def least_squares(y, tx):
	"""calculate the least squares solution."""

	a = tx.T.dot(tx)
	b = tx.T.dot(y)
	w = np.linalg.solve(a, b)
	e = y - tx.dot(w)
	loss = e.dot(e) / (2 * len(e))

	return w, loss


def ridge_regression(y, tx, lambda_):
	aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
	a = tx.T.dot(tx) + aI
	b = tx.T.dot(y)
	w = np.linalg.solve(a, b)
	e = y - tx.dot(w)
	loss = e.dot(e) / (2 * len(e))

	return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):

    ws = [initial_w]
    losses = []
    w = initial_w
    loss = 0
    #threshold = 1e-8
    for n_iter in range(max_iters):
        loss = sum(sum(np.logaddexp(0, tx.dot(w)) - y*(tx.dot(w))))
        prediction = sigmoid(tx.dot(w))
        gradient = tx.T.dot(prediction - y)

        # gradient w by descent update
        w = w - (gamma * gradient)
        ws.append(w)
        losses.append(loss)

        #if (len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold):
        #   break

    #finds best parameters
    min_ind = np.argmin(losses)
    loss = losses[min_ind]
    w = ws[min_ind][:]
    
    return w, loss
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
	ws = [initial_w]
	losses = []
	w = initial_w
	for n_iter in range(max_iters):
		#print(n_iter)
		# compute prediction, loss, gradient
		#tx should maybe not be transposed
		# not transposed when using large X
		loss = 0
		prediction = sigmoid(tx.dot(w))
		loss = sum(np.logaddexp(0, tx.dot(w)) - y*(tx.dot(w))+ (lambda_/2)*np.linalg.norm(w)**2)
		gradient = tx.T.dot(prediction - y) + (lambda_*np.linalg.norm(w))

		# gradient w by descent update
		w = w - (gamma * gradient)
		# store w and loss
		ws.append(w)
		losses.append(loss)

	#finds best parameters
	min_ind = np.argmin(losses)
	loss = losses[min_ind]
	w = ws[min_ind][:]
	return w, loss





##########  Data processing  ##########	
 
def standardize(x):
    """Standardize the original data set."""
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)

    return x



def remove999(x_train, pred_train, ids_train, x_test, ids_test): 

    x_train = np.concatenate((ids_train[:,None], x_train), axis=1)
    x_test = np.concatenate((ids_test[:,None], x_test), axis=1)  

    #Fix train data
    sig = x_train[pred_train == 1,:];
    back = x_train[pred_train == -1,:];

    for i in range(1,x_train.shape[1]):
        sig_mean = sum(sig[sig[:,i] != -999, i])/ len(sig[sig[:,i] != -999,i]);
        back_mean = np.sum(back[back[:,i] != -999,i])/ len(back[back[:,i] != -999,i]);
        test_mean = np.sum(x_test[x_test[:,i] != -999, i])/len(x_test[x_test[:,i] != -999,i]);
        train_mean = np.sum(x_train[x_train[:,i] != -999, i])/len(x_train[x_train[:,i] != -999,i]);

        sig[sig[:,i] == -999,i] = train_mean;
        back[back[:,i] == -999,i] = train_mean;
        x_test[x_test[:,i] == -999,i] = test_mean; 

    x_train_fixed = np.vstack((sig,back))
    x_train_fixed = x_train_fixed[x_train_fixed[:,0].argsort(),]
    
    x_train_fixed = x_train_fixed[:,1:]
    x_test_fixed = x_test[:,1:]

    return x_train_fixed, x_test_fixed


def removecols(input_data_train, input_data_test, cols):
    input_data_train = np.delete(input_data_train,cols, axis = 1)
    input_data_test = np.delete(input_data_test,cols, axis = 1)
    
    return input_data_train, input_data_test
    

def logpositive(x_train, x_test):
    for i in range(1,x_train.shape[1]):
        if (np.all(x_train[:,i]) > 0 and np.all(x_test[:,i] > 0)):
            x_train[:,i] = np.log10(x_train[:,i])
            x_test[:,i] = np.log10(x_test[:,i])

    return x_train, x_test


def findcorrelation(pred_train, sig_or_back):
    col = []
    for i in range(1,pred_train.shape[1]):
        if abs(np.cov(pred_train[:,i], sig_or_back)[0,1]) < 0.005:
            col.append(i)

    return col


########## Other implementations used ##########

def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    N = len(y)
    #tx = np.c_[np.ones(N), x]
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    return y, tx

def compute_mse(y, tx, w):
    """Compute the loss by mse."""
    e = y - tx.dot(w)
    mse =  1/(2 * len(e)) * e.dot(e)
    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0/(1 + np.exp(-t))
    #return np.exp(t)/(1 + np.exp(t))


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    ret = np.ones([len(x),1])
    for d in range (1,degree+1):

        ret = np.c_[ret,np.power(x,d)]
    return ret


def split_data(y, x, ratio, seed=10):
    np.random.seed(seed)
    N = len(y)

    index = np.random.permutation(N)
    index_tr = index[: int(np.floor(N*ratio))]
    index_te = index[int(np.floor(N*ratio)) :]
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]

    return x_tr, x_te, y_tr, y_te


def bootstrap_data(x, num_subSamp):
    temp_mean = np.zeros((1,x.shape[1]))
    temp_std = np.zeros((1,x.shape[1]))
    for i in range(num_subSamp):  
        #choosing a random subsample(with replacement) of the data with size equal to half of the sample size
        a = x[np.random.randint(x.shape[0], size=x.shape[0]//2), :]
        temp_mean += np.mean(a, axis=0)
        temp_std += np.std(a, axis=0)
    bootstrapMean = temp_mean/num_subSamp
    bootstrapStd = temp_std/num_subSamp
    return bootstrapMean, bootstrapStd


def standardize_with_bootstrapping(x,num_subSamp):
    """Standardize the original data set."""
    b_mean, b_std = bootstrap_data(x, num_subSamp)
    x -= b_mean
    x /= b_std

    return x



### USIKKER PÅ DISSE
def logistic_regression_hessian(y, tx, initial_w, max_iters, gamma):

    ws = [initial_w]
    losses = []
    w = initial_w
    loss = 0
    #threshold = 1e-8

    for n_iter in range(max_iters):

        loss = sum(sum(np.logaddexp(0, tx.dot(w)) - y*(tx.dot(w))))

        #print(loss)
        prediction = sigmoid(tx.dot(w))
        gradient = tx.T.dot(prediction - y)
        hessian = calculate_hessian(y, tx, w, prediction)

        # gradient w by descent update
        hessian_inv = inverse = np.linalg.inv(hessian)
        w = w - (hessian_inv*gradient*gamma)#np.linalg.solve(hessian, gradient)
        print(w)
        ws.append(w)
        losses.append(loss)

        #if (len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold):
        #   break

    #finds best parameters
    min_ind = np.argmin(losses)
    loss = losses[min_ind]
    w = ws[min_ind][:]
    
    return w, loss




###############KOK###########################
def calculate_hessian(y, tx, w, pred):
    """return the hessian of the loss function."""
    #pred = sigmoid(tx.dot(w))*(1-)
    pred = np.diag(pred.T[0])
    S = np.multiply(pred, (1-pred))
    XdotS = tx.T.dot(S)
    return XdotS.dot(tx)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]



def sigmoid2(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_loss_lr(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid2(tx.dot(w))
    print(pred)
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid2(tx.dot(w))
    #print(pred.shape)
    grad = tx.T.dot(pred - y)
    return grad


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss_lr(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w

