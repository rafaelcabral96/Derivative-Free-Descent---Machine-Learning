import numpy as np
from sklearn.metrics import log_loss, f1_score
from utils import convert_prob_into_class

def init_layers(nn_architecture, N_nn, seed = 99, dispersion_factor = 6):
    # random seed initiation
    np.random.seed(seed)
    # number of layers in our neural network
    number_of_layers = len(nn_architecture)
    # parameters storage initiation
    Weights = []
    Bias = []
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        
        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        # initiating the values of the W matrix
        # and vector b for subsequent layers
        W = np.random.randn(N_nn,layer_output_size, layer_input_size) * np.sqrt(dispersion_factor/(layer_input_size + layer_output_size))
              
        b = np.random.randn(N_nn,layer_output_size, 1) * np.sqrt(dispersion_factor/(layer_input_size + layer_output_size))
        
        Weights.append(W)
        Bias.append(b)


    return Weights, Bias


def activation(Z_curr, activation="relu"):
    # calculation of the input value for the activation function

    # selection of activation function
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    elif activation is "softmax":
        activation_func = softmax
    elif activation is "linear":
        activation_func = linear
    elif activation is "step":
        activation_func = step
    else:
        raise Exception('Non-supported activation function')
        
    # return of calculated activation A and the intermediate Z matrix
    return activation_func(Z_curr)

def full_forward_propagation(X, W, b, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # transfer the activation from the previous iteration
        A_prev = A_curr
        
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        W_curr = W[idx]
        # extraction of b for the current layer
        b_curr = b[idx]
        # calculation of activation for the current layer

        #ijk -> particle,output vector,sample

        if idx > 0:
            Z_curr = np.einsum('ijk,ikl->ijl',W_curr,A_prev) + b_curr
        else:
            Z_curr = np.einsum('ijk,kl->ijl',W_curr,A_prev) + b_curr

        A_curr = activation(Z_curr, activ_function_curr)
       
    return A_curr


def get_cost_value(Y_hat, Y, type = 'binary_cross_entropy'):
    
    if type == 'binary_cross_entropy':
        m = Y_hat.shape[1]
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    elif type == 'cross_entropy':
        cost = log_loss(Y,np.transpose(Y_hat))
    elif type == 'cross_entropy_T':
        cost = log_loss(np.transpose(Y),np.transpose(Y_hat))
    elif type == 'error_binary_classification':
        cost = 1.0-(Y_hat == Y).mean()
    elif type == "rmse":    #try this first
        cost =  np.mean( (Y - Y_hat)**2)      
    else:
        raise Exception('No Cost type found')

    return np.squeeze(cost)


def get_accuracy_value(Y_hat, Y, type = 'binary_accuracy'):
    
    if type == 'binary_accuracy':
        Y_hat_ = convert_prob_into_class(Y_hat)
        acc = (Y_hat_ == Y).all(axis=0).mean()
    elif type == 'binary_accuracy2':
        acc = (Y_hat == Y).mean()
    elif type == 'f1_score':
        acc = f1_score(np.argmax(Y_hat, 1).astype('int32'), np.argmax(Y, 1).astype('int32'), average='macro')
    elif type == "rmse":
        acc =  np.mean( (Y - Y_hat)**2)   
    else:
        raise Exception('No Accuracy type found')

    return acc


#ACTIVATION FUNCTIONS---------------------------------------
def linear(Z):
    return Z

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def step(Z):
    step = np.ones(Z.shape)
    step[Z<0] = 0
    return step

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / e_Z.sum(axis=0) 

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ
