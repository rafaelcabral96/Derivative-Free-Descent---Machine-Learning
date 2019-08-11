import numpy as np
import time
from utils import flatten_weights, flatten_weights_gradients, unflatten_weights, get_var


def update_cloud_derivative_free(cloudf, cost, lr, N, kernel_a, alpha, beta, gamma):
  
    #compute mean and standart deviation and difference matrix between particles
    cloud_mean = np.mean(cloudf, axis=0)
    cloud_var = get_var(cloudf) #np.var(cloudf, axis=0) works best
    params_diff_matrix = cloudf[:,np.newaxis] - cloudf
    
    #compute kernels
    norm = np.sum(params_diff_matrix**2, axis= 2) 
    kernels = np.exp(-kernel_a*norm)
    gkernels = -2*kernel_a*np.einsum('ijk,ij -> ijk',params_diff_matrix,kernels)

    cost = np.squeeze(np.array(cost))
    omega = np.divide(cloudf-cloud_mean,cloud_var)
    
   # Q = np.einsum('ijk,j -> ik', gkernels, cost) + np.einsum('ij,jk,j -> ik', kernels, omega, cost)
   # 
   # if alpha > 0 :
   #     R = np.einsum('ij,jk -> ik',kernels,cloudf-cloud_mean)
   # else:
   #     R = 0
   # 
   # if beta > 0 :
   #     P = np.einsum('ijk -> ik',gkernels) 
   # else:
   #     P = 0
   # 
   # if gamma > 0 :
   #     S = np.einsum('ij,jk -> ik', kernels,omega)
   # else:
   #     S = 0

   # cloudf -= lr * (Q + alpha*R + beta*P + gamma*S) * float(1/N)

    gamma1 = gamma
    gamma2 = alpha

    Q =  ( np.einsum('ij,jk -> ik', kernels, np.einsum('j,jk -> jk',(cost + gamma1),omega) + gamma2*(cloudf-cloud_mean) ) +
           np.einsum('j,ijk -> ik', cost + gamma1, gkernels) ) * float(1/N)

 #   if lr == "auto":
 #       lr =  (lr * N) / np.einsum('ij -> i',kernels)
 #       cloudf -= np.einsum('i,ik -> ik',lr,Q)
 #   else:

    cloudf -= lr*Q

    return cloudf, cloud_var



def update_cloud(cloudf, gradientsf, lr, N, kernel_a, alpha, beta, gamma):

    #compute mean and standart deviation and difference matrix between particles
    cloud_mean = np.mean(cloudf, axis=0)
    cloud_var = get_var(cloudf) 
    params_diff_matrix = cloudf[:,np.newaxis] - cloudf
    
    #compute kernels
    norm = np.sum(params_diff_matrix**2, axis=2) #no sqrt
    kernels = np.exp(-kernel_a*norm)
    gkernels = -2*kernel_a*np.einsum('ijk,ij -> ijk',params_diff_matrix,kernels)

    Q = np.einsum('ij,jk -> ik', kernels, gradientsf) * float(1/N) 

    if alpha > 0 :
        R = np.einsum('ij,jlk -> ik',kernels,params_diff_matrix) * float(1/N**2)
    else:
        R = 0

    if beta > 0 :
        P = np.einsum('ijk -> ik',gkernels) * float(1/N)
    else:
        P = 0

    if gamma > 0 :
        S = np.einsum('ij,jk -> ik', kernels, np.divide(cloudf-cloud_mean,cloud_var)) * float(1/N)
    else:
        S = 0

    cloudf -= lr* (Q + alpha*R + beta*P + gamma*S)

    return cloudf, cloud_var



def update_nn_weights(cloud, gradients, lr, N,  kernel_a, alpha, beta, gamma):
    
    #get cloud (flattened weights), gradients, nn shape and weight names
    cloudf, gradientsf, nn_shape, weight_names = flatten_weights_gradients(cloud, gradients, N)
    
    #get updated cloud and its variance
    cloudf, cloud_var = update_cloud(cloudf, gradientsf, lr, N, kernel_a, alpha, beta, gamma) 

    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(cloudf,nn_shape,weight_names,N)
    
    return new_nn_weights, cloud_var


def update_nn_weights_derivative_free(cloud, cost, lr, N, kernel_a, alpha, beta, gamma):
   
    #get cloud (flattened weights), gradients, nn shape and weight names
    cloudf, nn_shape, weight_names = flatten_weights(cloud,N)

    #get updated cloud and its variance
    cloudf, cloud_var = update_cloud_derivative_free(cloudf, cost, lr, N, kernel_a, alpha, beta, gamma)

    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(cloudf,nn_shape,weight_names,N)
    
    return new_nn_weights, cloud_var


def update_gd(params_values, grads_values, nn_architecture, learning_rate):
    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return [params_values], 1000


def update_gd_func(cloud, gradients, learning_rate):
    cloud -= learning_rate*gradients
    return cloud, np.nan
