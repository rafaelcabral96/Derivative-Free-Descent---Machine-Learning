import numpy as np
import time
from utils import flatten_weights, flatten_weights_gradients, unflatten_weights, norm2, kernel, gkernel, get_var, normal_test


def update_cloud_derivative_free(paramsf, cost, lr, N_nn, kernel_a, alpha, beta, gamma):
  
    #compute mean and standart deviation and difference matrix between particles
    params_mean = np.mean(paramsf, axis=0)
    params_var = get_var(paramsf,params_mean) #np.var(paramsf, axis=0) works best
    params_diff_matrix = paramsf[:,np.newaxis] - paramsf
    
    #compute kernels
    norm = np.sum(params_diff_matrix**2, axis=2) 
    kernels = np.exp(-kernel_a*norm)
    gkernels = -2*kernel_a*np.einsum('ijk,ij -> ijk',params_diff_matrix,kernels)

    cost = np.array(cost)
    omega = np.divide(paramsf-params_mean,params_var)
    
   # Q = np.einsum('ijk,j -> ik', gkernels, cost) + np.einsum('ij,jk,j -> ik', kernels, omega, cost)
   # 
   # if alpha > 0 :
   #     R = np.einsum('ij,jk -> ik',kernels,paramsf-params_mean)
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

   # paramsf -= lr * (Q + alpha*R + beta*P + gamma*S) * float(1/N_nn)

    gamma1 = gamma
    gamma2 = alpha

    paramsf -= lr * ( np.einsum('ij,jk -> ik', kernels, np.einsum('j,jk -> jk',(cost + gamma1),omega) + gamma2*(paramsf-params_mean) ) + np.einsum('j,ijk -> ik', cost + gamma1, gkernels) ) * float(1/N_nn)

    return paramsf, params_var


def update_cloud(paramsf, gradientsf, N_nn, lr, kernel_a, alpha, beta, gamma):

    #compute mean and standart deviation and difference matrix between particles
    params_mean = np.mean(paramsf, axis=0)
    params_var = get_var(paramsf,params_mean) 
    params_diff_matrix = paramsf[:,np.newaxis] - paramsf
    
    #compute kernels
    norm = np.sum(params_diff_matrix**2, axis=2) #no sqrt
    kernels = np.exp(-kernel_a*norm)
    gkernels = -2*kernel_a*np.einsum('ijk,ij -> ijk',params_diff_matrix,kernels)

    Q = np.einsum('ij,jk -> ik', kernels, gradientsf) * float(1/N_nn) 

    if alpha > 0 :
        R = np.einsum('ij,jlk -> ik',kernels,params_diff_matrix) * float(1/N_nn**2)
    else:
        R = 0

    if beta > 0 :
        P = np.einsum('ijk -> ik',gkernels) * float(1/N_nn)
    else:
        P = 0

    if gamma > 0 :
        S = np.einsum('ij,jk -> ik', kernels, np.divide(paramsf-params_mean,params_var)) * float(1/N_nn)
    else:
        S = 0

    paramsf -= lr* (Q + alpha*R + beta*P + gamma*S)

    return paramsf, params_var



def update_nn_weights(params, gradients, N_nn, lr, kernel_a, alpha, beta, gamma):
    
    #get cloud (flattened weights), gradients, nn shape and weight names
    paramsf, gradientsf, nn_shape, weight_names = flatten_weights_gradients(params, gradients, N_nn)
    
    #get updated cloud and its variance
    paramsf, params_var = update_cloud(paramsf, gradientsf, N_nn, lr, kernel_a, alpha, beta, gamma) 

    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(paramsf,nn_shape,weight_names,N_nn)
    
    return new_nn_weights, params_var


def update_nn_weights_derivative_free(params, cost, lr, N_nn, kernel_a, alpha, beta, gamma):
    
    #get cloud (flattened weights), gradients, nn shape and weight names
    paramsf, nn_shape, weight_names = flatten_weights(params,N_nn)

    #get updated cloud and its variance
    paramsf, params_var = update_cloud_derivative_free(paramsf, cost, lr, N_nn, kernel_a, alpha, beta, gamma)

    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(paramsf,nn_shape,weight_names,N_nn)
    
    return new_nn_weights, params_var


def update_sgd(params_values, grads_values, nn_architecture, learning_rate):

    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return [params_values], 1000


#OLD VERSIONS OF UPDATE FUNCTIONS------------------------------------------------------------------------------
#Same as update_nn_weights but without using numpy broadcasting -> slower but easies to understand 
def update_nn_weights_old(params,gradients,N_nn, lr, kernel_a, alpha, beta, gamma):
             
    #get flattened weights, gradients, nn shape and weight names
    paramsf, gradientsf, nn_shape, weight_names = flatten_weights_gradients(params, gradients, N_nn)

    #compute kernels
    kernels =  [[kernel(paramsf[i],paramsf[j],kernel_a) for j in range(N_nn)]   for i in range(N_nn)]
    
    if beta > 0:
        gkernels = [[gkernel(paramsf[i],paramsf[j],kernel_a) for j in range(N_nn)]  for i in range(N_nn)]
    
    #compute mean and standart deviation
    params_mean = np.mean(paramsf, axis=0)
    params_var = get_var(paramsf,params_mean) 

    #compute gradient flows
    updates = []
    for nn in range(N_nn):    
        R = 0
        P = 0
        S = 0
        
        Q = [kernels[nn][j]*gradientsf[j] for j in range(N_nn)]
        Q = np.mean(Q,axis=0)
        
        if alpha > 0:
            R = [[kernels[nn][j]*(paramsf[j] - paramsf[k]) for j in range(N_nn)] for k in range(N_nn)]
            R = [item for sublist in R for item in sublist] #Flatten list of lists
            R = np.sum(R,axis=0)*float(1/N_nn**2)
        
        if beta > 0:
            P = [gkernels[nn][j] for j in range(N_nn)] 
            P = np.mean(P,axis=0) 
            
        if gamma > 0:
            S = [kernels[nn][j] * np.divide((paramsf[j]-params_mean),params_var) for j in range(N_nn)]
            S = np.mean(S,axis=0)
    
        updates.append( -lr*(Q + alpha*R + beta*P + gamma*S)  )
    
    #update flattened tensors
    for nn in range(N_nn):
        paramsf[nn] = paramsf[nn] + updates[nn]    

    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(paramsf,nn_shape,weight_names,N_nn)
    
    return new_nn_weights, params_var


#Same as update_nn_weights_derivative_free but without using numpy broadcasting -> slower but easier to understand 
def update_nn_weights_derivative_free_old(params, cost, lr, N_nn, kernel_a, alpha, beta, gamma):

    #get flattened weights, nn shape and weight names
    paramsf, nn_shape, weight_names = flatten_weights(params,N_nn)  
  
    #compute kernels
    kernels =  [[kernel(paramsf[i],paramsf[j],kernel_a) for j in range(N_nn)]   for i in range(N_nn)]
    gkernels = [[gkernel(paramsf[i],paramsf[j],kernel_a) for j in range(N_nn)]  for i in range(N_nn)]
    
    #plt.imshow(kernels,vmin=0,vmax=1)
    #plt.colorbar()

    #compute mean and standart deviation
    params_mean = np.mean(paramsf, axis=0)
    params_var = get_var(paramsf,params_mean) 
        
    #compute gradient flows
    updates = []
    for nn in range(N_nn):    
        R = 0
        P = 0
        S = 0
        
        Q = [gkernels[nn][j]*cost[j] + kernels[nn][j]*cost[j]*np.divide((paramsf[j]-params_mean),params_var) for j in range(N_nn)]
        Q = np.mean(Q,axis=0)
        
        if alpha > 0:
            R = [[kernels[nn][j]*(paramsf[j] - paramsf[k]) for j in range(N_nn)] for k in range(N_nn)]
            R = [item for sublist in R for item in sublist] #Flatten list of lists
            R = np.sum(R,axis=0)*float(1/N_nn**2)
        
        if beta > 0:
            P = [gkernels[nn][j] for j in range(N_nn)] 
            P = np.mean(P,axis=0) 
            
        if gamma > 0:
            S = [kernels[nn][j] * np.divide((paramsf[j]-params_mean),params_var) for j in range(N_nn)]
            S = np.mean(S,axis=0)
    
        updates.append( -lr*(Q + alpha*R + beta*P + gamma*S)  )
    
    #update flattened tensors
    for nn in range(N_nn):
        paramsf[nn] = paramsf[nn] + updates[nn]    
  
    
    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(paramsf,nn_shape,weight_names,N_nn)
    
    return new_nn_weights, params_var


#UPDATE FUNCTIONS WITH COMPUTED TIME AS OUTPUT----------------------------------------------------------------------------
def update_nn_weights_profiled(params, gradients, N_nn, lr, kernel_a, alpha, beta, gamma):
    
    #get cloud (flattened weights), gradients, nn shape and weight names
    paramsf, gradientsf, nn_shape, weight_names = flatten_weights_gradients(params, gradients, N_nn)
    
    #time needed for the update
    time_temp = time.process_time()

    #get updated cloud and its variance
    paramsf, params_var = update_cloud(paramsf, gradientsf, N_nn, lr, kernel_a, alpha, beta, gamma) 
    update_time = time.process_time() - time_temp

    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(paramsf,nn_shape,weight_names,N_nn)
    
    return new_nn_weights, params_var, update_time


def update_nn_weights_derivative_free_profiled(params, cost, lr, N_nn, kernel_a, alpha, beta, gamma):
    
    #get cloud (flattened weights), gradients, nn shape and weight names
    paramsf, nn_shape, weight_names = flatten_weights(params,N_nn)

    #time needed for the update
    time_temp = time.process_time()

    #get updated cloud and its variance
    paramsf, params_var = update_cloud_derivative_free(paramsf, cost, lr, N_nn, kernel_a, alpha, beta, gamma)
    update_time = time.process_time() - time_temp

    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(paramsf,nn_shape,weight_names,N_nn)
    
    return new_nn_weights, params_var, update_time
