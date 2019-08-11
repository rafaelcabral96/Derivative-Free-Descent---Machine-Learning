import numpy as np
import time
from neural_networks import init_layers, full_forward_propagation, full_backward_propagation, get_cost_value, get_accuracy_value, timerfunc, get_mean, plot_cost, plot_list, plot_distance_matrix, kernel_a_finder
from utils import flatten_weights, flatten_weights_gradients, unflatten_weights, norm2, kernel, gkernel, get_var, normal_test



@timerfunc
def train2(X, Y, nn_architecture, epochs, learning_rate, method, n_batches, batch_size, cost_type, N, kernel_a, alpha_init, alpha_rate, beta, gamma, verbose, var_epsilon, dispersion_factor=6):
  
    if method == "sgd": 
        N = 1

    # initiation of neural net parameters
    params = [init_layers(nn_architecture,i,dispersion_factor) for i in range(N)]
    alpha = alpha_init

    # initiation of lists storing the history 
    cost_history = []
    cost_history_mean = []
    accuracy_history = []
    elapsed_epochs = 0    
       
    # performing calculations for subsequent iterations
    for i in range(epochs):
        
        for batch in range(n_batches):
   
            Y_hat = []
            costs = []
            cache = []
            grads = []

            start = batch*batch_size
            end = start + batch_size

            for j in range(N):  

                # step forward
                Y_hat_temp, cache_temp = full_forward_propagation(X[:,start:end], params[j], nn_architecture)
                Y_hat.append(Y_hat_temp)
                cache.append(cache_temp)
                
                # calculating cost and saving it to history
                costj = get_cost_value(Y_hat[j], Y[:,start:end], cost_type)
                costs.append(costj)  
                
                # step backward - calculating gradient
                if method in ["gradient","gradient_old","sgd"]:
                    gradsj =  full_backward_propagation(Y_hat[j], Y[:,start:end], cache[j], params[j], nn_architecture)
                    grads.append(gradsj)
            
            if method == "gradient":            params, var = update_nn_weights(params, grads, N, learning_rate,  kernel_a, alpha, beta, gamma)
            elif method == "gradient_old":      params, var = update_nn_weights_old(params, grads, N, learning_rate,  kernel_a, alpha, beta, gamma) 
            elif method == "nogradient":        params, var = update_nn_weights_derivative_free(params, costs, learning_rate, N, kernel_a, alpha, beta, gamma)
            elif method == "nogradient_old":    params, var = update_nn_weights_derivative_free_old(params, costs, learning_rate, N, kernel_a, alpha, beta, gamma) 
            elif method == "sgd":               params, var = update_sgd(params[0], grads[0], nn_architecture, learning_rate)
            else: raise Exception("No method found")

            #end of iteration       
            cost_history.append(costs)

            #mean position
            mean_param = get_mean(params)
            Y_hat_mean, _ = full_forward_propagation(X[:,start:end], mean_param, nn_architecture)
            cost_mean = get_cost_value(Y_hat_mean, Y[:,start:end], cost_type)
            cost_history_mean.append(cost_mean)

        #end of epoch----------------
        var_mean = np.mean(var) #mean of variances along dimensions of parameter space

        if(verbose):
            print("Iteration: {:05} - mean cost: {:.5f} - particle variance: {:.5f}".format(i, np.mean(costs), var_mean))

        alpha += alpha_rate
        elapsed_epochs += 1

        if var_mean < var_epsilon: 
            print("Convergence achieved - Particles are localized")
            break


    if not method == "sgd": 
        plot_cost(cost_history,cost_history_mean,'Training Cost Function')
        plot_list(np.mean(cost_history,axis=1),'Mean Cost Function') 
        plot_distance_matrix(params,N) 
    else:
        plot_list(cost_history,'Training Cost Function')

    print("Cost Function evaluated {:01} times".format(int(n_batches*elapsed_epochs*N)))

    return params, mean_param



@timerfunc
def train_with_profiling(X, Y, nn_architecture, epochs, learning_rate, method, n_batches, batch_size, cost_type, N, kernel_a, alpha_init, alpha_rate, beta, gamma, verbose, var_epsilon, dispersion_factor=6):
    
    import time 
    from optimizers import update_nn_weights_profiled, update_nn_weights_derivative_free_profiled   

    if method == "sgd": 
        N = 1

    profiling_time = {
        "full_forward_propagation": 0.0,
        "get_cost_value": 0.0,
        "full_backward_propagation": 0.0,
        "weights_update": 0.0,
        "weights_update_without_flattening": 0.0
        }

    # initiation of neural net parameters
    params = [init_layers(nn_architecture,i,dispersion_factor) for i in range(N)]
    alpha = alpha_init

    # initiation of lists storing the history 
    cost_history = []
    cost_history_mean = []
    accuracy_history = []
    
    elapsed_epochs = 0    
       
    # performing calculations for subsequent iterations
    for i in range(epochs):
        
        for batch in range(n_batches):
   
            Y_hat = []
            costs = []
            cache = []
            grads = []
            start = batch*batch_size
            end = start + batch_size

            for j in range(N):  

                # step forward
                full_forward_data = full_forward_propagation(X[:,start:end], params[j], nn_architecture) 
                
                time_temp = time.process_time()
                Y_hat_temp, cache_temp = full_forward_propagation(X[:,start:end], params[j], nn_architecture)
                profiling_time["full_forward_propagation"] += (time.process_time() - time_temp)    
                Y_hat.append(Y_hat_temp)
                cache.append(cache_temp)

                # calculating cost and saving it to history
                time_temp = time.process_time()
                costj = get_cost_value(Y_hat[j], Y[:,start:end], cost_type) 
                profiling_time["get_cost_value"] += (time.process_time() - time_temp)   
                costs.append(costj)  

                # step backward - calculating gradient
                if method in ["gradient","sgd"]:
                    time_temp = time.process_time() 
                    gradsj =  full_backward_propagation(Y_hat[j], Y[:,start:end], cache[j], params[j], nn_architecture)
                    profiling_time["full_backward_propagation"] += (time.process_time() - time_temp)    
                    grads.append(gradsj)
            
            time_temp = time.process_time() 
            if method == "gradient":            params, var, cputime = update_nn_weights_profiled(params, grads, N, learning_rate,  kernel_a, alpha, beta, gamma)
            elif method == "nogradient":        params, var, cputime = update_nn_weights_derivative_free_profiled(params, costs, learning_rate, N, kernel_a, alpha, beta, gamma)
            elif method == "sgd":               params, var = update_sgd(params[0], grads[0], nn_architecture, learning_rate)
            else: raise Exception("No method found")
            profiling_time["weights_update"] += (time.process_time() - time_temp)   

            if method in ["gradient","nogradient"]:
                profiling_time["weights_update_without_flattening"] += cputime 

            #end of iteration       
            cost_history.append(costs)

            #mean position
            mean_param = get_mean(params)
            Y_hat_mean, _ = full_forward_propagation(X[:,start:end], mean_param, nn_architecture)
            cost_mean = get_cost_value(Y_hat_mean, Y[:,start:end], cost_type)
            cost_history_mean.append(cost_mean)

        #end of epoch----------------
        var_mean = np.mean(var) #mean of variances along dimensions of parameter space

        if(verbose):
            print("Iteration: {:05} - mean cost: {:.5f} - particle variance: {:.5f}".format(i, np.mean(costs), var_mean))

        alpha = alpha + alpha_rate
        elapsed_epochs += 1

        if var_mean < var_epsilon: 
            print("Convergence achieved - Particles are localized")
            break


    if not method == "sgd": 
        plot_cost(cost_history,cost_history_mean,'Training Cost Function')
        plot_list(np.mean(cost_history,axis=1),'Mean Cost Function') 
        plot_distance_matrix(params,N) 
    else:
        plot_list(cost_history,'Training Cost Function')

    print("Cost Function evaluated {:01} times".format(int(n_batches*elapsed_epochs*N)))
    print("")
    print("CPU TIME --------------------------------------")
    for key,value in profiling_time.items():
        print(key, value)
    print("")

    return params, mean_param



@timerfunc
def train_experimental(X, Y, nn_architecture, epochs, learning_rate, method, n_batches, batch_size, cost_type, N, kernel_a, alpha_init, alpha_rate, beta, gamma, verbose, var_epsilon, dispersion_factor=6):
  
    from optimizers import update_cloud_derivative_free
    from utils import flatten_weights, unflatten_weights

    if method == "sgd": 
        N = 1

    # initiation of neural net parameters
    params = [init_layers(nn_architecture,i,dispersion_factor) for i in range(N)]
    alpha = alpha_init

    # initiation of lists storing the history 
    cost_history = []
    cost_history_mean = []
    accuracy_history = []
    
    elapsed_epochs = 0    

    #find optimal kernel_a
    if kernel_a == "auto":
        print("Finding kernel constant...")
        paramsf, _, _ = flatten_weights(params,N) 
        params_diff_matrix = paramsf[:,np.newaxis] - paramsf
        norm = np.sum(params_diff_matrix**2, axis=2) 
        for kernel_a in [0.0001, 0.0005,0.001,0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
            if(np.mean(np.einsum('ij -> i',np.exp(-kernel_a*norm))/N)) < 0.5:
                break

        print("Kernel constant found: " + str(kernel_a))

    if learning_rate == "auto":
        learning_rate = 1
        lr_decay = True
    else:
        lr_decay = False

    # performing calculations for subsequent iterations
    for i in range(epochs):
        
        for batch in range(n_batches):
   
            Y_hat = []
            costs = []
            cache = []
            grads = []

            start = batch*batch_size
            end = start + batch_size

            for j in range(N):  

                # step forward
                Y_hat_temp, cache_temp = full_forward_propagation(X[:,start:end], params[j], nn_architecture)
                Y_hat.append(Y_hat_temp)
                cache.append(cache_temp)
                
                # calculating cost and saving it to history
                costj = get_cost_value(Y_hat[j], Y[:,start:end], cost_type)
                costs.append(costj)  
            
            #get cloud (flattened weights), gradients, nn shape and weight names
            paramsf, nn_shape, weight_names = flatten_weights(params,N)


            #get updated cloud and its variance
            paramsf, var = update_cloud_derivative_free(paramsf, costs, learning_rate, N, kernel_a, alpha, beta, gamma)

            #restore NN weight shapes 
            params = unflatten_weights(paramsf,nn_shape,weight_names,N)

            if (lr_decay):
                if i == 0:
                    paramsf_previous = paramsf
                    gt = 0

                delta = paramsf_previous - paramsf
                gt = gt + np.absolute(delta)
                learning_rate = 1 / np.sqrt(1 + gt)

                paramsf_previous = paramsf
                #print(np.mean(learning_rate))   
            
            #end of iteration       
            cost_history.append(costs)

            #mean position
            mean_param = get_mean(params)
            Y_hat_mean, _ = full_forward_propagation(X[:,start:end], mean_param, nn_architecture)
            cost_mean = get_cost_value(Y_hat_mean, Y[:,start:end], cost_type)
            cost_history_mean.append(cost_mean)

        #end of epoch----------------
        var_mean = np.mean(var) #mean of variances along dimensions of parameter space

        if(verbose):
            print("Iteration: {:05} - mean cost: {:.5f} - particle variance: {:.5f}".format(i, np.mean(costs), var_mean))

        alpha = alpha + alpha_rate
        elapsed_epochs += 1

        if var_mean < var_epsilon: 
            print("Convergence achieved - Particles are localized")
            break


    if not method == "sgd": 
        plot_cost(cost_history,cost_history_mean,'Training Cost Function')
        plot_list(np.mean(cost_history,axis=1),'Mean Cost Function') 
        plot_distance_matrix(params,N) 
    else:
        plot_list(cost_history,'Training Cost Function')

    print("Cost Function evaluated {:01} times".format(int(n_batches*elapsed_epochs*N)))

    return params, mean_param






            #    if i == 0:
            #        paramsf_m2 = paramsf
            #        paramsf_m1 = paramsf 
            #            
            #    paramsf_m3 = paramsf_m2
            #    paramsf_m2 = paramsf_m1
            #    paramsf_m1 = paramsf

            #    if i > 3:
            #        delta_1 = paramsf_m1 - paramsf_m2
            #        delta_2 = paramsf_m2 - paramsf_m3

            #        learning_rate_update = np.full_like(learning_rate, 1)

            #        learning_rate_uptade[np.less(delta_1*delta_2,0.0)] = 0.5
            #        learning_rate_uptade[np.greater(delta_1*delta_2,0.0)] = 1.2

            #        learning_rate_update.clip(learning_rate_update,0.0001,2)

            #        learning_rate = learning_rate * learning_rate_update


#OLD VERSIONS OF UPDATE FUNCTIONS------------------------------------------------------------------------------
#Same as update_nn_weights but without using numpy broadcasting -> slower but easies to understand 
def update_nn_weights_old(cloud,gradients,N, lr, kernel_a, alpha, beta, gamma):
             
    #get flattened weights, gradients, nn shape and weight names
    cloudf, gradientsf, nn_shape, weight_names = flatten_weights_gradients(cloud, gradients, N)

    #compute kernels
    kernels =  [[kernel(cloudf[i],cloudf[j],kernel_a) for j in range(N)]   for i in range(N)]
    
    if beta > 0:
        gkernels = [[gkernel(cloudf[i],cloudf[j],kernel_a) for j in range(N)]  for i in range(N)]
    
    #compute mean and standart deviation
    cloud_mean = np.mean(cloudf, axis=0)
    cloud_var = get_var(cloudf,cloud_mean) 

    #compute gradient flows
    updates = []
    for nn in range(N):    
        R = 0
        P = 0
        S = 0
        
        Q = [kernels[nn][j]*gradientsf[j] for j in range(N)]
        Q = np.mean(Q,axis=0)
        
        if alpha > 0:
            R = [[kernels[nn][j]*(cloudf[j] - cloudf[k]) for j in range(N)] for k in range(N)]
            R = [item for sublist in R for item in sublist] #Flatten list of lists
            R = np.sum(R,axis=0)*float(1/N**2)
        
        if beta > 0:
            P = [gkernels[nn][j] for j in range(N)] 
            P = np.mean(P,axis=0) 
            
        if gamma > 0:
            S = [kernels[nn][j] * np.divide((cloudf[j]-cloud_mean),cloud_var) for j in range(N)]
            S = np.mean(S,axis=0)
    
        updates.append( -lr*(Q + alpha*R + beta*P + gamma*S)  )
    
    #update flattened tensors
    for nn in range(N):
        cloudf[nn] = cloudf[nn] + updates[nn]    

    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(cloudf,nn_shape,weight_names,N)
    
    return new_nn_weights, cloud_var


#Same as update_nn_weights_derivative_free but without using numpy broadcasting -> slower but easier to understand 
def update_nn_weights_derivative_free_old(cloud, cost, lr, N, kernel_a, alpha, beta, gamma):

    #get flattened weights, nn shape and weight names
    cloudf, nn_shape, weight_names = flatten_weights(cloud,N)  
  
    #compute kernels
    kernels =  [[kernel(cloudf[i],cloudf[j],kernel_a) for j in range(N)]   for i in range(N)]
    gkernels = [[gkernel(cloudf[i],cloudf[j],kernel_a) for j in range(N)]  for i in range(N)]
    
    #plt.imshow(kernels,vmin=0,vmax=1)
    #plt.colorbar()

    #compute mean and standart deviation
    cloud_mean = np.mean(cloudf, axis=0)
    cloud_var = get_var(cloudf,cloud_mean) 
        
    #compute gradient flows
    updates = []
    for nn in range(N):    
        R = 0
        P = 0
        S = 0
        
        Q = [gkernels[nn][j]*cost[j] + kernels[nn][j]*cost[j]*np.divide((cloudf[j]-cloud_mean),cloud_var) for j in range(N)]
        Q = np.mean(Q,axis=0)
        
        if alpha > 0:
            R = [[kernels[nn][j]*(cloudf[j] - cloudf[k]) for j in range(N)] for k in range(N)]
            R = [item for sublist in R for item in sublist] #Flatten list of lists
            R = np.sum(R,axis=0)*float(1/N**2)
        
        if beta > 0:
            P = [gkernels[nn][j] for j in range(N)] 
            P = np.mean(P,axis=0) 
            
        if gamma > 0:
            S = [kernels[nn][j] * np.divide((cloudf[j]-cloud_mean),cloud_var) for j in range(N)]
            S = np.mean(S,axis=0)
    
        updates.append( -lr*(Q + alpha*R + beta*P + gamma*S)  )
    
    #update flattened tensors
    for nn in range(N):
        cloudf[nn] = cloudf[nn] + updates[nn]    
  
    
    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(cloudf,nn_shape,weight_names,N)
    
    return new_nn_weights, cloud_var




#UPDATE FUNCTIONS WITH COMPUTED TIME AS OUTPUT----------------------------------------------------------------------------
def update_nn_weights_profiled(cloud, gradients, N, lr, kernel_a, alpha, beta, gamma):
    
    #get cloud (flattened weights), gradients, nn shape and weight names
    cloudf, gradientsf, nn_shape, weight_names = flatten_weights_gradients(cloud, gradients, N)
    
    #time needed for the update
    time_temp = time.process_time()

    #get updated cloud and its variance
    cloudf, cloud_var = update_cloud(cloudf, gradientsf, N, lr, kernel_a, alpha, beta, gamma) 
    update_time = time.process_time() - time_temp

    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(cloudf,nn_shape,weight_names,N)
    
    return new_nn_weights, cloud_var, update_time


def update_nn_weights_derivative_free_profiled(cloud, cost, lr, N, kernel_a, alpha, beta, gamma):
    
    #get cloud (flattened weights), gradients, nn shape and weight names
    cloudf, nn_shape, weight_names = flatten_weights(cloud,N)

    #time needed for the update
    time_temp = time.process_time()

    #get updated cloud and its variance
    cloudf, cloud_var = update_cloud_derivative_free(cloudf, cost, lr, kernel_a, alpha, beta, gamma)
    update_time = time.process_time() - time_temp

    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(cloudf,nn_shape,weight_names,N)
    
    return new_nn_weights, cloud_var, update_time


def norm2(params_i,params_j): 
    #l2 norm squared
    return np.sum(np.subtract(params_i,params_j)**2)

def kernel(params_i,params_j,const): 
    n = norm2(params_i,params_j)       
    return np.exp(-const*n)

def gkernel(params_i,params_j,const):
    n = norm2(params_i,params_j)       
    return -2.0*const*(params_i-params_j)*np.exp(-const*n)