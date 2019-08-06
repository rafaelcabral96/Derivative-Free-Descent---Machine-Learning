import numpy as np
from neural_networks import init_layers, full_forward_propagation, full_backward_propagation, get_cost_value, get_accuracy_value
from utils import timerfunc, get_mean, plot_cost, plot_list, plot_distance_matrix
from optimizers import update_sgd, update_nn_weights, update_nn_weights_old, update_nn_weights_derivative_free, update_nn_weights_derivative_free_old      

@timerfunc
def train(X, Y, nn_architecture, epochs, learning_rate, method, n_batches, batch_size, cost_type, N_nn, kernel_a, alpha_init, alpha_rate, beta, gamma, verbose, var_epsilon, dispersion_factor=6):
  
    if method == "sgd": 
        N_nn = 1

    # initiation of neural net parameters
    params = [init_layers(nn_architecture,i,dispersion_factor) for i in range(N_nn)]
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

            for j in range(N_nn):  

                # step forward
                Y_hat_temp, cache_temp = full_forward_propagation(X[:,start:end], params[j], nn_architecture)
                Y_hat.append(Y_hat_temp)
                cache.append(cache_temp)
                
                # calculating cost and saving it to history
                costj = get_cost_value(Y_hat[j], Y[:,start:end], cost_type)
                costs.append(costj)  
                
                #accuracy = get_accuracy_value(Y_hat, Y)
                #accuracy_history.append(accuracy)

                # step backward - calculating gradient
                if method in ["gradient","gradient_old","sgd"]:
                    gradsj =  full_backward_propagation(Y_hat[j], Y[:,start:end], cache[j], params[j], nn_architecture)
                    grads.append(gradsj)
            
            if method == "gradient":            params, var = update_nn_weights(params, grads, N_nn, learning_rate,  kernel_a, alpha, beta, gamma)
            elif method == "gradient_old":      params, var = update_nn_weights_old(params, grads, N_nn, learning_rate,  kernel_a, alpha, beta, gamma) 
            elif method == "nogradient":        params, var = update_nn_weights_derivative_free(params, costs, learning_rate, N_nn, kernel_a, alpha, beta, gamma)
            elif method == "nogradient_old":    params, var = update_nn_weights_derivative_free_old(params, costs, learning_rate, N_nn, kernel_a, alpha, beta, gamma) 
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

        alpha = alpha + alpha_rate
        elapsed_epochs += 1

        if var_mean < var_epsilon: 
            print("Convergence achieved - Particles are localized")
            break


    if not method == "sgd": 
        plot_cost(cost_history,cost_history_mean,'Training Cost Function')
        plot_list(np.mean(cost_history,axis=1),'Mean Cost Function') 
        plot_distance_matrix(params,N_nn) 
    else:
        plot_list(cost_history,'Training Cost Function')

    print("Cost Function evaluated {:01} times".format(int(n_batches*elapsed_epochs*N_nn)))

    return params, mean_param



@timerfunc
def train_with_profiling(X, Y, nn_architecture, epochs, learning_rate, method, n_batches, batch_size, cost_type, N_nn, kernel_a, alpha_init, alpha_rate, beta, gamma, verbose, var_epsilon, dispersion_factor=6):
    
    import time 
    from optimizers import update_nn_weights_profiled, update_nn_weights_derivative_free_profiled   

    if method == "sgd": 
        N_nn = 1

    profiling_time = {
        "full_forward_propagation": 0.0,
        "get_cost_value": 0.0,
        "full_backward_propagation": 0.0,
        "weights_update": 0.0,
        "weights_update_without_flattening": 0.0
        }

    # initiation of neural net parameters
    params = [init_layers(nn_architecture,i,dispersion_factor) for i in range(N_nn)]
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

            for j in range(N_nn):  

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
            if method == "gradient":            params, var, cputime = update_nn_weights_profiled(params, grads, N_nn, learning_rate,  kernel_a, alpha, beta, gamma)
            elif method == "nogradient":        params, var, cputime = update_nn_weights_derivative_free_profiled(params, costs, learning_rate, N_nn, kernel_a, alpha, beta, gamma)
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
        plot_distance_matrix(params,N_nn) 
    else:
        plot_list(cost_history,'Training Cost Function')

    print("Cost Function evaluated {:01} times".format(int(n_batches*elapsed_epochs*N_nn)))
    print("")
    print("CPU TIME --------------------------------------")
    for key,value in profiling_time.items():
        print(key, value)
    print("")

    return params, mean_param


#IN DEVELOPMENT
@timerfunc
def find_func_minimum(func, cloud, max_iterations, learning_rate, method, cost_type, N_nn, kernel_a, alpha_init, alpha_rate, beta, gamma, verbose, var_epsilon):

    from optimizers import update_cloud, update_cloud_derivative_free
  
    if method == "sgd": 
        N_nn = 1

    alpha = alpha_init

    # initiation of lists storing the history 
    cost_history = []
    cost_history_mean = []
    elapsed_iterations = 0    
       
    # performing calculations for subsequent iterations
    for i in range(max_iterations):
        
        function_values = func(cloud)

        #if method in ["sgd", "gradient", "gradient_old"]:
        #    function_gradients = gradient(func,cloud)
    
        if method == "gradient":            cloud, var = update_cloud(cloud, function_gradients, N_nn, learning_rate,  kernel_a, alpha, beta, gamma)
        elif method == "nogradient":        cloud, var = update_cloud_derivative_free(cloud, function_values, learning_rate, N_nn, kernel_a, alpha, beta, gamma)
        elif method == "sgd":               cloud, var = update_gd_func(params[0], grads[0], nn_architecture, learning_rate)
        else: raise Exception("No method found")

        #end of iteration       
        cost_history.append(function_values)

        #mean position
        mean_param = np.mean(cloud,axis=0)
        cost_mean = func(mean_param)
        cost_history_mean.append(cost_mean)

        #end of epoch----------------
        var_mean = np.mean(var) #mean of variances along dimensions of parameter space

        if(verbose):
            print("Iteration: {:05} - mean cost: {:.5f} - particle variance: {:.5f}".format(i, np.mean(costs), var_mean))

        alpha = alpha + alpha_rate
        elapsed_iterations += 1

        if var_mean < var_epsilon: 
            print("Convergence achieved - Particles are localized")
            break


    if not method == "sgd": 
        plot_cost(cost_history,cost_history_mean,'Training Cost Function')
        plot_list(np.mean(cost_history,axis=1),'Mean Cost Function') 
        plot_distance_matrix(params,N_nn) 
    else:
        plot_list(cost_history,'Training Cost Function')

    print("Cost Function evaluated {:01} times".format(int(elapsed_iterations*N_nn)))

    return params, mean_param
