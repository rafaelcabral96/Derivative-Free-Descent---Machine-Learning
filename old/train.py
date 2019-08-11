import numpy as np
from neural_networks import init_layers, full_forward_propagation, full_backward_propagation, get_cost_value, get_accuracy_value
from utils import timerfunc, get_mean, plot_cost, plot_list, plot_distance_matrix, kernel_a_finder
from optimizers import update_gd, update_nn_weights, update_nn_weights_derivative_free 


@timerfunc
def train_nn(X, Y, cloud, nn_architecture, method, max_epochs, n_batches,  batch_size, learning_rate, 
          cost_type, N, kernel_a, alpha_init, alpha_rate, beta, gamma, verbose, var_epsilon):


    # initiation of lists storing the cost history 
    cost_history = []
    cost_history_mean = []
    
    alpha = alpha_init
    elapsed_epochs = 0   
    print("\nTraining started...")   

    # performing calculations for subsequent iterations
    for i in range(max_epochs):
        
        for batch in range(n_batches):
   
            start = batch*batch_size
            end = start + batch_size

            Y_hat = []
            costs = []
            cache = []
            grads = []

            for j in range(N):  

                # step forward
                Y_hat_temp, cache_temp = full_forward_propagation(X[:,start:end], cloud[j], nn_architecture)
                Y_hat.append(Y_hat_temp)
                cache.append(cache_temp)
                
                # calculating cost and saving it to history
                costj = get_cost_value(Y_hat[j], Y[:,start:end], cost_type)
                costs.append(costj)                
                # step backward - calculating gradient
                if method in ["gradient_descent", "swarm"]:
                    gradsj =  full_backward_propagation(Y_hat[j], Y[:,start:end], cache[j], cloud[j], nn_architecture)
                    grads.append(gradsj)  

            
            if method == "swarm":              cloud, cloud_var = update_nn_weights(cloud, grads, learning_rate, N, kernel_a, alpha, beta, gamma)
            elif method == "swarm_derivfree":  cloud, cloud_var = update_nn_weights_derivative_free(cloud, costs, learning_rate, N, kernel_a, alpha, beta, gamma)
            elif method == "gradient_descent": cloud, cloud_var = update_gd(cloud[0], grads[0], nn_architecture, learning_rate)
            else: raise Exception("No method found")

            #end of iteration       
            cost_history.append(costs)

            #mean particle position and its cost
            cloud_mean = get_mean(cloud)
            Y_hat_mean, _ = full_forward_propagation(X[:,start:end], cloud_mean, nn_architecture)
            cost_mean = get_cost_value(Y_hat_mean, Y[:,start:end], cost_type)
            cost_history_mean.append(cost_mean)

        #end of epoch----------------
        cloud_var = np.mean(cloud_var) #mean of variances along dimensions of parameter space

        if(verbose):
            print("Iteration: {:05} - Cloud mean cost: {:.5f} - Cloud variance: {:.5f}".format(i, cost_mean, cloud_var))

        alpha += alpha_rate
        elapsed_epochs += 1

        if cloud_var < var_epsilon: 
            print("Convergence achieved - particles are localized")
            break

    if i == (max_epochs - 1): print("Maximum amount of epochs reached")

    print("\nFunction value at cloud mean: " + str(cost_mean))
    print("Cost function evaluated {:01} times".format(int(n_batches*elapsed_epochs*N)))

    return cloud, cloud_mean, cloud_var, cost_history, cost_history_mean



@timerfunc
def train_fmin(func, cloud, max_iterations, var_epsilon, learning_rate, method, N, kernel_a, alpha_init, alpha_rate, beta, gamma, verbose):

    from optimizers import update_cloud, update_cloud_derivative_free, update_gd_func
    from utils import gradient
  
    alpha = alpha_init

    # initiation of lists storing the history 
    cost_history = []
    cost_history_mean = []
    elapsed_iterations = 0    
       
    # performing calculations for subsequent iterations
    for i in range(max_iterations):
        
        function_values = func(cloud.T)

        if method in ["gradient_descent", "swarm"]:
            gradients = gradient(func,cloud.T).T
        
        if method == "swarm":                 cloud, cloud_var = update_cloud(cloud, gradients, N, learning_rate, kernel_a, alpha, beta, gamma)
        elif method == "swarm_derivfree":     cloud, cloud_var = update_cloud_derivative_free(cloud, function_values, N, learning_rate, kernel_a, alpha, beta, gamma)
        elif method == "gradient_descent":    cloud, cloud_var = update_gd_func(cloud, gradients, learning_rate)
        else: raise Exception("No method found")

        #end of iteration       
        cost_history.append(function_values)

        #mean position
        cloud_mean = np.mean(cloud, axis=0)
        cloud_mean_func = func(cloud_mean)
        cost_history_mean.append(cloud_mean_func)

        #end of epoch----------------
        cloud_var = np.mean(cloud_var) #mean of variances along dimensions of parameter space

        if(verbose):
            print("Iteration: {:05} - Cloud mean cost: {:.5f} - Cloud variance: {:.5f}".format(i, cloud_mean_func, cloud_var))

        alpha = alpha + alpha_rate
        elapsed_iterations += 1

        if cloud_var < var_epsilon: 
            print("Convergence achieved - Particles are localized")
            break

    print("\nFunction value at cloud mean: " + str(cloud_mean_func))
    print("Function evaluated {:01} times".format(int(elapsed_iterations*N)))

    return cloud, cloud_mean, cloud_var, cost_history, cost_history_mean
