import numpy as np
from neural_networks import init_layers, full_forward_propagation, full_backward_propagation, get_cost_value, get_accuracy_value
from utils import timerfunc, get_mean, plot_cost, plot_list, plot_distance_matrix
from optimizers import update_sgd, update_gradients, update_nogradients, update_gradients_old, update_nogradients_old

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
                if method in ["gradient","sgd"]:
                    gradsj =  full_backward_propagation(Y_hat[j], Y[:,start:end], cache[j], params[j], nn_architecture)
                    grads.append(gradsj)
            
            if method == "gradient":            params, var = update_gradients(params, grads, N_nn, learning_rate,  kernel_a, alpha, beta, gamma)
            elif method == "gradient_old":        params, var = update_gradients_old(params, grads, N_nn, learning_rate,  kernel_a, alpha, beta, gamma) 
            elif method == "nogradient":        params, var = update_nogradients(params, costs, learning_rate, N_nn, kernel_a, alpha, beta, gamma)
            elif method == "nogradient_old":    params, var = update_nogradients_old(params, costs, learning_rate, N_nn, kernel_a, alpha, beta, gamma) 
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