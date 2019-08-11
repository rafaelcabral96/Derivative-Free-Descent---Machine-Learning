import numpy as np
from neural_networks_experimental import init_layers, full_forward_propagation, get_cost_value, get_accuracy_value
from utils import timerfunc, get_mean, plot_cost, plot_list, plot_distance_matrix
from optimizers import update_cloud_derivative_free

@timerfunc
def train(X, Y, nn_architecture, epochs, learning_rate, method, n_batches, batch_size, cost_type, N_nn, kernel_a, alpha_init, alpha_rate, beta, gamma, verbose, var_epsilon, dispersion_factor=6):
  
    if method == "sgd": 
        N_nn = 1

    # initiation of neural net parameters
    W,b = init_layers(nn_architecture,N_nn,42,dispersion_factor)

    Wshape = [np.shape(W_layer) for W_layer in W]
    bshape = [np.shape(b_layer) for b_layer in b]
    Wsize = np.prod(Wshape)

    # initiation of lists storing the history 
    cost_history = []
    cost_history_mean = []
    accuracy_history = []
    
    elapsed_epochs = 0    
    
    alpha = alpha_init

    # performing calculations for subsequent iterations
    for epoch in range(epochs):
        
        for batch in range(n_batches):

            start = batch*batch_size
            end = start + batch_size

            # step forward
            Y_hat = full_forward_propagation(X[:,start:end], W, b, nn_architecture)

            # calculating cost and saving it to history
            costs = np.squeeze(np.mean((Y_hat - Y)**2,axis=2))
            cost_history.append(costs)


            #flatten weights
            W = [W_layer.tolist() for W_layer in W]
            W = list(map(list, zip(*W)))
            b = [b_layer.tolist() for b_layer in b]
            b = list(map(list, zip(*b)))

            paramsf = []
            for i in range(N_nn):
                particle = np.array([])
                for j in range(len(W[0])):
                    particle = np.concatenate((particle,np.array(W[i][j]).flatten()),axis=None)
                    particle = np.concatenate((particle,np.array(b[i][j]).flatten()),axis=None) 
                paramsf.append(particle)
            paramsf = np.array(paramsf)


            #get updated cloud and its variance
            paramsf, params_var = update_cloud_derivative_free(paramsf, costs, learning_rate, kernel_a, alpha, beta, gamma)

            #restore NN weight shapes 
            W = []
            b = []
            init = 0 
            for param in range(len(Wshape)):
                pos = np.prod(Wshape[param][1:])
                Wj = np.reshape(paramsf[:,init:(init+pos)],Wshape[param])
                init = init + pos
                pos = np.prod(bshape[param][1:])
                bj = np.reshape(paramsf[:,init:(init+pos)],bshape[param])
                W.append(Wj)
                b.append(bj)

        #end of epoch----------------
        var_mean = np.mean(params_var) #mean of variances along dimensions of parameter space

        if(verbose):
            print("Iteration: {:05} - mean cost: {:.5f} - particle variance: {:.5f}".format(epoch, np.mean(costs), var_mean))

        alpha = alpha + alpha_rate
        elapsed_epochs += 1

        if var_mean < var_epsilon: 
            print("Convergence achieved - Particles are localized")
            break


    if not method == "sgd": 
        plot_cost(cost_history,0,'Training Cost Function')
        plot_list(np.mean(cost_history,axis=1),'Mean Cost Function') 
    else:
        plot_list(cost_history,'Training Cost Function')

    print("Cost Function evaluated {:01} times".format(int(n_batches*elapsed_epochs*N_nn)))

