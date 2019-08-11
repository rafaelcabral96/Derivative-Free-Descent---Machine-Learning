from neural_networks import init_layers, full_forward_propagation, get_accuracy_value
from utils import get_mean, get_var, plot_cost, plot_list, kernel_a_finder, flatten_weights
from train import train_nn, train_fmin

import numpy as np

class _swarm:

    def __init__(self):
        self.N = None
        self.cloud = None
        self.cloud_mean = None
        self.cloud_var = None
        self.cost_history = None
        self.cost_history_mean = None
        self.train_method = None

    def get_cloud(self):
        return self.cloud

    def get_cloud_mean(self):
        return self.cloud_mean

    def get_cloud_var(self):
        return self.cloud_var
        
    def get_cost_history(self):
        return self.cost_history

    def get_cost_history_mean(self):
        return self.cost_history_mean

    def get_num_variables(self):
        return self.num_variables

    def plot_cost_history(self):
        if not self.train_method == "sgd": 
            plot_cost(self.cost_history, self.cost_history_mean, 'Training Cost Function')
            plot_list(np.mean(self.cost_history,axis=1), 'Mean Cost Function') 
        else:
            plot_list(cost_history, 'Training Cost Function')

    def clear():
        self.N = None
        self.cloud = None
        self.cloud_mean = None
        self.cost_history = None
        self.cost_history_mean = None
        self.train_method = None



class neuralnet(_swarm):

    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.num_variables = sum( [layer["input_dim"] * layer["output_dim"] + layer["output_dim"] for layer in architecture])
        self.cloudf = None

    def init_cloud(self, N, dispersion_factor=6):
        self.N = N
        self.cloud = [init_layers(self.architecture,i,dispersion_factor) for i in range(N)]
        self.cloud_mean = get_mean(self.cloud)
        cloudf, _, _ = flatten_weights(self.cloud, self.N) 
        self.cloud_var  = get_var(cloudf)

    def train(self, X, Y, method, max_epochs, n_batches, batch_size,
              learning_rate, cost_type, kernel_a = "auto",  alpha_init = 0,
               alpha_rate = 1, beta = 0, gamma = 1, verbose = True, var_epsilon = 0):

        if (method == "sgd") and (self.N > 1): 
            self.init_cloud(1)

        #transposing input and output data
        X = X.T
        Y = Y.T

        #find optimal kernel_a
        if kernel_a == "auto":
            cloudf, _, _ = flatten_weights(self.cloud, self.N) 
            kernel_a = kernel_a_finder(cloudf, self.N)

        self.train_method = method
        self.cloud, self.cloud_mean, self.cloud_var, self.cost_history, self.cost_history_mean =  train_nn(X, Y, self.cloud, self.architecture, method, max_epochs, n_batches,  batch_size, learning_rate, cost_type, self.N, kernel_a, alpha_init, alpha_rate, beta, gamma, verbose, var_epsilon)

    def forward_propagation(self, X, cloud):
        Y, _ = full_forward_propagation(X.T, cloud, self.architecture)
        return Y.T

    def prediction_accuracy_particle(self, X_test, Y_test, acc_type, particle=0):
        Y_test_hat = self.forward_propagation(X_test, self.cloud[particle])
        acc_test = get_accuracy_value(Y_test_hat, Y_test, acc_type)
        print("Test set accuracy using particle {:}: {:.5f}".format(particle, acc_test))

    def prediction_accuracy_mean_particle(self, X_test, Y_test, acc_type):
        Y_test_hat = self.forward_propagation(X_test, self.cloud_mean)
        acc_test = get_accuracy_value(Y_test_hat, Y_test, acc_type)
        print("Test set accuracy using cloud mean: {:.5f}".format(acc_test))

    def set_cloud(self, cloud):
        self.cloud = cloud
        self.cloud_mean = get_mean(self.cloud)
        self.cloud_var  = get_var(self.cloud)

    def get_num_variables(self):
        return self.num_variables



class function(_swarm):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def init_cloud(self, cloud):
        self.cloud = cloud.T
        self.N = len(self.cloud)
        self.cloud_mean = np.mean(self.cloud, axis = 0)
        self.cloud_var = np.mean(get_var(self.cloud))

    def minimize(self, max_iterations, var_epsilon, learning_rate, 
         method = "swarm_derivfree", kernel_a = "auto", alpha_init = 0, 
         alpha_rate = 1, beta = 0, gamma = 1, verbose= False):

        #number of particles
        if method == "sgd": 
            self.N = 1

        #find optimal kernel_a
        if kernel_a == "auto":
            kernel_a = kernel_a_finder(self.cloud, self.N)

        self.train_method = method
        self.cloud, self.cloud_mean, self.cloud_var, self.cost_history, self.cost_history_mean = train_fmin(self.func, self.cloud, max_iterations, var_epsilon, learning_rate, method, self.N, kernel_a, alpha_init, alpha_rate, beta, gamma, verbose)

    def evaluate(self,cloud):
        return self.func(cloud.T).T

