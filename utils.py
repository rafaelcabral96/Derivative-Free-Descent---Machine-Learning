import numpy as np
import time 
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


def load_mnist(one_hot=True):
    from tensorflow.examples.tutorials.mnist import input_data  
    mnist = input_data.read_data_sets('Datasets/MNIST_data/', one_hot=one_hot)
    return mnist.train.images, mnist.train.labels

def timerfunc(func):
    #timer decorator
    def function_timer(*args, **kwargs):
        start = time.clock()
        start_cputime = time.process_time()

        value = func(*args, **kwargs)

        end_cputime = time.process_time()
        end = time.clock() 
        runtime_cpu = end_cputime - start_cputime
        runtime = end - start  
        msg = "The clock time (CPU time) for {func} was {time:.5f} ({cpu_time:.5f}) seconds"
        print(msg.format(func=func.__name__,
                         time=runtime,
                         cpu_time = runtime_cpu))
        return value
    return function_timer


def flatten_weights(params,N_nn):
    tensor_names = list(params[0].keys())

    nn_shape = []
    for param in params[0]:
        nn_shape.append(np.shape(params[0][param]))

    paramsf = []
    for nn in range(N_nn):
        flatten = np.array([])
        for param in params[nn].values():
            flatten = np.concatenate((flatten,np.ndarray.flatten(param)),axis=None)
        paramsf.append(flatten)   
    paramsf = np.array(paramsf)

    return paramsf, nn_shape, tensor_names

def flatten_weights_gradients(params, gradients, N_nn):
    n_params = len(params[0])

    weight_names = list(params[0].keys())
    nn_shape = []
    for param in params[0]:
        nn_shape.append(np.shape(params[0][param]))
            
    paramsf = []
    gradientsf = []
    for nn in range(N_nn):
        flatten = np.array([])
        flatten_g = np.array([])
        for param in range(n_params):
            flatten_temp = np.ndarray.flatten(params[nn][weight_names[param]])
            flatten = np.concatenate((flatten,flatten_temp),axis=None)
            flatten_g_temp =  np.ndarray.flatten(gradients[nn]["d" + weight_names[param]])
            flatten_g = np.concatenate((flatten_g,flatten_g_temp),axis=None)
        paramsf.append(flatten)    
        gradientsf.append(flatten_g)
    paramsf = np.array(paramsf)
    gradientsf = np.array(gradientsf)

    return paramsf, gradientsf, nn_shape, weight_names

def unflatten_weights(paramsf,shapes,weight_names,N_nn):
    n_params = len(shapes)
    new_nn_weights = []
    for nn in range(N_nn):
        init = 0
        new_params_nn = {}
        for param in range(n_params):
            num_params = int(np.prod(shapes[param]))
            new_params_nn[weight_names[param]] = np.reshape(paramsf[nn][init:(init + num_params)],shapes[param])
            init += num_params
        new_nn_weights.append(new_params_nn)

    return new_nn_weights

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def norm2(params_i,params_j): 
    #l2 norm squared
    return np.sum(np.subtract(params_i,params_j)**2)

def kernel(params_i,params_j,const): 
    n = norm2(params_i,params_j)       
    return np.exp(-const*n)

def gkernel(params_i,params_j,const):
    n = norm2(params_i,params_j)       
    return -2.0*const*(params_i-params_j)*np.exp(-const*n)

def get_mean(params):
    params_mean = {}
    for key in params[0]:
        params_mean[key] = np.mean([params[j][key] for j in range(len(params))],axis=0)
    return params_mean

def get_var(params,params_mean):
     return np.var(params, axis = 0)
     #return np.mean([ np.linalg.norm(param-params_mean)**2 for param in params])

def normal_test(paramsf):
    k, _ = stats.normaltest( (paramsf- np.mean(paramsf,axis=0))/np.var(paramsf,axis=0) , axis = 0)
    print(np.mean(k))
    #print("Normality test p-value - percentage of particles rejected: "  + str(1 - np.mean(np.greater(p,0.05))))

#PLOTS-----------------------------------------------

def plot_cost(train_cost,cost_mean,legend= True,title = 'Training Cost Function'):
    
    from matplotlib.lines import Line2D

    headers = ["NN" + str(i) for i in range(len(train_cost[0]))]    
    df = pd.DataFrame(train_cost, columns=headers)
    if cost_mean != 0: df['mean'] = pd.Series(cost_mean, index=df.index)

    styles = ['-']*len(train_cost[0])
    if cost_mean != 0: styles.append('k-')

    df=df.astype(float)

    plt.figure()
    df.plot(style = styles,legend = legend)
    plt.xlabel('Iteration')
    plt.ylabel(title)
    plt.legend([Line2D([0], [0], color = 'black', lw=4)],['Mean Tensor'])

def plot_list(data, title = 'Mean cost function'):
    
    plt.figure()
    plt.plot(data)
    plt.xlabel('Iteration')
    plt.ylabel(title)


def plot_distance_matrix(params, N_nn):
    
    n_params = len(params[0])
    tensor_names = list(params[0].keys())

    #flatten all tensors
    paramsf = []
    for nn in range(N_nn):
        flatten = np.ndarray.flatten(params[nn][tensor_names[0]])
        for param in range(n_params):
            flatten_temp = np.ndarray.flatten(params[nn][tensor_names[param]])
            flatten = np.concatenate((flatten,flatten_temp),axis=None)
        paramsf.append(flatten)    
        
    plot_matrix = [[norm2(paramsf[i],paramsf[j]) for j in range(N_nn)] for i in range(N_nn)] 
    plt.figure()
    plt.imshow(np.asarray(plot_matrix))
    plt.title("Distance between NN")
    plt.colorbar()

