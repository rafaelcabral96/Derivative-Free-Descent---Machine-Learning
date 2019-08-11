import numpy as np
import time 
import pandas as pd
import matplotlib.pyplot as plt

#from scipy import stats


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
        msg = "\nThe clock time (CPU time) for {func} was {time:.5f} ({cpu_time:.5f}) seconds"
        print(msg.format(func=func.__name__,
                         time=runtime,
                         cpu_time = runtime_cpu))
        return value
    return function_timer


def flatten_weights(cloud,N):
    tensor_names = list(cloud[0].keys())

    nn_shape = []
    for param in cloud[0]:
        nn_shape.append(np.shape(cloud[0][param]))

    cloudf = []
    for nn in range(N):
        flatten = np.array([])
        for param in cloud[nn].values():
            flatten = np.concatenate((flatten,np.ndarray.flatten(param)),axis=None)
        cloudf.append(flatten)   
    cloudf = np.array(cloudf)

    return cloudf, nn_shape, tensor_names

def flatten_weights_gradients(cloud, gradients, N):
    n_params = len(cloud[0])
    weight_names = list(cloud[0].keys())
    nn_shape = []
    for param in cloud[0]:
        nn_shape.append(np.shape(cloud[0][param]))
            
    cloudf = []
    gradientsf = []
    for nn in range(N):
        flatten = np.array([])
        flatten_g = np.array([])
        for param in range(n_params):
            flatten_temp = np.ndarray.flatten(cloud[nn][weight_names[param]])
            flatten = np.concatenate((flatten,flatten_temp),axis=None)
            flatten_g_temp =  np.ndarray.flatten(gradients[nn]["d" + weight_names[param]])
            flatten_g = np.concatenate((flatten_g,flatten_g_temp),axis=None)
        cloudf.append(flatten)    
        gradientsf.append(flatten_g)
    cloudf = np.array(cloudf)
    gradientsf = np.array(gradientsf)

    return cloudf, gradientsf, nn_shape, weight_names

def unflatten_weights(cloudf,shapes,weight_names,N):
    n_params = len(shapes)
    new_nn_weights = []
    for nn in range(N):
        init = 0
        new_params_nn = {}
        for param in range(n_params):
            num_params = int(np.prod(shapes[param]))
            new_params_nn[weight_names[param]] = np.reshape(cloudf[nn][init:(init + num_params)],shapes[param])
            init += num_params
        new_nn_weights.append(new_params_nn)

    return new_nn_weights

def kernel_a_finder(cloudf, N):

    print("Finding kernel constant...")

    cloud_diff_matrix = cloudf[:,np.newaxis] - cloudf
    norm = np.sum(cloud_diff_matrix**2, axis=2) 
    kernel_a_pool = [0.00001, 0.0005, 0.0001, 0.0005,0.001,0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100]
    for kernel_a in kernel_a_pool:
        if(np.mean(np.einsum('ij -> i',np.exp(-kernel_a*norm))/N)) < 0.5:
            break

    print("Kernel constant found: " + str(kernel_a))
    return kernel_a

def gradient(func, x, h = None):
    if h is None:
        # Note the hard coded value found here is the square root of the
        # floating point precision, which can be found from the function
        # call np.sqrt(np.finfo(float).eps).
        h = 1.49011611938477e-08
    xph = x + h
    dx = xph - x
    return (func(xph) - func(x)) / dx

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_mean(cloud):
    params_mean = {}
    for key in cloud[0]:
        params_mean[key] = np.mean([cloud[j][key] for j in range(len(cloud))],axis=0)
    return params_mean

def get_var(cloud):
     return np.var(cloud, axis = 0)
     #return np.mean([ np.linalg.norm(param-params_mean)**2 for param in cloud])

#def normal_test(cloudf):
#    k, _ = stats.normaltest( (cloudf- np.mean(cloudf,axis=0))/np.var(cloudf,axis=0) , axis = 0)
#    print(np.mean(k))
#    print("Normality test p-value - percentage of particles rejected: "  + str(1 - np.mean(np.greater(p,0.05))))


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
    plt.legend([Line2D([0], [0], color = 'black', lw=4)],['Cloud mean'])

def plot_list(data, title = 'Mean cost function'):
    
    plt.figure()
    plt.plot(data)
    plt.xlabel('Iteration')
    plt.ylabel(title)


def plot_distance_matrix(cloud, N):
    
    n_params = len(cloud[0])
    tensor_names = list(cloud[0].keys())

    #flatten all tensors
    cloudf = []
    for nn in range(N):
        flatten = np.ndarray.flatten(cloud[nn][tensor_names[0]])
        for param in range(n_params):
            flatten_temp = np.ndarray.flatten(cloud[nn][tensor_names[param]])
            flatten = np.concatenate((flatten,flatten_temp),axis=None)
        cloudf.append(flatten)    
        
    plot_matrix = [[norm2(cloudf[i],cloudf[j]) for j in range(N)] for i in range(N)] 
    plt.figure()
    plt.imshow(np.asarray(plot_matrix))
    plt.title("Distance between NN")
    plt.colorbar()

