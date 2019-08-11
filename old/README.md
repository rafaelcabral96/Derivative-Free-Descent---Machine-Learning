## Description
**mlswarm** contains two operable classes:
1. *neuralnet* - train neural networks
2. *function* - minimize functions

For each, there are three available algorithms:
1. *swarm* - swarm-like optimization algorithm
2. *swarm_derivfree* - similar to the former but derivative free, by using gaussian clouds
3. *gradient_descent* - gradient descent optimization

For a *neuralnet* object there are three main methods (see examples):
1. nn = neuralnet(...) - define neural network architecture and create neural network
2. nn = init_cloud(N) - Initialize cloud with N particles
3. nn.train(...) - Define the training data, algorithm parameters and start the algorithm

For a *function* object there are three main methods (see examples):
1. func = function(lambda x: ...) - create a function object 
2. func = init_cloud(...) - Define array of initial particle positions
3. func.minimize(...) - Define the algorithms parameters and start the algorithm


## Advantages
1. It is possible to use linear and non-differentiable activation functions because there is no backpropagation -  [reference](https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/)

2. (Multivariate) linear regression, which uses linear activation function, can be done (see Example 1.)

3. Steps function can be used for binary classification (see Example 5.). Steps activation functions are probably the simplest and fasted to compute.

4. Since there is no backpropagation, we do not have to worry with the gradients collapsing or exploding, therefore there is more freedom when defining the initial values of the weights. I should try other initialization schemes.

5. Usually the mean of the cloud has a higher validation metric on the test set then the individual particles. I think that by training different particles and using the mean of those particles, we are diminishing the problem of overfitting, hence the better results.

6. So far, better than other derivative-free methods when training Neural Networks -> Generic Algorithms, Simulated Annealing, hill climb, random hill climb available the package Python *mlrose*



## Notes

1. Using 

```
np.var(paramsf, axis=0) 
```

instead of 

```
np.mean([ np.linalg.norm(param-params_mean)**2 for param in params]) 
```

to compute the cloud variance leads to significantly better results in Neural Networks

2. Using kernel = kernel(-norm/(2\*var)) leads to unstable results in Neural Networks 


## A few ideas

1. Check the shape of the cloud over iterations - for instance calculate the p-value for normality test - From what I have seen the test is usually positive across iterations and particles

2. Training each layer of the network separatly. In this view a particle will contain several subparticles representing the different network layers

3. Force the clouds to have a certain variance in order to avoid being stuck in local minimuns

4. L2 norm between particles increases with the dimension of parameter space. This requires choosing different kernel_a for different training sessions. Maybe use an heuristic to choose kernel_a? 

sum_i(kernel(i,j)) belongs in [1/N,N] ranging from no iteraction between particles to maximum interaction between particles.

sum_i(kernel(i,j)) / N can be seen as a measure for the intensity between particles

Maybe choose kernel_a such that  sum_i(kernel(i,j))/N ~ 0.5 i.e the interection intensity is 50% 

I have done this if you choose kernel_a="auto"

