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

3. L2 norm between particles increases with the dimension of parameter space. Will this lead to problems when computing the variance for larger NN?

4. Check the shape of the cloud over iterations - for instance calculate the p-value for normality test 


## Advantages

1. It is possible to use linear activation functions because there is no backpropagation -  [reference](https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/)

2. Model architecture and activation function restrictions due to backpropagation are not present in derivative-free methods

3. So far, better than other derivative-free methods when training Neural Networks -> Generic Algorithms, Simulated Annealing, 