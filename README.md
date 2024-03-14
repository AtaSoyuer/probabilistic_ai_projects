# Probabilistic AI (2021 Fall) Course projects at ETH Zurich

## Project 1
Gaussian Process regression to model air pollution and predict fine particle concentration at new coordinate locations. Worked on model selection methods by evidence maximization, tackling the GP inference complexity O(n<sup>3</sup>) and asymmetric cost inherent in the metric. 

Two different implementations that passed the baseline are proposed with `scikit-learn` and `gpytorch` respectively.

**Method #1** (`scikit-learn`): GP regressor with optimal kernel selection via custom k-fold CV, K-means clustering to reduce sample size, custom prediction adjustment to adapt to asymmetric costs.  
**Method #2** (`gpytorch`): GP regressor with structured kernel interpolation, custom prediction adjustment to adapt to asymmetric costs.

*Baseline evaluation:* asymmetric MSE cost on the test set.

## Project 2
Bayesian neural network implementation based on the ’Bayes by Backprop’ algorithm by Blundell et al. (2015) [[1]](https://proceedings.mlr.press/v37/blundell15.html) for multi-class classification on the MNIST dataset. The test set contains added uncertainty via pixelation and random rotations. Additional calibration measurements of predictions (see Guo et al. (2017) [[2]](http://proceedings.mlr.press/v70/guo17a.html)). Implemented a Bayesian NN class with approproately chosen weight prior and posterior classes for well calibrated results.

*Baseline evaluation:* compound score of both accuracy and empirical expected calibration error (ECE).

## Project 3
Implementation of Bayesian optimization (active Learning) under constraints to the feasible domain (2D grid), following Gelbart et al. (2014) [[3]](https://arxiv.org/abs/1403.5607).  

*Baseline evaluation:* mean normalized regret under constraint satisfaction on 27 different tasks. 


## Project 4
Reinforcement learning task using Generalized Advantage Estimation (GAE) as presented in Schulman et al. (2016) [[4]](https://arxiv.org/abs/1506.02438). It is a model-free policy gradient approach with two neural networks as actor and critic respectively. The control task is to learn a policy to smoothly descend a lunar lander to the ground in between two flags with minimal fuel use and without damaging it. Implemented an actor-critic dual network structure with carying policy gradient methods.

*Baseline evaluation:* estimated expected cumulative reward of the final policy over an episode. 


https://github.com/AtaSoyuer/probabilistic_ai_projects/assets/56206273/0ed00bd8-6afd-49fd-9f35-57b3d02d68e2


*Fig:* Visualisation of the lander acting under the optimal policy computed at evaluation time.


