#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def single_layer_forward_propagation(A_prev, W_curr, b_curr):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    return sigmoid(Z_curr), Z_curr

def backprop(x, y, biases, weightsT, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and transposed weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_wT): tuple containing the gradient for all the biases
                and weightsT. nabla_b and nabla_wT should be the same shape as 
                input biases and weightsT
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_wT = [np.zeros(wT.shape) for wT in weightsT]

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    ###
    
    activations = []
    A_curr = x
    activations.append(A_curr)
    before_activation = []
    for hidden_layer in range(1, num_layers):
        A_prev = A_curr
        W_curr = weightsT[hidden_layer - 1]
        b_curr = biases[hidden_layer - 1]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr)
        activations.append(A_curr)
        before_activation.append(Z_curr)

    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).df_wrt_a(activations[-1], y)
    
    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###
    
    for hidden_layer in range(num_layers-1, 0, -1):
        delta = np.multiply(delta, sigmoid_prime(before_activation[hidden_layer-1]))
        nabla_b[hidden_layer-1] = delta
        nabla_wT[hidden_layer-1] = np.dot(activations[hidden_layer-1], delta.T).T
        delta = np.dot(weightsT[hidden_layer-1].T, delta)
        
    return (nabla_b, nabla_wT)

