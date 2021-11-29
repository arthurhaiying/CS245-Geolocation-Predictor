# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:32:36 2021

@author: luche
"""
import numpy as np

def computer_transition_matrix(W):
  """
  Compute the nxn transition matrix T, where n is the number of 
  nodes. Return T.

  Parameters:
  W: nxn weight matrix, where W_ij measures the similarity between
    node i and node j
  """

  return W/np.sum(W,axis=0)

def vanilla_label_propagation(Y, T, mask, max_iter):
  """
  Do vanilla label propagation until convergence. Return the 
  new label distribution

  Parameters:
  Y: nxc matrix of the original label distribution,
    where n is the number of nodes, c is the number of labels.
    each row of Y is a distribution over c labels
  T: nxn probability transition matrix, where n is the number of 
    nodes. T_ij is the probability to jump from node j to i
  mask: nxn diagonal matrix that chooses the rows without
    ground truth label. mask_ii is 1 if node i does not 
    have ground truth label and 0 otherwise.
  max_iter: maximum number of iteration
     
  """
  Y_cur = Y
  Y_prev = np.zeros(Y.shape)
  counter = 0
  while not np.allclose(Y_cur, Y_prev) and counter < max_iter:
    Y_prev = Y_cur
    # propagate
    Y_cur = np.matmul(T, Y_cur)
    # row-normalize
    Y_cur = Y_cur/np.sum(Y_cur,axis=1,keepdims=True)
    Y_cur = np.nan_to_num(Y_cur)
    # clamp the labeled data
    Y_cur = np.matmul(mask, Y_cur) + Y-np.matmul(mask, Y)
    #print(Y_cur)
    # increase the counter
    counter += 1
  
  print("counter: ", counter)
  return Y_cur


def run_test_case():
    W = np.array([[1,0.5,0.5,0,0,0],
                  [0.5,1,0.5,0.4,0,0],
                  [0.5,0.5,1,0,0,0],
                  [0,0.4,0,1,0.5,0.5],
                  [0,0,0,0.5,1,0],
                  [0,0,0,0.5,0,1]])
    print("W: ", W)
    T = computer_transition_matrix(W)
    print("T: ", T)
    Y = np.array([[1,0,0],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [0,1,0],])
    print("Y: ", Y)
    mask = np.array([[0,0,0,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,1,0,0,0],
                     [0,0,0,1,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,0],])
    print("mask: ", mask)
    
    print("results: ")
    print(vanilla_label_propagation(Y, T, mask, 1000))

if __name__ == "__main__":
    run_test_case()