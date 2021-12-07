# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:32:36 2021

@author: luche
"""
import numpy as np
from sklearn.preprocessing import normalize

def computer_transition_matrix(W):
  """
  Compute the nxn transition matrix T, where n is the number of 
  nodes. Return T.

  Parameters:
  W: nxn weight matrix, where W_ij measures the similarity between
    node i and node j
  """
  return normalize(W, axis=1, norm='l1')

def vanilla_label_propagation(Y, T, mask, max_iter, ground_truth, num_masked_out):
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
    Y_cur = normalize(Y_cur, axis=1, norm='l1')
    # clamp the labeled data
    Y_cur = np.matmul(mask, Y_cur) + Y-np.matmul(mask, Y)
    # increase the counter
    counter += 1
    print("iteration: ", counter)
    
    # compute accuracy
    num_correct = num_masked_out - np.count_nonzero(np.argmax(ground_truth, axis = 1)- np.argmax(Y_cur, axis = 1))
    print("num_correct: ", num_correct)          
    print("accuray: ", num_correct/num_masked_out)
    
  return Y_cur


def oracle_label_propagation(ground_truth, T):
    """
    Assume we know the ground truth label of the neighbors when predicting 
    the label for each node.
    """
    np.fill_diagonal(T,0)
    predictions = np.matmul(T, ground_truth)
    num_correct = ground_truth.shape[0] - np.count_nonzero(np.argmax(ground_truth, axis = 1)- np.argmax(predictions, axis = 1))
    print("num_correct: ", num_correct)          
    print("accuray: ", num_correct/ground_truth.shape[0])
    return predictions


def run_test_case():
    W = np.array([[0,0.5,0.6,0,0,0],
                  [0.5,0,0.5,0.4,0,0],
                  [0.6,0.5,0,0,0,0],
                  [0,0.4,0,0,0.5,0.5],
                  [0,0,0,0.5,0,0],
                  [0,0,0,0.5,0,0]])
    print("W: ")
    print(W)
    T = computer_transition_matrix(W)
    print("T: ", T)
    Y = np.array([[1,0,0],
                  [0,0,1],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [1/3,1/3,1/3],
                  [0,1,0],])
    print("Y: ", Y)
    mask = np.array([[0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,1,0,0,0],
                     [0,0,0,1,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,0],])
    print("mask: ", mask)
    
    ground_truth = np.array([[1,0,0],
                             [0,0,1],
                             [1,0,0],
                             [0,1,0],
                             [0,1,0],
                             [0,1,0],])
    
    print("vanilla label propagation")
    print(vanilla_label_propagation(Y, T, mask, 10, ground_truth, 3))
    print("oracle label propagation")
    print(oracle_label_propagation(ground_truth, T))

if __name__ == "__main__":
    run_test_case()