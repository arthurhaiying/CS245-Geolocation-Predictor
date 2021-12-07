# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:15:47 2021

@author: luche
"""

from preprocess import *
import pickle
from VanillaLabelPropagation import *
import random

def test_vanilla_on_dataset():
    data_rows, user_to_mentions, user_to_coordinates = read_dataset_without_saving(dataset_filename, csv_filename)
    mention_graph = build_mention_graph(user_to_mentions)
        
    idx_to_user = {}
    for i, (k, v) in enumerate(user_to_coordinates.items()):
        idx_to_user[i] = k
    
    #print("computing coordinates_to_cities")
    #user_to_city = coordinates_to_cities(user_to_coordinates)
    #print(len(user_to_city))
    # save
    #print("saving")
    #with open("user_to_city.pkl", "wb") as pkl_handle:
    #    pickle.dump(user_to_city, pkl_handle)
    #print("saved!")
    
    # load
    print("load user_to_city")
    with open("user_to_city.pkl", "rb") as pkl_handle:
        user_to_city = pickle.load(pkl_handle)
    
    user_to_label, counter = cities_to_labels(user_to_city)
    
    Y = build_label_distribution(user_to_label, counter)
    
    print("Y: ", Y.shape)
    
    W = build_weight_matrix(mention_graph)
    
    print("W: ", W.shape)

    T = computer_transition_matrix(W)
    
    print("T: ", T.shape)
    
    num_masked_out = 5000
    num_samples = Y.shape[0]#9475
    num_labels = Y.shape[1]#2419
    
    print("num_samples ", num_samples)
    print("num_labels ", num_labels)
    
    mask = np.zeros((num_samples, num_samples))
    
    masked_out = random.sample(range(num_samples), num_masked_out)
    
    for idx in masked_out:
        mask[idx, idx] = 1
        
    print("mask: ", mask.shape)
    
    ground_truth = Y
    
    # only keep part of the ground truth label
    Y = Y - np.matmul(mask, Y)
    
    for idx in masked_out:
        Y[idx] = 1/num_labels
    
    # vanilla label propagation
    print("vanilla label propagation")
    Y_pred = vanilla_label_propagation(Y, T, mask, 10, ground_truth, num_masked_out)
    
    # oracle label propagation
    print("oracle label propagation")
    oracle_predictions = oracle_label_propagation(ground_truth, T)
    
    print("\n")
    
    # check the graph
    for idx in masked_out[:10]: 
        print("user: ", idx_to_user[idx])
        print("vanilla's prediction: ", np.argmax(Y_pred[idx]))
        print("oracle's prediction: ", np.argmax(oracle_predictions[idx]))
        displace_user_information(idx, idx_to_user, mention_graph, user_to_label, W, Y_pred)
        print("\n")
        
if __name__ == "__main__":
    test_vanilla_on_dataset()