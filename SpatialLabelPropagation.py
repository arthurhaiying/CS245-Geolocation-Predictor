from collections.abc import Sequence
from geopy import distance as Distance
import pandas as pd
import json
import numpy as np
import random
import math
from preprocess import mention_graph, data_rows, user_to_coordinates


INFINITY = np.inf

def distance(p1, p2):
    # compute great circle distance between two coordinates
    return Distance.great_circle(p1, p2).kilometers


def geometric_mean(points, weighted=False, weights=None):
    n_points = len(points)
    if weighted:
        if len(weights) != n_points:
            raise ValueError("Weights must have same size as points")
        # if not all(isinstance(w, int) for w in weights):
        #     raise ValueError("for mean of points, weight must be the number of mentions")

    if n_points == 1:
        return points[0]

    if weighted:
        # for weighted mean, duplicate points based on its weight (num of mentions)
        points2 = [] 
        for w,p in zip(weights, points):
            points2 += [p]*int(w)
        points = points2

    x, y, z = 0.0, 0.0, 0.0 # mean on 2D plane
    for p in points:
        lat, lot = float(p[0]), float(p[1])
        lat, lot = math.radians(lat), math.radians(lot)
        x += math.cos(lat) * math.cos(lot)
        y += math.cos(lat) * math.sin(lot)
        z += math.sin(lat)

    # compute mean on 2D plane
    x = x / n_points
    y = y / n_points
    z = z / n_points

    # convert mean to sphere coord
    center_lot = math.atan2(y, x)
    center_square_root = math.sqrt(x**2 + y**2)
    center_lat = math.atan2(z, center_square_root)
    center_point = math.degrees(center_lat), math.degrees(center_lot)
    return center_point
    

def geometric_median(points,weighted=False,weights=None):
    n_points = len(points)
    if weighted and len(weights) != n_points:
        raise ValueError("Weights must have same size as points")

    if n_points == 1: 
        return points[0]
    # select one point with minimum total (weighted) distance to all points
    p_min = None
    dist_sum_min = INFINITY
    for p in points:
        if not weighted:
            dist_sum = sum(distance(p,p2) for p2 in points)
        else:
            dist_sum = sum(int(w)*distance(p,p2) for w,p2 in zip(weights,points))
        if dist_sum < dist_sum_min:
            dist_sum_min = dist_sum
            p_min = p

    return p_min


class SpatialLabelPropagator:
    # Define different select methods
    select_method_dict = {"GEO_MEDIAN":geometric_median, 
                           "GEO_MEAN": geometric_mean}

    def __init__(self, mention_graph, train_nodes, true_label_dict, select_method, weighted=False, max_iter=1000):
        self.mention_graph = mention_graph
        self.nodes = mention_graph.keys()
        self.train_nodes = train_nodes
        #self.test_nodes = test_nodes
        self.true_label_dict = true_label_dict 
        self.weighted = weighted
        self.max_iter = max_iter
        # check argument validity
        for node in train_nodes:
            if node not in true_label_dict.keys():
                raise ValueError("Node {} must have a label".format(node))
        if select_method not in self.select_method_dict.keys():
            raise ValueError("Select method must be GEO_MEDIAN or GEO_MEAN")
        self.select_method = self.select_method_dict[select_method]

        self.estimated_label_dict = None
        # store estimated labels

    def set_select_method(self, select_method):
        if select_method not in self.select_method_dict.keys():
            raise ValueError("Invalid select method")
        self.select_method = self.select_method_dict[select_method]

    def labelprop(self):
        estimated_label_dict = {} # store estimated labels
        estimated_label_dict.update(self.true_label_dict)
        # initially only train nodes have labels
        for i in range(self.max_iter):
            print("Iter: {} labels: {}".format(i, estimated_label_dict))
            estimated_label_dict = self.update_labels(estimated_label_dict)
        self.estimated_label_dict = estimated_label_dict

    def update_labels(self, estimated_label_dict):
        '''  propagate labels for one step '''
        next_estimated_label_dict = {}
        for node in self.nodes:
            if node in self.train_nodes:
                # skip train nodes
                next_estimated_label_dict[node] = self.true_label_dict[node]
                continue

            locations, weights = [], []
            for k, w in self.mention_graph[node].items():
                # for each neighbor k that has estimated/true label
                if k in estimated_label_dict.keys():
                    locations.append(estimated_label_dict[k])
                    weights.append(int(w))
            if len(locations) != 0:
                # compute new label
                new_label = self.select_method(locations,weighted=self.weighted,weights=weights)
                next_estimated_label_dict[node] = new_label

        return next_estimated_label_dict

    def predict(self, test_nodes):
        test_label_dict = {k:v for k,v in self.estimated_label_dict.items()  if k in test_nodes}
        return test_label_dict

##########################################################################
# Test cases
###########################################################################


def test_case1():
    # test_mention_graph = {
    #     'usr0': {'usr1': 2, 'usr2': 3},
    #     'usr1': {'usr0': 2, 'usr3': 4},
    #     'usr2': {'usr0': 3, 'usr3': 2, 'usr4': 1},
    #     'usr3': {'usr1': 4, 'usr2': 2, 'usr4': 2},
    #     'usr4': {'usr2': 1, 'usr3': 2}
    # }

    # location1 = (20, 50) # LA
    # location2 = (-40, -50) # SH
    # train_nodes = ['usr0', 'usr3']
    # test_nodes = ['usr1', 'usr2', 'usr4']
    # true_label_dict = {'usr0': location1, 'usr3': location2}
    # model = SpatialLabelPropagator(test_mention_graph, train_nodes, true_label_dict, "GEO_MEDIAN", weighted=False, max_iter=10)
    # model.labelprop()
    # test_labels = model.predict(test_nodes)

    total_length = len(data_rows)
    train_length = int(0.8 * total_length)
    # test_length = 0.2 * total_length
    train_nodes = []
    test_nodes = []
    for i in range(train_length):
        train_nodes.append(data_rows[i]['user_id'])
    for i in range(train_length, total_length):
        test_nodes.append(data_rows[i]['user_id'])


    model = SpatialLabelPropagator(mention_graph, train_nodes, user_to_coordinates, "GEO_MEDIAN", weighted=True, max_iter=5)
    model.labelprop()
    test_labels = model.predict(test_nodes)

    # print("Test labels: {}".format(test_labels))
    with open('slpmedian.txt', 'w') as outfile:
        json.dump(test_labels, outfile)

    # model = SpatialLabelPropagator(test_mention_graph, train_nodes, true_label_dict, "GEO_MEAN", weighted=False, max_iter=10)
    # model.labelprop()
    # test_labels = model.predict(test_nodes)

    model = SpatialLabelPropagator(mention_graph, train_nodes, user_to_coordinates, "GEO_MEAN", weighted=True, max_iter=5)
    model.labelprop()
    test_labels = model.predict(test_nodes)
    # print("test labels2: {}".format(test_labels))
    with open('slpmean.txt', 'w') as outfile:
        json.dump(test_labels, outfile)


def test_median():
    total_length = len(data_rows)
    train_length = int(0.8 * total_length)
    # test_length = 0.2 * total_length
    train_nodes = []
    test_nodes = []
    for i in range(train_length):
        train_nodes.append(data_rows[i]['user_id'])
    for i in range(train_length, total_length):
        test_nodes.append(data_rows[i]['user_id'])
    model = SpatialLabelPropagator(mention_graph, train_nodes, user_to_coordinates, "GEO_MEDIAN", weighted=True, max_iter=5)
    model.labelprop()
    test_labels = model.predict(test_nodes)

    user = list(test_labels.keys())[0]
    print("Test labels: {}".format(test_labels[user]))


    # with open('slpmedian.txt', 'w') as outfile:
    #     json.dump(test_labels, outfile)

    # model.set_select_method("GEO_MEAN")
    # model.labelprop()
    # test_labels = model.predict(test_nodes)
    # # print("test labels2: {}".format(test_labels))
    # with open('slpmean.txt', 'w') as outfile:
    #     json.dump(test_labels, outfile)

def test_mean():
    total_length = len ( data_rows )
    train_length = int ( 0.8 * total_length )
    # test_length = 0.2 * total_length
    train_nodes = []
    test_nodes = []
    for i in range ( train_length ):
        train_nodes.append ( data_rows[i]['user_id'] )
    for i in range ( train_length, total_length ):
        test_nodes.append ( data_rows[i]['user_id'] )
    model = SpatialLabelPropagator ( mention_graph, train_nodes, user_to_coordinates, "GEO_MEAN", weighted=True, max_iter=5 )
    model.labelprop ()
    test_labels = model.predict ( test_nodes )

    user = list(test_labels.keys())[0]
    print("test labels2: {}".format(test_labels[user]))

    # with open ( 'slpmean.txt', 'w' ) as outfile:
    #     json.dump ( test_labels, outfile )


if __name__ == '__main__':
    # p1 = (47.528139,-122.197916)
    # p2 = (40.668643,-73.981635)
    # p3 = (41.876133,-87.674191)
    # d_12 = distance(p1, p2)
    # d_13 = distance(p1, p3)
    # d_23 = distance(p2, p3)
    # median = geometric_median([p1,p2,p3],weighted=True,weights=[0.6,0.2,0.2])
    # mean = geometric_mean([p1,p2,p3])
    # mean2 = geometric_mean([p1,p2,p3],weighted=True,weights=[1,2,1])
    # print("distances 12: {} 23: {}".format(d_12, d_23))
    # print("median: {}".format(median))
    # print("mean: {}".format(mean))
    # print("mean2: {}".format(mean2))
    # test cases
    # print("Start test cases...")
    # test_median()
    # print("Finish test cases.")

    # print ( "Start test cases..." )
    # test_mean()
    # print ( "Finish test cases." )

    print ( "Start test cases..." )
    test_case1()
    print ( "Finish test cases." )