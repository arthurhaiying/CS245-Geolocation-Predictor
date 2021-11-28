from collections.abc import Sequence
from geopy import distance as Distance
import numpy as np
import random
import math

INFINITY = np.inf

def distance(p1, p2):
    # compute great circle distance between two coordinates
    return Distance.great_circle(p1, p2).kilometers


def geometric_mean(points, weighted=False, weights=None):
    n_points = len(points)
    if weighted:
        if len(weights) != n_points:
            raise ValueError("Weights must have same size as points")
        if not all(isinstance(w, int) for w in weights):
            raise ValueError("for mean of points, weight must be the number of mentions")

    if n_points == 1:
        return points[0]
    if weighted:
        # for weighted mean, duplicate points based on its weight (num of mentions)
        points2 = [] 
        for w,p in zip(weights, points):
            points2 += [p]*w
        points = points2

    x, y, z = 0.0, 0.0, 0.0 # mean on 2D plane
    for p in points:
        lat, lot = p[0], p[1]
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
            dist_sum = sum(w*distance(p,p2) for w,p2 in zip(weights,points))
        if dist_sum < dist_sum_min:
            dist_sum_min = dist_sum
            p_min = p

    return p_min

if __name__ == '__main__':
    p1 = (47.528139,-122.197916)
    p2 = (40.668643,-73.981635)
    p3 = (41.876133,-87.674191)
    d_12 = distance(p1, p2)
    d_13 = distance(p1, p3)
    d_23 = distance(p2, p3)
    median = geometric_median([p1,p2,p3],weighted=True,weights=[0.6,0.2,0.2])
    mean = geometric_mean([p1,p2,p3])
    mean2 = geometric_mean([p1,p2,p3],weighted=True,weights=[1,2,1])
    print("distances 12: {} 23: {}".format(d_12, d_23))
    print("median: {}".format(median))
    print("mean: {}".format(mean))
    print("mean2: {}".format(mean2))