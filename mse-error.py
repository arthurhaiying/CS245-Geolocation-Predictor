import json
import math
from geopy import distance as Distance
from preprocess import user_to_coordinates
from SpatialLabelPropagation import distance

abs_threshold = 1
mse_threshold = 1

def readFile(text):
    """

    :param text: json / txt file
    :return: diectionary data read from file
    """
    f = open('slpmean.txt')
    data = json.load(f)
    f.close()
    return data

def coord(lat, lot):
    """
    Transfer [lo, la] to [x, y, z]

    :param lo: float
    :param la: float
    :return: list of [x, y, z]
    """
    lat, lot = math.radians(lat), math.radians(lot)
    x = math.cos(lat) * math.cos(lot)
    y = math.cos(lat) * math.sin(lot)
    z = math.sin(lat)
    return x, y, z


# absolute distance
def absolute(text):
    """
    Calculate the absolute error

    :return: list of absolute errors
    """
    data = readFile(text)
    errors = []
    correct = 0

    for user, location in data.items():
        p1 = user_to_coordinates[user]
        p2 = location
        error = distance(p1, p2)
        if error <= abs_threshold:
            correct += 1
        errors.append(error)

    accuracy = correct / len(errors)
    return accuracy


# mse
def mse(text, dim=3):
    """
    Calculate the mse error

    dim can be 2 / 3

    :return: list of mse errors
    """
    data = readFile(text)
    errors = []
    correct = 0

    for user, location in data.items():
        true_lat, true_lot = user_to_coordinates[user]
        pred_lat, pred_lot = location
        if dim == 2:
            error = math.sqrt((float(true_lat) - float(pred_lat)) ** 2 + (float(true_lot) - float(pred_lot)) ** 2)
        else:
            true_x, true_y, true_z = coord(true_lat, true_lot)
            pred_x, pred_y, pred_z = coord(pred_lat, pred_lot)
            error = math.sqrt((float(true_x) - float(pred_x)) ** 2 + (float (true_y) - float(pred_y)) ** 2 + (float(true_z) - float(pred_z)) ** 2)

        if error <= mse_threshold:
            correct += 1
        errors.append(error)

    accuracy = correct / len(errors)
    return accuracy


if __name__ == '__main__':
    # abs
    mean_absolute = absolute("slpmean.txt")
    median_absolute = absolute("slpmedian.txt")
    print("mean absolute:", mean_absolute)
    print("median absolute", median_absolute)

    # mse 2D
    mean_mse = mse("slpmean.txt", 2)
    median_mse = mse("slpmedian.txt", 2)
    print("mean mse:", mean_mse)
    print("median mse", median_mse)

    # mse 3D
    mean_mse = mse("slpmean.txt")
    median_mse = mse("slpmedian.txt")
    print("mean mse:", mean_mse)
    print("median mse", median_mse)
