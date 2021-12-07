import json
import math
from geopy import distance as Distance
from preprocess import user_to_coordinates
from SpatialLabelPropagation import distance

abs_threshold = 15
mse2threshold = 1
mse3threshold = 0.01

def readFile(text):
    """
    Read file
    :param text: json / txt file
    :return: diectionary data read from file
    """
    f = open(text)
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
def mse(text, dim):
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
            threshold = mse2threshold
        else:
            true_x, true_y, true_z = coord(float(true_lat), float(true_lot))
            pred_x, pred_y, pred_z = coord(float(pred_lat), float(pred_lot))
            error = math.sqrt((float(true_x) - float(pred_x)) ** 2 + (float(true_y) - float(pred_y)) ** 2 + (float(true_z) - float(pred_z)) ** 2)
            threshold = mse3threshold

        if error <= threshold:
            correct += 1
        errors.append(error)

    accuracy = correct / len(errors)
    return accuracy


# if __name__ == '__main__':
# abs
mean_absolute = absolute("slpmean82_4iter.txt")
median_absolute = absolute("slpmedian82_4iter.txt")
print("mean absolute:", mean_absolute)
print("median absolute", median_absolute)

# mse 2D
mean_mse_2d = mse("slpmean82_4iter.txt", 2)
median_mse_2d = mse("slpmedian82_4iter.txt", 2)
print("mean mse 2D:", mean_mse_2d)
print("median mse 2D:", median_mse_2d)

# mse 3D
mean_mse_3d = mse("slpmean82_4iter.txt", 3)
median_mse_3d = mse("slpmedian82_4iter.txt", 3)
print("mean mse 3D:", mean_mse_3d)
print("median mse 3D:", median_mse_3d)
