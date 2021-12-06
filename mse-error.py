import json
import math
from preprocess import user_to_coordinates

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
        true_lon, true_la = user_to_coordinates[user]
        pred_lon, pred_la = location
        error = abs(float(true_lon) - float(pred_lon)) + abs(float(true_la) - float(pred_la))
        if error <= abs_threshold:
            correct += 1
        errors.append(error)

    accuracy = correct / len(errors)
    return accuracy


# mse
def mse(text):
    """
    Calculate the mse error

    :return: list of mse errors
    """
    data = readFile(text)
    errors = []
    correct = 0

    for user, location in data.items():
        true_lon, true_la = user_to_coordinates[user]
        pred_lon, pred_la = location
        error = math.sqrt((float(true_lon) - float(pred_lon)) ** 2 + (float(true_la) - float(pred_la)) ** 2)
        if error <= mse_threshold:
            correct += 1
        errors.append(error)

    accuracy = correct / len(errors)
    return accuracy


if __name__ == '__main__':
    # mean_absolute = absolute("slpmean.txt")
    # median_absolute = absolute("slpmedian.txt")
    # print("mean absolute:", mean_absolute)
    # print("median absolute", median_absolute)

    mean_mse = mse("slpmean.txt")
    median_mse = mse("slpmedian.txt")
    print("mean mse:", mean_mse)
    print("median mse", median_mse)
