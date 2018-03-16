import sys
import numpy as np


def parse_file(file_path):
    """
    parse input file
    :param file_path: path to data.txt
    :return: X, Y
    """
    training_examples = np.loadtxt(file_path)
    X = training_examples[:,:-1]
    Y = training_examples[:,-1]
    return X,Y


def consistency_algorithm(X,Y):
    """
    consistency algorithm
    :param X: examples
    :param Y: labels
    :return: hypothesis
    """
    ht = np.ones((2,X.shape[1]))
    for iter, x in enumerate(X):
        if Y[iter] == 1 and calc_hypothesis(ht,x) == 0:
            for i, elem in enumerate(x):
                if x[i] == 1:
                    ht[1][i] = 0
                if x[i] == 0:
                    ht[0][i] = 0
    return ht


def calc_hypothesis(h, x):
    """
    calculates label on hypothesis
    :param h: hypothesis
    :param x: example
    :return: label
    """
    for i, elem in enumerate(h[0]):
        if elem and x[i] == 0:
            return 0
    for i, elem in enumerate(h[1]):
        if elem and x[i] == 1:
            return 0
    return 1


def write_to_output(h):
    """
    writes hypothesis to output.txt
    :param h: hypothesis
    :return: N/A
    """
    data = []
    for i in range(h.shape[1]):
        if h[0][i] == 1:
            data.append('x'+str(i+1))
        if h[1][i] == 1:
            data.append('not(x'+str(i+1)+')')
    str_data = ','.join(str(j) for j in data)
    with open('output.txt', 'w') as fo:
        fo.write(str_data)


if __name__ == '__main__':
    file_path = sys.argv[1]
    X,Y = parse_file(file_path)
    h = consistency_algorithm(X, Y)
    write_to_output(h)