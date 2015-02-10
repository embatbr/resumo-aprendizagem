"""Training and classification using a k-nearest-neighbors.
"""


import numpy as np


def kNN(x, k, data):
    """Returns the indices for the k nearest neighbours.
    """
    neighbours = list()
    num_neighbours = 0

    distances = np.linalg.norm(x - data, axis=1)    # euclidian distance
    indices = distances.argsort()[: k]

    return (indices, distances[indices])

def classify(training_data, test_data, training_classes, test_classes, k,
             weighted=True):
    """Classifies a dataset given the training data and classes and the expected
    classes.
    """
    classified = list()

    for x in test_data:
        (indices, distances) = kNN(x, k, training_data)
        sum_distances = np.sum(distances)
        neighbours_classes = training_classes[indices]

        if weighted:
            c = np.dot(neighbours_classes, distances) / sum_distances
            c = int(c + 0.5)
        else:
            k1 = len(neighbours_classes[neighbours_classes == 1])
            k2 = len(neighbours_classes[neighbours_classes == 2])
            c = 2 if k2 > k1 else 1
        classified.append(c)

    return np.array(classified)


if __name__ == '__main__':
    from samples import read_data
    import matplotlib.pyplot as plt
    import math

    dataset = read_data(['data1a', 'data1b', 'data2'])
    (data1a, data1b, data2) = tuple(dataset)
    training_data = np.concatenate((data1a[ : 75], data1b[ : 75], data2[ : 75]))
    training_classes = np.concatenate((np.ones(150, np.int8), 2*np.ones(75, np.int8)))
    test_data = np.concatenate((data1a[75 : ], data1b[75 : ], data2[75 : ]))
    test_classes = np.concatenate((np.ones(50, np.int8), 2*np.ones(25, np.int8)))

    num_samples = 0.75 * (len(training_data) + len(test_data))
    k_max = int(math.sqrt(num_samples))

    for k in range(1, k_max + 1, 2):
        print('k = %d' % k)

        for (w, s) in zip([True, False], ['WEIGHTED', 'NON-WEIGHTED']):
            classified = classify(training_data, test_data, training_classes,
                                  test_classes, k, w)
            equal = (classified == test_classes)
            numequal = len(equal[equal == True])
            hits = numequal / len(test_classes)

            print('%s: %.2f%%' % (s, hits*100))