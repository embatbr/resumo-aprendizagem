import numpy as np
from numpy.random import multivariate_normal as gaussian
import os


DATASET_DIR = 'dataset/'
OUTPUTS_DIR = 'outputs/'


# Classe 1a
means1a = np.array([60, 30])
cov1a = np.diag(np.array([9, 144]))

# Classe 1b
means1b = np.array([52, 30])
cov1b = np.diag(np.array([9, 9]))

# Classe 2
means2 = np.array([45, 22])
cov2 = np.diag(np.array([100, 9]))

def generate_dataset(concat=True, shuffle=True, save=False):
    data1a = gaussian(means1a, cov1a, 100)
    data1b = gaussian(means1b, cov1b, 100)
    data2 = gaussian(means2, cov2, 100)

    if save:
        if not os.path.exists(DATASET_DIR):
            os.mkdir(DATASET_DIR)
        np.save('%sdata1a' % DATASET_DIR, data1a)
        np.save('%sdata1b' % DATASET_DIR, data1b)
        np.save('%sdata2' % DATASET_DIR, data2)

    if concat:
        data = np.concatenate((data1a, data1b, data2))
        if shuffle:
            np.random.shuffle(data)
        return data

    return(data1a, data1b, data2)

def read_data(filenames):
    dataset = list()

    for filename in filenames:
        data = np.load('%s%s.npy' % (DATASET_DIR, filename))
        dataset.append(data)

    return dataset


if __name__ == '__main__':
    import pylab

    try:
        dataset = read_data(['data1a', 'data1b', 'data2'])
        (data1a, data1b, data2) = tuple(dataset)
        print('dataset read')
    except IOError:
        (data1a, data1b, data2) = generate_dataset(False, False, True)
        print('dataset generated')

    if not os.path.exists(OUTPUTS_DIR):
        os.mkdir(OUTPUTS_DIR)

    fig = pylab.figure()
    pylab.plot(data1a.T[0], data1a.T[1], 'r.')
    pylab.plot(data1b.T[0], data1b.T[1], 'r.')
    pylab.plot(data2.T[0], data2.T[1], 'b.')
    pylab.plot(means1a.T[0], means1a.T[1], 'ro')
    pylab.plot(means1b.T[0], means1b.T[1], 'ro')
    pylab.plot(means2.T[0], means2.T[1], 'bo')
    pylab.xlabel('$x_1$')
    pylab.ylabel('$x_2$')
    pylab.savefig('%ssamples_by_class.png' % OUTPUTS_DIR)

    fig = pylab.figure()
    pylab.plot(data1a.T[0], data1a.T[1], 'g.')
    pylab.plot(data1b.T[0], data1b.T[1], 'y.')
    pylab.plot(data2.T[0], data2.T[1], 'b.')
    pylab.plot(means1a.T[0], means1a.T[1], 'go')
    pylab.plot(means1b.T[0], means1b.T[1], 'yo')
    pylab.plot(means2.T[0], means2.T[1], 'bo')
    pylab.xlabel('$x_1$')
    pylab.ylabel('$x_2$')
    pylab.savefig('%ssamples_by_cluster.png' % OUTPUTS_DIR)

    pylab.show()