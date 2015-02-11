"""Example of a Swarm Particle Optimization.
"""


import numpy as np
from numpy.random import multivariate_normal as gaussian
from random import randrange


OUTPUTS_DIR = 'outputs/swarm/'


if __name__ == '__main__':
    import pylab

    start = np.array([10, 20])
    arrival = np.array([75, 90])
    covmatrix = np.diag(np.array([9, 16]))

    num_particles = 100
    position = gaussian(start, covmatrix, num_particles)
    best = position
    distances = np.linalg.norm(position - arrival, axis=1)
    gbest = position[distances.argsort()[0]]
    speed = np.random.rand(100, 2)

    fig = pylab.gcf()
    for i in range(100):
        print(i)
        pylab.plot(start[0], start[1], 'ro')
        pylab.plot(arrival[0], arrival[1], 'bo')
        pylab.plot(position.T[0], position.T[1], 'r.')
        pylab.plot(gbest.T[0], gbest.T[1], 'g.')

        # updating
        # BUG: speed doesn't stop increasing
        speed = speed + 2*np.random.rand(100, 2)*(best - position) +\
                2*np.random.rand(100, 2)*(gbest - position)
        position = position + speed

        # calculating best positions
        best_distances = np.linalg.norm(best - arrival, axis=1)
        distances = np.linalg.norm(position - arrival, axis=1)
        indices = best[distances < best_distances].argsort()
        best[indices] = position[indices]
        best_distances = np.linalg.norm(best - arrival, axis=1)
        gbest = best[best_distances.argsort()[0]]

        pylab.xlim(0, 100)
        pylab.ylim(0, 100)
        fig.tight_layout()
        pylab.savefig('%siter-%03d.png' % (OUTPUTS_DIR, i))
        fig.clf()