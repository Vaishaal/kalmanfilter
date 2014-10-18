import csv
import argparse
import numpy as np
from matplotlib import pyplot
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def kalman(truth, measurement, mm, update, update_noise, n_iter=1000):
    fig = pylab.plt.figure()
    m,n = measurement.shape
    x = np.zeros((m,m,n))
    p = np.zeros((m,m,n,n))
    g = np.zeros((m,n,n))
    # set up
    AAT_inv = np.linalg.inv((mm.T.dot(mm)))
    x[0,0] = AAT_inv.dot(mm.T.dot(measurement[0]))
    p[0,0] = AAT_inv
    predictions = []
    for k in range(n_iter):
        print update_noise[k]
        x[k+1,k] = update.dot(x[k,k]) + update_noise[k]
        p[k+1,k] = update.dot(p[k,k]).dot(update) + np.identity(2)
        mmk = mm
        s = np.linalg.inv(mmk.dot(p[k+1,k]).dot(mmk) + np.identity(2))
        g[k+1] = p[k+1,k].dot(mmk).dot(s)
        x[k+1,k+1] = x[k+1,k] + g[k+1].dot(measurement[k+1] - mmk.dot(x[k+1,k]))
        predictions.append(x[k+1,k+1])
        p[k+1,k+1] = (np.identity(2) - g[k+1].dot(mmk)).dot(p[k+1,k])

    predictions = np.array(predictions)

    print np.linalg.norm(measurement[:n_iter] - truth[:n_iter])
    print np.linalg.norm(predictions - measurement[:n_iter])
    print np.linalg.norm(predictions - truth[:n_iter])
    x,y = zip(*predictions)
    xt,yt = zip(*truth[:n_iter])
    xm,ym = zip(*measurement[:n_iter])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter( xt,yt, range(n_iter))
    ax.scatter( x,y, range(n_iter),c="red")
    ax.scatter( x,y, range(n_iter),c="purple")
    #pylab.plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A kalman filter')
    parser.add_argument('measurement', help='csv of measurements')
    parser.add_argument('truth', help='csv of true positions')
    args = parser.parse_args()
    measurements = np.genfromtxt(args.measurement, delimiter=",")
    truth = np.genfromtxt(args.truth, delimiter=",")
    update = np.identity(4)

    update_noise = np.random.multivariate_normal((0,0), np.identity(2), measurements.shape[0])
    kalman(truth, measurements, np.identity(2), update, update_noise, 1000)



