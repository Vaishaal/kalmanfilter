import csv
import argparse
import numpy as np
from matplotlib import pyplot
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def kalman(x,p,g, truth, measurement, mm, update, update_noise, n_iter=1000):
    fig = pylab.plt.figure()
    # set up
    AAT_inv = np.linalg.inv((mm.T.dot(mm)))
    x[0,0] = AAT_inv.dot(mm.T.dot(measurement[0]))
    p[0,0] = AAT_inv
    predictions = []
    for k in range(n_iter):
        x[k+1,k] = update.dot(x[k,k])
        p[k+1,k] = update.dot(p[k,k]).dot(update) + np.identity(2)
        mmk = mm
        s = np.linalg.inv(mmk.dot(p[k+1,k]).dot(mmk) + np.identity(2))
        g[k+1] = p[k+1,k].dot(mmk).dot(s)
        x[k+1,k+1] = x[k+1,k] + g[k+1].dot(measurement[k+1] - mmk.dot(x[k+1,k]))
        predictions.append(x[k+1,k+1])
        p[k+1,k+1] = (np.identity(2) - g[k+1].dot(mmk)).dot(p[k+1,k])

    predictions = np.array(predictions)
    old_error = sum(np.linalg.norm(measurement[:n_iter] - truth[:n_iter], axis=1))
    new_error = sum(np.linalg.norm(predictions - truth[:n_iter], axis=1))
    return new_error, old_error
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A kalman filter')
    parser.add_argument('measurement', help='csv of measurements')
    parser.add_argument('truth', help='csv of true positions')
    args = parser.parse_args()
    measurements = np.genfromtxt(args.measurement, delimiter=",")
    truth = np.genfromtxt(args.truth, delimiter=",")
    update = np.identity(2)
    m,n = measurements.shape
    # part 1
    x = np.zeros((m,m,n))
    p = np.zeros((m,m,n,n))
    g = np.zeros((m,n,n))
    update_noise = np.random.multivariate_normal((0,0), np.identity(2), measurements.shape[0])
    print kalman(x,p,g,truth, measurements, np.identity(2), update, update_noise, 500)

    # part 2
    '''
    n = 4
    x = np.zeros((m,m,n))
    p = np.zeros((m,m,n,n))
    g = np.zeros((m,n,n))
    mm = np.vstack((np.identity(2), np.zeros((2,2))))
    update_noise = np.random.multivariate_normal((0,0,0,0), np.identity(4), measurements.shape[0])
    print kalman(x,p,g,truth, measurements, mm, update, update_noise, 1000)
    '''


