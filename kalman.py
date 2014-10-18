import csv
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A kalman filter')
    parser.add_argument('measurement', help='csv of measurements')
    parser.add_argument('truth', help='csv of true positions')
    args = parser.parse_args()
    measurements = np.genfromtxt(args.measurement, delimiter=",")
    truth = np.genfromtxt(args.measurement, delimiter=",")



