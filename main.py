import argparse
import glob
import os
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from scipy.signal import find_peaks


class Measurement:

    def __init__(self, filename):
        parts = os.path.basename(filename).split("_")

        self.molecule = parts[4]
        self.date = parts[5]

        with open(filename) as f:
            lines = f.readlines()
            lines = [line.replace("\n", "").strip().split("\t") for line in lines]
            lines = {datetime.strptime((line[0] + " " + line[1]), '%m/%d/%Y %H:%M:%S'): [line[2]] for line in lines}

        self.data = lines

        with open(filename.replace("C1-01", "C1-02")) as f:
            lines = f.readlines()
            lines = [line.replace("\n", "").strip().split("\t") for line in lines]
            for line in lines:
                self.data[datetime.strptime((line[0] + " " + line[1]), '%m/%d/%Y %H:%M:%S')].append(line[2])

        arr = np.asarray(list(self.data.keys()))
        arr = [a.total_seconds() for a in arr - arr[0]]
        vals = list(self.data.values())

        self.data = {arr[k]: vals[k] for k in range(len(arr))}


def read_measurements(folder_path):
    files = glob.glob(os.path.join(folder_path, "**C1-01.ASC"))

    measurements = []
    for f in files:
        try:
            measurements.append(Measurement(f))
        except:
            warnings.warn(f"Error occurred while processing {f}, skipping. "
                          f"Please check if both files exist and the data is separated using tab character.")

    return measurements


def parse_args():
    parser = argparse.ArgumentParser()

    # Input folder path
    parser.add_argument('-p', '--folderpath', default="dataornek", required=True)

    # Optional parameters
    parser.add_argument('-l', '--leakage', default=2.000E-06)
    parser.add_argument('-ma', '--membranearea', default=44)
    parser.add_argument('-mt', '--membranethickness', default=76)
    parser.add_argument('-pv', '--productvolume', default=30.1)
    parser.add_argument('-t', '--temperature', default=308.15)
    parser.add_argument('-gc', '--gasconstant', default=0.2780405104)

    # Filtering parameters
    parser.add_argument('-ci', '--cropinterval', default=0.02)
    parser.add_argument('-amn', '--anglemin', default=10)
    parser.add_argument('-amx', '--anglemax', default=70)
    parser.add_argument('-me', '--minelement', default=10)

    args = parser.parse_args()
    return args


def process_measurements(measurements, args):
    x_data = np.asarray(list(measurements.data.keys()))
    y_data = np.asarray(list(measurements.data.values()), dtype=np.float32)
    conv_diff = np.convolve(y_data[:, 1], np.array([1 / 3, 1 / 3, 1 / 3]), mode='same')
    # Find peak points
    ups_and_downs = np.convolve(conv_diff, np.array([1, 0, -1]), mode='same')
    ups_and_downs = np.convolve(ups_and_downs, np.ones(5), mode='same')

    peaks = find_peaks(abs(ups_and_downs), threshold=0.01)
    peaks = np.maximum(0, peaks[0] - 1)

    # Plot the data
    plt.plot(x_data, y_data[:, 1])

    # Crop between two red points and apply ransac
    peaks = np.insert(peaks, 0, 0)
    peaks = np.append(peaks, len(x_data) - 2)
    res_peaks = []
    for i in range(len(peaks) - 1):
        x_crop = np.arange(peaks[i] + 5, peaks[i + 1])
        y_crop = y_data[:, 1][x_crop]
        if len(x_crop) < args.minelement:
            continue
        mn, mx = min(y_crop), max(y_crop)
        interval = mx - mn
        step = interval * args.cropinterval
        interval_min = mn + step
        interval_max = mx - step
        y_mask = (y_crop > interval_min) * (y_crop < interval_max)

        if np.sum(y_mask) < args.minelement:
            continue

        x_segment = x_data[x_crop[y_mask]]
        y_segment = y_data[x_crop[y_mask], 1]
        y_segment_vals = y_data[x_crop[y_mask], 0]

        # Fit line to segment

        x_data_interval = (max(x_data) - min(x_data))
        y_data_interval = (max(y_data[:, 1]) - min(y_data[:, 1]))

        x_norm = (x_segment - min(x_data)) / x_data_interval
        y_norm = (y_segment - min(y_data[:, 1])) / y_data_interval

        poly = np.polyfit(x_norm, y_norm, 1)
        angle = np.rad2deg(np.arctan(poly[0]))
        slope = poly[0]

        if angle < args.anglemin or angle > args.anglemax:
            continue

        # Check inliers and extend the lines
        distances = np.abs(((y_segment - min(y_segment)) / y_data_interval) -
                           (x_segment - min(x_segment)) / x_data_interval * poly[0] + poly[1])
        max_dist = np.max(np.abs(y_norm - x_norm * poly[0] + poly[1]))
        inliers = distances < max_dist

        plt.plot(x_segment[inliers], y_segment[inliers], color="red")
        plt.scatter(x_data[peaks[i + 1]], y_data[peaks[i + 1], 1], s=20, facecolors='none', edgecolors='r')

        y_inverse_norm = (x_norm * poly[0] + poly[1]) * y_data_interval + min(y_data[:, 1])
        r2_score = sklearn.metrics.r2_score(y_inverse_norm, y_segment)

        if r2_score < 0.9:
            print(f"Measurement {measurements.molecule} has r2 score of {r2_score}. Skipping.")
            continue

        Gas_constant_STP = 0.082 * 76 * 1000 / 22414
        P_1 = np.power(10, 10) * (slope - args.leakage) * args.productvolume * \
              (args.membranethickness * np.power(1/10, 4)) / ((
                (args.membranearea * np.power(1/10, 2)) * Gas_constant_STP * (args.temperature + 273.15) * np.mean(
            y_segment_vals)))
        Q_1 = (P_1 / (args.membranethickness * np.power(1/10, 4))) / 10000

        print("====================")
        print(f"Measurement Name: {measurements.molecule}")
        print(f"P_1: {P_1}")
        print(f"Q_1: {Q_1}")
        print("====================")

    plt.title("Original Data")
    plt.ylabel("mbar")
    plt.xlabel("time")
    plt.legend()
    plt.show()

    pass


def main():
    args = parse_args()
    measurements = read_measurements(args.folderpath)
    for measurement in measurements:
        process_measurements(measurement, args)


# This is the main script to parse molecule reading and generate report
if __name__ == '__main__':
    main()
