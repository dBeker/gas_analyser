import glob
import os
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from scipy.signal import find_peaks


class Measurement:

    def __init__(self, filename, args):
        parts = os.path.basename(filename).split("_")

        self.molecule = parts[4]
        self.date = parts[5]

        with open(filename) as f:
            lines = f.readlines()
            lines = [line.replace("\n", "").strip().split("\t") for line in lines]
            lines = {datetime.strptime((line[0] + " " + line[1]), '%m/%d/%Y %H:%M:%S'): [line[2]] for line in lines}

        self.data = lines

        with open(filename.replace(args.fileending01, args.fileending02)) as f:
            lines = f.readlines()
            lines = [line.replace("\n", "").strip().split("\t") for line in lines]
            for line in lines:
                self.data[datetime.strptime((line[0] + " " + line[1]), '%m/%d/%Y %H:%M:%S')].append(line[2])

        arr = np.asarray(list(self.data.keys()))
        arr = [a.total_seconds() for a in arr - arr[0]]
        vals = list(self.data.values())

        self.data = {arr[k]: vals[k] for k in range(len(arr))}


def read_measurements(args):
    files = glob.glob(os.path.join(args.folderpath, f"**{args.fileending01}"))

    measurements = []
    for f in files:
        try:
            measurements.append(Measurement(f, args))
        except:
            warnings.warn(f"Error occurred while processing {f}, skipping. "
                          f"Please check if both files exist and the data is separated using tab character.")

    return measurements


class Args:
    folderpath = 'dataornek'
    fileending01 = 'C3-01.ASC'
    fileending02 = 'C3-02.ASC'
    leakage = 0.0003545
    membranearea = 53
    membranethickness = 52.1
    productvolume = 32.313
    temperature = 308.15
    gasconstant = 0.2780405104
    cropinterval = 0.02
    anglemin = 10
    anglemax = 90
    minelement = 10
    kernel_size = 3  # Has to be an odd number
    min_height = 0.5


# C1 productvolume 30.1
# C2 productvolume 33
# C3 productvolume 32.313


def process_measurements(measurements, args):
    x_data = np.asarray(list(measurements.data.keys()))
    y_data = np.asarray(list(measurements.data.values()), dtype=np.float32)

    # Filter Noise
    data = []
    for idx in range(len(y_data[:, 1])):
        min_idx = np.maximum(idx - args.kernel_size // 2 - 1, 0)
        max_idx = np.minimum(idx + args.kernel_size // 2 + 1, len(y_data[:, 1]))
        data.append(np.max(y_data[:, 1][min_idx:max_idx]))
    y_data[:, 1] = np.asarray(data)

    diff_y_data_1 = -np.diff(y_data[:, 1])
    peaks = find_peaks(diff_y_data_1, threshold=args.min_height)
    peaks = np.maximum(0, peaks[0] - 1)
    print(f"DEBUG: Peaks found.len: {len(peaks)}")

    # Plot the data
    plt.plot(x_data, y_data[:, 1])

    # Crop between two red points and apply ransac
    peaks = np.insert(peaks, 0, 0)
    peaks = np.append(peaks, len(x_data) - 2)

    res = {"name": [], "slope": [], "r2": [], "permeability": [], "permeance": []}
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

        # Refit
        poly = np.polyfit(x_segment[inliers], y_segment[inliers], 1)
        slope = poly[0]

        y_inverse = x_segment[inliers] * poly[0] + poly[1]
        r2_score = sklearn.metrics.r2_score(y_inverse, y_segment[inliers])

        if r2_score < 0.9:
            print(f"Measurement {measurements.molecule} has r2 score of {r2_score}. Skipping.")
            continue

        Gas_constant_STP = 0.082 * 76 * 1000 / 22414
        Permeability = np.power(10.0, 10) * (slope - args.leakage) * args.productvolume * \
                       (args.membranethickness / np.power(10, 4)) / ((
                (args.membranearea / 100) * Gas_constant_STP * args.temperature * np.mean(
            y_segment_vals)))
        Permeance = (Permeability / (args.membranethickness / np.power(10, 4))) / 10000

        WIDTH = 4
        PRECISION = 4

        res["name"].append(f"{measurements.molecule}")
        res["slope"].append(f"{slope:{WIDTH}.{PRECISION}}")
        res["r2"].append(f"{r2_score:{WIDTH}.{PRECISION}}")
        res["permeability"].append(f"{Permeability:{WIDTH}.{PRECISION}}")
        res["permeance"].append(f"{Permeance:{WIDTH}.{PRECISION}}")

    print("====================")
    for k, v in res.items():
        txt = "".join(["{: >20} " for v_itm in v])
        print(txt.format(*v))
    print("====================")

    plt.title((f"Original Data of {measurements.molecule}"))
    plt.ylabel("Pressure (mbar)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()


def main():
    args = Args()
    measurements = read_measurements(args)
    print(measurements)
    for measurement in measurements:
        process_measurements(measurement, args)


# This is the main script to parse molecule reading and generate report
if __name__ == '__main__':
    main()
